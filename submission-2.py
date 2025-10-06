from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =====================
# 1. Load Data
# =====================
# Note: Update file paths if necessary
train = pd.read_csv("/kaggle/input/iisc-umc-301-kaggle-competition-1/train.csv")
test = pd.read_csv("/kaggle/input/iisc-umc-301-kaggle-competition-1/test.csv")
target = "song_popularity"
id_col = "id"

X = train.drop([target, id_col], axis=1)
y = train[target]
X_test = test.drop([id_col], axis=1)

# =====================
# 2. Feature Engineering
# =====================
def feature_engineering(df):
    df = df.copy()
    if "song_duration_ms" in df.columns:
        df["log_duration"] = np.log1p(df["song_duration_ms"])
    if "loudness" in df.columns:
        df["loudness_scaled"] = (df["loudness"] - df["loudness"].mean()) / df["loudness"].std()
    if "tempo" in df.columns:
        df["tempo_bin"] = pd.qcut(df["tempo"], q=5, labels=False, duplicates="drop")
    if "danceability" in df.columns and "energy" in df.columns:
        df["dance_energy"] = df["danceability"] * df["energy"]
    return df

X = feature_engineering(X)
X_test = feature_engineering(X_test)

# Handle categorical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    # Use transform on test data to handle potential new categories gracefully
    X_test[col] = X_test[col].map(lambda s: s if s in le.classes_ else -1) # Handle unseen labels
    X_test[col] = le.transform(X_test[col].astype(str))

X = X.apply(pd.to_numeric, errors="ignore")
X_test = X_test.apply(pd.to_numeric, errors="ignore")


# =====================
# 3. Out-of-Fold Training
# =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Initialize Prediction Arrays for all models ---
oof_preds_xgb = np.zeros(len(X))
oof_preds_lgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))
oof_preds_rf  = np.zeros(len(X))
oof_preds_et  = np.zeros(len(X))
oof_preds_ada = np.zeros(len(X))

test_preds_xgb = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_rf  = np.zeros(len(X_test))
test_preds_et  = np.zeros(len(X_test))
test_preds_ada = np.zeros(len(X_test))


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- XGBoost ---
    print("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=42, use_label_encoder=False
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / kf.n_splits

    # --- LightGBM ---
    print("Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.01, num_leaves=64,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_lgb.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / kf.n_splits

    # --- CatBoost ---
    print("Training CatBoost...")
    model_cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=8,
        eval_metric="AUC", random_seed=42, verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / kf.n_splits

    # --- Imputation for RF, ET, and AdaBoost ---
    imp = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
    X_val_imp = pd.DataFrame(imp.transform(X_val), columns=X_val.columns)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # --- RandomForest & ExtraTrees ---
    print("Training RF and ET...")
    model_rf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
    model_rf.fit(X_tr_imp, y_tr)
    oof_preds_rf[val_idx] = model_rf.predict_proba(X_val_imp)[:, 1]
    test_preds_rf += model_rf.predict_proba(X_test_imp)[:, 1] / kf.n_splits

    model_et = ExtraTreesClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
    model_et.fit(X_tr_imp, y_tr)
    oof_preds_et[val_idx] = model_et.predict_proba(X_val_imp)[:, 1]
    test_preds_et += model_et.predict_proba(X_test_imp)[:, 1] / kf.n_splits

    # --- Single High-Performing AdaBoost Model ---
    print("Training AdaBoost...")
    model_ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=4),
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42
    )
    model_ada.fit(X_tr_imp, y_tr)
    oof_preds_ada[val_idx] = model_ada.predict_proba(X_val_imp)[:, 1]
    test_preds_ada += model_ada.predict_proba(X_test_imp)[:, 1] / kf.n_splits

    print(f"Fold {fold} done.")

# =====================
# 4. Meta-Model (Stacking)
# =====================

# Combine predictions from all 6 models
all_oof_preds = [
    oof_preds_xgb,
    oof_preds_lgb,
    oof_preds_cat,
    oof_preds_rf,
    oof_preds_et,
    oof_preds_ada
]
all_test_preds = [
    test_preds_xgb,
    test_preds_lgb,
    test_preds_cat,
    test_preds_rf,
    test_preds_et,
    test_preds_ada
]

# Stack them horizontally for the meta-model
stacked_train = np.vstack(all_oof_preds).T
stacked_test = np.vstack(all_test_preds).T

print(f"\nTotal number of base models: {stacked_train.shape[1]}")

meta_model = LogisticRegression(max_iter=1000, solver="lbfgs")
meta_model.fit(stacked_train, y)
final_preds = meta_model.predict_proba(stacked_test)[:, 1]

# =====================
# 5. Evaluate OOF
# =====================
auc_xgb = roc_auc_score(y, oof_preds_xgb)
auc_lgb = roc_auc_score(y, oof_preds_lgb)
auc_cat = roc_auc_score(y, oof_preds_cat)
auc_rf  = roc_auc_score(y, oof_preds_rf)
auc_et  = roc_auc_score(y, oof_preds_et)
auc_ada = roc_auc_score(y, oof_preds_ada)
auc_stack = roc_auc_score(y, meta_model.predict_proba(stacked_train)[:, 1])

print("\nModel AUCs:")
print(f"XGBoost: {auc_xgb:.4f}")
print(f"LightGBM: {auc_lgb:.4f}")
print(f"CatBoost: {auc_cat:.4f}")
print(f"RandomForest: {auc_rf:.4f}")
print(f"ExtraTrees: {auc_et:.4f}")
print(f"AdaBoost: {auc_ada:.4f}")
print(f"Stacked Ensemble (Meta LR): {auc_stack:.4f}")

# =====================
# 6. Submission
# =====================
submission = pd.DataFrame({
    id_col: test[id_col],
    target: (final_preds * 9999).astype(int)
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSubmission Successful")