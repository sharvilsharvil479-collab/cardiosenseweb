"""
CardioSense — Updated Training Script
Teri original ip-2.py pe based hai, sirf model tuning better ki hai
taaki 95%+ accuracy achieve ho sake.

Changes:
- SMOTE se class imbalance handle kiya
- Feature engineering add ki (important interactions)
- Better hyperparameter grids
- Voting Ensemble add kiya
- Scaler bhi save kiya (zaroori hai production ke liye)
"""

import numpy as np
import pandas as pd
import warnings, pickle
warnings.filterwarnings('ignore')

# ── Step 1: Data Load ─────────────────────────────────────────────────
# heart.csv path apna update karo
heart_df = pd.read_csv(r"C:\Users\Sharvil-001\Downloads\cardiosense_v3_final\cardiosense\heart.csv (2).xls")   # ← apna path likho
print("Dataset shape:", heart_df.shape)
print(heart_df.head())

# ── Step 2: EDA (original se same) ───────────────────────────────────
print("\nNull values:\n", heart_df.isnull().sum())
print("\nDuplicates:", heart_df.duplicated().sum())

# ── Step 3: Categorical Encoding ─────────────────────────────────────
cat_col = heart_df.select_dtypes(include='object').columns
for col in cat_col:
    print(f"\n{col}: {heart_df[col].unique()} → {list(range(heart_df[col].nunique()))}")
    heart_df[col].replace(heart_df[col].unique(), range(heart_df[col].nunique()), inplace=True)

# ── Step 4: Missing Value Treatment ──────────────────────────────────
from sklearn.impute import KNNImputer

heart_df['Cholesterol'].replace(0, np.nan, inplace=True)
heart_df['RestingBP'].replace(0, np.nan, inplace=True)

imputer = KNNImputer(n_neighbors=5)
after_impute = imputer.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute, columns=heart_df.columns)

# Integer columns
cols_int = heart_df.columns.drop('Oldpeak')
heart_df[cols_int] = heart_df[cols_int].astype('int32')
print("\nAfter cleaning:\n", heart_df.info())

# ── Step 5: Feature Engineering (accuracy boost) ─────────────────────
# Ye naye features model ko zyada pattern pakadne mein madad karte hain
heart_df['AgeMaxHR_ratio']     = heart_df['Age'] / (heart_df['MaxHR'] + 1)
heart_df['OldpeakSlope']       = heart_df['Oldpeak'] * heart_df['ST_Slope']
heart_df['AnginaChestPain']    = heart_df['ExerciseAngina'] * heart_df['ChestPainType']
heart_df['HighBP_flag']        = (heart_df['RestingBP'] > 140).astype(int)
heart_df['HighChol_flag']      = (heart_df['Cholesterol'] > 240).astype(int)
heart_df['LowHR_flag']         = (heart_df['MaxHR'] < 130).astype(int)
heart_df['Age_Risk']           = ((heart_df['Age'] > 55) & (heart_df['Sex'] == 0)).astype(int)
print("\nFeatures after engineering:", heart_df.shape[1], "features")

# ── Step 6: Train-Test Split ──────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = heart_df.drop('HeartDisease', axis=1)
y = heart_df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc  = scaler.transform(x_test)

print(f"\nTrain size: {len(x_train)}, Test size: {len(x_test)}")
print("Class distribution:\n", y_train.value_counts())

# ── Step 7: SMOTE (class imbalance fix) ──────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    x_train_res, y_train_res = sm.fit_resample(x_train_sc, y_train)
    print(f"\nAfter SMOTE — Train size: {len(x_train_res)}")
    use_smote = True
except ImportError:
    print("\nSMOTE nahi mila (imbalanced-learn install karo): pip install imbalanced-learn")
    print("SMOTE ke bina chal rahe hain...")
    x_train_res, y_train_res = x_train_sc, y_train
    use_smote = False

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score

# ── Step 8: Models Train Karo ─────────────────────────────────────────

print("\n" + "="*60)
print("MODEL TRAINING SHURU")
print("="*60)

# --- 8a. Logistic Regression ---
from sklearn.linear_model import LogisticRegression

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight='balanced'),
    {'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
     'solver': ['liblinear', 'lbfgs'],
     'penalty': ['l2']},
    cv=10, scoring='accuracy', n_jobs=-1
)
lr_grid.fit(x_train_res, y_train_res)
lr_best  = lr_grid.best_estimator_
lr_pred  = lr_best.predict(x_test_sc)
lr_acc   = accuracy_score(y_test, lr_pred)
lr_auc   = roc_auc_score(y_test, lr_best.predict_proba(x_test_sc)[:,1])
print(f"\n[Logistic Regression]")
print(f"  Accuracy : {lr_acc*100:.2f}%")
print(f"  AUC-ROC  : {lr_auc:.4f}")
print(f"  Best params: {lr_grid.best_params_}")

# --- 8b. SVM ---
from sklearn.svm import SVC

svm_grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),
    {'C': [0.5, 1, 5, 10, 20, 50],
     'gamma': ['scale', 0.01, 0.05, 0.1],
     'kernel': ['rbf']},
    cv=10, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(x_train_res, y_train_res)
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(x_test_sc)
svm_acc  = accuracy_score(y_test, svm_pred)
print(f"\n[SVM]")
print(f"  Accuracy : {svm_acc*100:.2f}%")
print(f"  Best params: {svm_grid.best_params_}")

# --- 8c. Decision Tree ---
from sklearn.tree import DecisionTreeClassifier

dt_grid = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced'),
    {'max_depth': [3, 4, 5, 6, 7],
     'min_samples_split': [2, 3, 5],
     'min_samples_leaf': [1, 2, 3],
     'criterion': ['gini', 'entropy'],
     'random_state': [42]},
    cv=10, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(x_train_res, y_train_res)
dt_best  = dt_grid.best_estimator_
dt_pred  = dt_best.predict(x_test_sc)
dt_acc   = accuracy_score(y_test, dt_pred)
print(f"\n[Decision Tree]")
print(f"  Accuracy : {dt_acc*100:.2f}%")
print(f"  Best params: {dt_grid.best_params_}")

# --- 8d. Random Forest ---
from sklearn.ensemble import RandomForestClassifier

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {'n_estimators': [300, 500, 800],
     'max_depth': [8, 12, 15, 20, None],
     'min_samples_split': [2, 3],
     'min_samples_leaf': [1, 2],
     'max_features': ['sqrt', 'log2']},
    cv=10, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(x_train_res, y_train_res)
rf_best  = rf_grid.best_estimator_
rf_pred  = rf_best.predict(x_test_sc)
rf_acc   = accuracy_score(y_test, rf_pred)
rf_auc   = roc_auc_score(y_test, rf_best.predict_proba(x_test_sc)[:,1])
print(f"\n[Random Forest]")
print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  AUC-ROC  : {rf_auc:.4f}")
print(f"  Best params: {rf_grid.best_params_}")

# --- 8e. XGBoost ---
from xgboost import XGBClassifier

# scale_pos_weight for imbalance
pos_w = (y_train == 0).sum() / (y_train == 1).sum() if not use_smote else 1

xgb = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=1.2,
    min_child_weight=3,
    scale_pos_weight=pos_w,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_grid = GridSearchCV(
    xgb,
    {'max_depth': [4, 5, 6],
     'learning_rate': [0.01, 0.02, 0.05],
     'n_estimators': [500, 800]},
    cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(x_train_res, y_train_res)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(x_test_sc)
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_auc  = roc_auc_score(y_test, xgb_best.predict_proba(x_test_sc)[:,1])
print(f"\n[XGBoost]")
print(f"  Accuracy : {xgb_acc*100:.2f}%")
print(f"  AUC-ROC  : {xgb_auc:.4f}")
print(f"  Best params: {xgb_grid.best_params_}")

# --- 8f. Gradient Boosting ---
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.85,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)
gb.fit(x_train_res, y_train_res)
gb_pred = gb.predict(x_test_sc)
gb_acc  = accuracy_score(y_test, gb_pred)
print(f"\n[Gradient Boosting]")
print(f"  Accuracy : {gb_acc*100:.2f}%")

# --- 8g. Voting Ensemble (best 3 combine karo) ---
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf',  rf_best),
        ('xgb', xgb_best),
        ('gb',  gb),
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(x_train_res, y_train_res)
vc_pred = voting_clf.predict(x_test_sc)
vc_acc  = accuracy_score(y_test, vc_pred)
vc_auc  = roc_auc_score(y_test, voting_clf.predict_proba(x_test_sc)[:,1])
print(f"\n[Voting Ensemble (RF+XGB+GB)]")
print(f"  Accuracy : {vc_acc*100:.2f}%")
print(f"  AUC-ROC  : {vc_auc:.4f}")

# ── Step 9: Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL ACCURACY SUMMARY")
print("="*60)
results = {
    'Logistic Regression' : lr_acc,
    'SVM'                 : svm_acc,
    'Decision Trees'      : dt_acc,
    'Random Forest'       : rf_acc,
    'XGBoost'             : xgb_acc,
    'Gradient Boosting'   : gb_acc,
    'Voting Ensemble'     : vc_acc,
}
for name, acc in sorted(results.items(), key=lambda x:-x[1]):
    bar = '█' * int(acc*30)
    print(f"  {name:25s} {acc*100:6.2f}%  {bar}")

# ── Step 10: Save Models ──────────────────────────────────────────────
print("\nModels save kar rahe hain...")

# Important: scaler bhi save karo!
pickle.dump(scaler,    open('scaler.pkl', 'wb'))
pickle.dump(lr_best,   open('LogisticRegression.pkl', 'wb'))
pickle.dump(svm_best,  open('svm.pkl', 'wb'))
pickle.dump(dt_best,   open('DecisionTreeClassifier.pkl', 'wb'))
pickle.dump(rf_best,   open('RandomForestClassifier.pkl', 'wb'))
pickle.dump(xgb_best,  open('XGBClassifier.pkl', 'wb'))
pickle.dump(gb,        open('GradientBoostingClassifier.pkl', 'wb'))

print("\n✅ Sab models save ho gaye:")
print("  - scaler.pkl (ZAROORI - prediction ke liye)")
print("  - LogisticRegression.pkl")
print("  - svm.pkl")
print("  - DecisionTreeClassifier.pkl")
print("  - RandomForestClassifier.pkl")
print("  - XGBClassifier.pkl")
print("  - GradientBoostingClassifier.pkl")
print("\nAb sab .pkl files cardiosense/ folder mein daal do aur python app.py chalaao!")
