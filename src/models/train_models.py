"""
Smart OBD-II Electrical Diagnostic System
Module: Deep AI Model Training Pipeline (XGBoost + Random Forest + Isolation Forest)

Trains:
1. Isolation Forest: Unsupervised Anomaly Detection for Out-of-Distribution Electrical Telemetry
2. XGBoost Multi-Class Classifier: State-of-the-Art Gradient Boosted Trees for 4-Class Automotive Diagnostics
3. Random Forest Classifier: Robust Ensemble Backup
Evaluates Precision, Recall, Macro F1-Score, Confusion Matrix, and saves production artifacts.
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineering import ElectricalFeatureExtractor
from src.generator.synthetic_generator import PATTERN_NAMES, SimulationPattern


def train_deep_models(
    raw_data_path: Path,
    models_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42,
):
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🚀 DEEP OBD-II ELECTRICAL AI MODEL TRAINING PIPELINE (XGBOOST + ENSEMBLE) ⚡")
    print("=" * 75)
    print(f"📖 Loading raw dataset from: {raw_data_path}")

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Run generate_dataset.py first!")

    df = pd.read_csv(raw_data_path)
    print(f"📊 Total Raw Telemetry Records: {len(df):,} rows across {df['session_id'].nunique()} vehicle sessions.")

    # 1. Feature Extraction
    print("\n⚙️ Extracting domain-specific electrical diagnostic features...")
    X, y_multi, y_anomaly = ElectricalFeatureExtractor.transform_dataset_to_feature_matrix(df)
    print(f"✅ Feature Matrix Shape: {X.shape[0]} sessions x {X.shape[1]} features.")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test, y_anom_train, y_anom_test = train_test_split(
        X, y_multi, y_anomaly, test_size=test_size, random_state=random_state, stratify=y_multi
    )
    print(f"🔀 Training Sessions: {len(X_train)} | Testing Sessions: {len(X_test)}")

    # 3. Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # MODEL 1: Isolation Forest (Anomaly Detector)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("🌲 [1/3] Training Isolation Forest (Unsupervised Anomaly Detector)...")
    
    X_normal_train = X_train_scaled[y_anom_train == 0]
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=random_state,
    )
    iso_forest.fit(X_normal_train)

    raw_preds = iso_forest.predict(X_test_scaled)
    iso_preds = np.where(raw_preds == -1, 1, 0)
    anom_acc = accuracy_score(y_anom_test, iso_preds)
    anom_f1 = f1_score(y_anom_test, iso_preds, pos_label=1)
    print(f"✅ Isolation Forest Anomaly Detection Accuracy: {anom_acc * 100:.2f}% | F1-Score: {anom_f1:.4f}")

    # -------------------------------------------------------------
    # MODEL 2: XGBoost Multi-Class Classifier
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("⚡ [2/3] Training XGBoost State-of-the-Art Multi-Class Classifier...")

    xgb_classifier = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        random_state=random_state,
        eval_metric="mlogloss",
    )
    xgb_classifier.fit(X_train, y_train)

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=skf, scoring="f1_macro")
    print(f"📊 5-Fold Stratified CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred_xgb = xgb_classifier.predict(X_test)
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb, average="macro")
    print(f"🎯 XGBoost Test Accuracy: {xgb_acc * 100:.2f}% | Macro F1-Score: {xgb_f1:.4f}")

    # -------------------------------------------------------------
    # MODEL 3: Random Forest Classifier (Ensemble)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("🌳 [3/3] Training Random Forest Diagnostic Classifier...")

    rf_classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_split=2,
        random_state=random_state,
    )
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"🎯 Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # Detailed Classification Report
    target_names = [f"Class {p.value}: {PATTERN_NAMES[p]['en']}" for p in SimulationPattern]
    report = classification_report(y_test, y_pred_xgb, target_names=target_names, digits=4)
    print("\n📋 Detailed XGBoost Classification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred_xgb)
    print("🔢 Confusion Matrix:")
    print(cm)

    # Feature Importances
    print("\n🌟 Top Feature Importances (XGBoost):")
    xgb_importances = xgb_classifier.feature_importances_
    indices = np.argsort(xgb_importances)[::-1]
    for rank, idx in enumerate(indices, 1):
        print(f"   {rank:02d}. {X.columns[idx]:<28} : {xgb_importances[idx] * 100:.2f}%")

    # -------------------------------------------------------------
    # 4. Save Production Artifacts
    # -------------------------------------------------------------
    iso_path = models_dir / "isolation_forest.joblib"
    xgb_path = models_dir / "electrical_classifier.joblib"  # Primary classifier
    rf_path = models_dir / "rf_backup.joblib"
    scaler_path = models_dir / "scaler.joblib"
    meta_path = models_dir / "features_meta.joblib"

    joblib.dump(iso_forest, iso_path)
    joblib.dump(xgb_classifier, xgb_path)
    joblib.dump(rf_classifier, rf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(ElectricalFeatureExtractor.FEATURE_COLUMNS, meta_path)

    print("\n" + "=" * 75)
    print(f"💾 Production AI Models successfully saved to: {models_dir}")
    print(f"   - {iso_path.name}")
    print(f"   - {xgb_path.name} (XGBoost Primary)")
    print(f"   - {rf_path.name} (RandomForest Backup)")
    print(f"   - {scaler_path.name}")
    print(f"   - {meta_path.name}")
    print("=" * 75)


def main():
    raw_csv = PROJECT_ROOT / "data" / "raw" / "obd_electrical_telemetry_raw.csv"
    models_dir = PROJECT_ROOT / "models"
    train_deep_models(raw_csv, models_dir)


if __name__ == "__main__":
    main()
