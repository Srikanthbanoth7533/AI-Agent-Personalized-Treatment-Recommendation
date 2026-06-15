import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.data_processing import DataPreprocessor

class MLEngine:
    def __init__(self, models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.models_dir = models_dir
        self.rf_path = os.path.join(models_dir, "random_forest.joblib")
        self.xgb_path = os.path.join(models_dir, "xgboost.joblib")
        self.preprocessor = DataPreprocessor()

    def train_and_evaluate(self):
        print("Loading and preprocessing data...")
        X_train, y_train, X_test, y_test, le = self.preprocessor.process_and_save_encoder()
        self.preprocessor.get_disease_metadata() # also save disease metadata

        print("\n--- Training RandomForest Baseline ---")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate RF
        rf_preds = rf.predict(X_test)
        rf_metrics = self.calculate_metrics(y_test, rf_preds)
        print(f"RandomForest Test Metrics:")
        for k, v in rf_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
        print(f"RandomForest 5-Fold CV Accuracy: {np.mean(rf_cv_scores):.4f} (+/- {np.std(rf_cv_scores):.4f})")
        
        # Save RF
        joblib.dump(rf, self.rf_path)
        print(f"Saved RandomForest model to {self.rf_path}")

        print("\n--- Training XGBoost Optimized ---")
        xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42, 
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        
        # Evaluate XGB
        xgb_preds = xgb.predict(X_test)
        xgb_metrics = self.calculate_metrics(y_test, xgb_preds)
        print(f"XGBoost Test Metrics:")
        for k, v in xgb_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        xgb_cv_scores = cross_val_score(xgb, X_train, y_train, cv=5)
        print(f"XGBoost 5-Fold CV Accuracy: {np.mean(xgb_cv_scores):.4f} (+/- {np.std(xgb_cv_scores):.4f})")
        
        # Save XGB
        joblib.dump(xgb, self.xgb_path)
        print(f"Saved XGBoost model to {self.xgb_path}")

        # Save metrics summary
        metrics_summary = {
            "random_forest": {
                **rf_metrics,
                "cv_mean": float(np.mean(rf_cv_scores)),
                "cv_std": float(np.std(rf_cv_scores))
            },
            "xgboost": {
                **xgb_metrics,
                "cv_mean": float(np.mean(xgb_cv_scores)),
                "cv_std": float(np.std(xgb_cv_scores))
            }
        }
        joblib.dump(metrics_summary, os.path.join(self.models_dir, "metrics_summary.joblib"))
        
        return rf, xgb, metrics_summary

    def calculate_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }

    def predict_topk(self, symptom_vector, k=3, model_type="xgboost"):
        """
        Predict top-k diseases with confidence scores.
        symptom_vector: 1D array/list of 0s and 1s matching training columns.
        """
        # Load model and encoder
        model_path = self.xgb_path if model_type == "xgboost" else self.rf_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
            
        model = joblib.load(model_path)
        le = joblib.load(self.preprocessor.encoder_path)
        
        # Ensure 2D shape for input
        X = np.array(symptom_vector).reshape(1, -1)
        
        # Get probability distribution
        probs = model.predict_proba(X)[0]
        
        # Sort and get top-k
        topk_indices = np.argsort(probs)[::-1][:k]
        topk_probs = probs[topk_indices]
        topk_diseases = le.inverse_transform(topk_indices)
        
        results = []
        for disease, prob in zip(topk_diseases, topk_probs):
            results.append({
                "disease": disease,
                "confidence": float(prob)
            })
            
        return results

if __name__ == "__main__":
    engine = MLEngine()
    engine.train_and_evaluate()
