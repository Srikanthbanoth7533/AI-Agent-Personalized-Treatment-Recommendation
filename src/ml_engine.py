import os

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    joblib = None
    np = None

# Try importing compiled model fallback
try:
    from src.compiled_model import predict_topk_compiled, SYMPTOMS
except ImportError:
    predict_topk_compiled = None
    SYMPTOMS = []

class MLEngine:
    def __init__(self, models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.models_dir = models_dir
        self.rf_path = os.path.join(models_dir, "random_forest.joblib")
        self.xgb_path = os.path.join(models_dir, "xgboost.joblib")
        
        # We only import DataPreprocessor if ML libs are present
        if HAS_ML_LIBS:
            from src.data_processing import DataPreprocessor
            self.preprocessor = DataPreprocessor()
        else:
            self.preprocessor = None

    def train_and_evaluate(self):
        if not HAS_ML_LIBS:
            raise RuntimeError("Cannot train models without scikit-learn and xgboost installed.")
            
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.model_selection import cross_val_score
        
        print("Loading and preprocessing data...")
        X_train, y_train, X_test, y_test, le = self.preprocessor.process_and_save_encoder()
        self.preprocessor.get_disease_metadata()

        print("\n--- Training RandomForest Baseline ---")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        rf_preds = rf.predict(X_test)
        rf_metrics = self.calculate_metrics(y_test, rf_preds)
        print(f"RandomForest Test Metrics:")
        for k, v in rf_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
        print(f"RandomForest 5-Fold CV Accuracy: {np.mean(rf_cv_scores):.4f} (+/- {np.std(rf_cv_scores):.4f})")
        
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
        
        xgb_preds = xgb.predict(X_test)
        xgb_metrics = self.calculate_metrics(y_test, xgb_preds)
        print(f"XGBoost Test Metrics:")
        for k, v in xgb_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        xgb_cv_scores = cross_val_score(xgb, X_train, y_train, cv=5)
        print(f"XGBoost 5-Fold CV Accuracy: {np.mean(xgb_cv_scores):.4f} (+/- {np.std(xgb_cv_scores):.4f})")
        
        joblib.dump(xgb, self.xgb_path)
        print(f"Saved XGBoost model to {self.xgb_path}")

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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }

    def predict_topk(self, symptom_vector, k=3, model_type="xgboost"):
        # If ML libraries are not available, use compiled model fallback
        if not HAS_ML_LIBS or predict_topk_compiled is not None and not os.path.exists(self.xgb_path):
            if predict_topk_compiled is None:
                raise RuntimeError("No model or compiled inference engine available.")
            
            # Map symptom vector to symptom dict
            symptom_dict = {}
            for i, val in enumerate(symptom_vector):
                if i < len(SYMPTOMS) and val == 1:
                    symptom_dict[SYMPTOMS[i]] = 1
            return predict_topk_compiled(symptom_dict)[:k]
            
        model_path = self.xgb_path if model_type == "xgboost" else self.rf_path
        if not os.path.exists(model_path):
            # Fall back to compiled if joblib missing
            if predict_topk_compiled is not None:
                symptom_dict = {}
                for i, val in enumerate(symptom_vector):
                    if i < len(SYMPTOMS) and val == 1:
                        symptom_dict[SYMPTOMS[i]] = 1
                return predict_topk_compiled(symptom_dict)[:k]
            raise FileNotFoundError(f"Model not found at {model_path}.")
            
        model = joblib.load(model_path)
        le = joblib.load(os.path.join(self.models_dir, "label_encoder.joblib"))
        
        X = np.array(symptom_vector).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        
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
