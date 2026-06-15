class RiskEngine:
    def __init__(self):
        # List of diseases that are considered high or medium risk
        self.high_risk_diseases = [
            "Malaria", "Tuberculosis", "Pneumonia", "Heart attack", 
            "Dengue", "Typhoid", "Hepatitis B", "Hepatitis C", 
            "Hepatitis D", "Hepatitis E", "Chronic cholestasis", "Diabetes"
        ]
        self.medium_risk_diseases = [
            "Hypertension", "Jaundice", "Migraine", "Arthritis", 
            "Gastroesophageal reflux disease", "Peptic ulcer disease", 
            "Varicose veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia"
        ]
        
        # Symptoms that trigger high/medium risk directly
        self.high_risk_symptoms = [
            "chest_pain", "breathlessness", "slurred_speech", 
            "weakness_of_one_body_side", "blood_in_sputum", 
            "coma", "stomach_bleeding", "toxic_look_(typhos)"
        ]
        self.medium_risk_symptoms = [
            "high_fever", "vomiting", "yellowish_skin", 
            "loss_of_balance", "dizziness", "unsteadiness", 
            "acute_liver_failure", "neck_pain"
        ]

    def assess_risk(self, predicted_disease, confidence, symptoms_found, biomarkers=None):
        """
        Assess overall patient risk based on prediction, symptoms, and biomarkers.
        Returns:
            dict containing:
                "level": "Low" | "Medium" | "High"
                "reasons": list of strings explaining why
        """
        reasons = []
        level = "Low"
        
        # 1. Check Biomarkers
        if biomarkers:
            glucose = biomarkers.get("glucose")
            hemo = biomarkers.get("hemoglobin")
            chol = biomarkers.get("cholesterol")
            
            if glucose is not None:
                if glucose < 50 or glucose > 250:
                    level = "High"
                    reasons.append(f"Critical glucose level ({glucose} mg/dL).")
                elif glucose < 70 or glucose > 120:
                    level = max(level, "Medium", key=self._risk_key)
                    reasons.append(f"Abnormal glucose level ({glucose} mg/dL).")
                    
            if hemo is not None:
                if hemo < 8.0:
                    level = "High"
                    reasons.append(f"Critical hemoglobin level ({hemo} g/dL).")
                elif hemo < 11.5 or hemo > 18.0:
                    level = max(level, "Medium", key=self._risk_key)
                    reasons.append(f"Abnormal hemoglobin level ({hemo} g/dL).")
                    
            if chol is not None:
                if chol >= 300:
                    level = "High"
                    reasons.append(f"Critically high cholesterol ({chol} mg/dL).")
                elif chol >= 240:
                    level = max(level, "Medium", key=self._risk_key)
                    reasons.append(f"High cholesterol level ({chol} mg/dL).")

        # 2. Check Symptoms
        for symptom in symptoms_found:
            # Map symptom with spaces back to underscore if needed
            sym_key = symptom.replace(" ", "_")
            if sym_key in self.high_risk_symptoms:
                level = "High"
                reasons.append(f"Presence of critical symptom: {symptom.replace('_', ' ')}.")
            elif sym_key in self.medium_risk_symptoms:
                level = max(level, "Medium", key=self._risk_key)
                reasons.append(f"Presence of concerning symptom: {symptom.replace('_', ' ')}.")

        # 3. Check Predicted Disease
        if predicted_disease in self.high_risk_diseases:
            if confidence > 0.4:
                level = "High"
                reasons.append(f"Predicted disease '{predicted_disease}' has high clinical severity.")
            else:
                level = max(level, "Medium", key=self._risk_key)
                reasons.append(f"Possible risk of high-severity disease: '{predicted_disease}' (low confidence).")
        elif predicted_disease in self.medium_risk_diseases:
            if confidence > 0.4:
                level = max(level, "Medium", key=self._risk_key)
                reasons.append(f"Predicted disease '{predicted_disease}' has moderate clinical severity.")

        # Default reason if low risk
        if not reasons:
            reasons.append("Symptoms and signs indicate a low-risk, self-limiting condition. Please monitor your health.")

        return {
            "level": level,
            "reasons": reasons
        }

    def _risk_key(self, level):
        levels = {"Low": 0, "Medium": 1, "High": 2}
        return levels.get(level, 0)

if __name__ == "__main__":
    engine = RiskEngine()
    assessment = engine.assess_risk("Malaria", 0.85, ["high_fever", "vomiting"])
    print("Assessment:", assessment)
