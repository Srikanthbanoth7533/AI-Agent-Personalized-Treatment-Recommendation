import re
import spacy
import joblib
import os

class NLPEngine:
    def __init__(self, models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.models_dir = models_dir
        self.symptoms_path = os.path.join(models_dir, "symptoms.joblib")
        
        # Load spaCy NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to blank model if not installed yet
            self.nlp = spacy.blank("en")
            
        # Load symptoms list
        if os.path.exists(self.symptoms_path):
            self.symptoms = joblib.load(self.symptoms_path)
        else:
            # Fallback default symptom list if not trained yet
            self.symptoms = []

        # Synonym mapping for common phrases to official symptom names
        self.synonym_map = {
            "itchy": "itching",
            "itch": "itching",
            "rash": "skin_rash",
            "rashes": "skin_rash",
            "spots": "red_spots_over_body",
            "vomit": "vomiting",
            "vomiting": "vomiting",
            "throwing up": "vomiting",
            "puking": "vomiting",
            "nausea": "nausea",
            "sick to stomach": "nausea",
            "fever": "high_fever",
            "high temp": "high_fever",
            "temperature": "high_fever",
            "cough": "cough",
            "coughing": "cough",
            "cold": "continuous_sneezing",
            "sneezing": "continuous_sneezing",
            "sneeze": "continuous_sneezing",
            "shiver": "shivering",
            "shivering": "shivering",
            "chills": "chills",
            "chill": "chills",
            "joint pain": "joint_pain",
            "aching joints": "joint_pain",
            "stomach pain": "stomach_pain",
            "stomach ache": "stomach_pain",
            "belly ache": "stomach_pain",
            "belly pain": "belly_pain",
            "headache": "headache",
            "head ache": "headache",
            "migraine": "headache",
            "fatigue": "fatigue",
            "tired": "fatigue",
            "tiredness": "fatigue",
            "weakness": "fatigue",
            "lethargy": "lethargy",
            "weight loss": "weight_loss",
            "losing weight": "weight_loss",
            "weight gain": "weight_gain",
            "gaining weight": "weight_gain",
            "anxiety": "anxiety",
            "anxious": "anxiety",
            "indigestion": "indigestion",
            "heartburn": "acidity",
            "acidity": "acidity",
            "yellow skin": "yellowish_skin",
            "jaundice": "yellowish_skin",
            "loss of appetite": "loss_of_appetite",
            "not hungry": "loss_of_appetite",
            "poor appetite": "loss_of_appetite",
            "chest pain": "chest_pain",
            "breathlessness": "breathlessness",
            "breathless": "breathlessness",
            "short of breath": "breathlessness",
            "difficulty breathing": "breathlessness",
            "dizziness": "dizziness",
            "dizzy": "dizziness",
            "lightheaded": "dizziness",
            "constipation": "constipation",
            "diarrhea": "diarrhoea",
            "diarrhoea": "diarrhoea",
            "loose stools": "diarrhoea",
            "sweating": "sweating",
            "sweat": "sweating",
            "muscle pain": "muscle_pain",
            "muscle ache": "muscle_pain",
            "sore muscles": "muscle_pain",
        }

    def clean_text(self, text):
        text = text.lower()
        # Remove punctuation except spaces
        text = re.sub(r"[^\w\s\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_symptoms(self, text):
        """
        Extract symptoms from natural language text.
        Returns a dictionary mapping symptom names (e.g. 'joint_pain') to a boolean (True/False).
        """
        # Load symptoms if it wasn't loaded
        if not self.symptoms and os.path.exists(self.symptoms_path):
            self.symptoms = joblib.load(self.symptoms_path)

        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)
        
        # Extract lemmas
        lemmas = [token.lemma_ for token in doc]
        lemma_text = " ".join(lemmas)
        
        matched_symptoms = set()
        
        # 1. Check direct matches / synonyms first
        for key, val in self.synonym_map.items():
            if key in cleaned_text or key in lemma_text:
                matched_symptoms.add(val)
                
        # 2. Check dataset symptoms
        for symptom in self.symptoms:
            # Clean symptom name (e.g. 'joint_pain' -> 'joint pain')
            cleaned_sym = symptom.replace("_", " ")
            if cleaned_sym in cleaned_text or cleaned_sym in lemma_text:
                matched_symptoms.add(symptom)
                
        # Construct boolean vector mapping all symptoms
        symptom_vector = {}
        for symptom in self.symptoms:
            symptom_vector[symptom] = 1 if symptom in matched_symptoms else 0
            
        return {
            "symptoms_found": list(matched_symptoms),
            "vector": symptom_vector
        }

if __name__ == "__main__":
    nlp_eng = NLPEngine()
    text = "I have been experiencing a bad headache, joint pain, and some stomach ache today."
    res = nlp_eng.extract_symptoms(text)
    print("Symptoms found:", res["symptoms_found"])
