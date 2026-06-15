import pandas as pd
import numpy as np
import joblib
import os

class DataPreprocessor:
    def __init__(self, data_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\data",
                 models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.encoder_path = os.path.join(models_dir, "label_encoder.joblib")
        
        self.train_path = os.path.join(data_dir, "Training.csv")
        self.test_path = os.path.join(data_dir, "Testing.csv")
        self.desc_path = os.path.join(data_dir, "description.csv")
        self.prec_path = os.path.join(data_dir, "precautions.csv")

    def load_raw_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Load description and precautions (they have no header in raw CSV)
        desc_df = pd.read_csv(self.desc_path, header=None, names=["Disease", "Description"])
        prec_df = pd.read_csv(self.prec_path, header=None, 
                             names=["Disease", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"])
        
        # Strip whitespaces from strings
        desc_df["Disease"] = desc_df["Disease"].str.strip()
        desc_df["Description"] = desc_df["Description"].str.strip()
        
        prec_df["Disease"] = prec_df["Disease"].str.strip()
        for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            prec_df[col] = prec_df[col].str.strip()
            
        train_df["prognosis"] = train_df["prognosis"].str.strip()
        test_df["prognosis"] = test_df["prognosis"].str.strip()

        return train_df, test_df, desc_df, prec_df

    def process_and_save_encoder(self):
        train_df, test_df, desc_df, prec_df = self.load_raw_data()
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        # Fit encoder on prognosis (make sure to combine train and test just in case)
        all_diseases = pd.concat([train_df["prognosis"], test_df["prognosis"]]).unique()
        le.fit(all_diseases)
        
        # Save LabelEncoder
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(le, self.encoder_path)
        
        # Get X and y
        X_train = train_df.drop(columns=["prognosis"])
        y_train = le.transform(train_df["prognosis"])
        
        X_test = test_df.drop(columns=["prognosis"])
        y_test = le.transform(test_df["prognosis"])
        
        # Save symptom names list
        symptoms = X_train.columns.tolist()
        joblib.dump(symptoms, os.path.join(self.models_dir, "symptoms.joblib"))
        
        return X_train, y_train, X_test, y_test, le

    def get_disease_metadata(self):
        _, _, desc_df, prec_df = self.load_raw_data()
        
        # Merge description and precautions on Disease
        merged = pd.merge(desc_df, prec_df, on="Disease", how="outer")
        
        metadata = {}
        for _, row in merged.iterrows():
            disease = row["Disease"]
            if pd.isna(disease):
                continue
            
            precautions = []
            for p in [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]]:
                if pd.notna(p) and str(p).strip() != "":
                    precautions.append(str(p).strip())
                    
            metadata[disease] = {
                "description": row["Description"] if pd.notna(row["Description"]) else "No description available.",
                "precautions": precautions
            }
        
        # Save metadata dict
        joblib.dump(metadata, os.path.join(self.models_dir, "disease_metadata.joblib"))
        return metadata

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test, le = preprocessor.process_and_save_encoder()
    metadata = preprocessor.get_disease_metadata()
    print("Preprocessing completed!")
    print(f"Number of symptoms: {X_train.shape[1]}")
    print(f"Number of diseases: {len(le.classes_)}")
