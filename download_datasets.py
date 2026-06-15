import urllib.request
import os

DATA_DIR = r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\data"

urls = {
    "Training.csv": "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Training.csv",
    "Testing.csv": "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Testing.csv",
    "description.csv": "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_Description.csv",
    "precautions.csv": "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_precaution.csv"
}

print("Starting dataset downloads...")
for filename, url in urls.items():
    dest_path = os.path.join(DATA_DIR, filename)
    print(f"Downloading {url} -> {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("Dataset downloads completed.")
