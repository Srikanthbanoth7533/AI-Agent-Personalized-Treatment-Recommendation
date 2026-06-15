import re
import os
import pdfplumber
import PyPDF2
from PIL import Image

try:
    import pytesseract
    # Try to find standard installation path for tesseract on Windows
    std_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(std_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = std_tesseract_path
except ImportError:
    pytesseract = None

class ReportAnalyzer:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path):
        text = ""
        # Method 1: Use pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
        # Method 2 fallback: Use PyPDF2
        if not text.strip():
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
                
        return text

    def extract_text_from_image(self, file_path):
        if pytesseract is None:
            return "[Error: pytesseract library not installed] Fallback: Please ensure tesseract OCR is installed."
            
        try:
            img = Image.open(file_path)
            # Try to run OCR
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"pytesseract OCR failed: {e}")
            # If tesseract is not in path, return a clear message
            return f"[Error: Tesseract OCR execution failed: {e}]. Please install Tesseract on your system."

    def parse_biomarkers(self, text):
        """
        Extract Glucose, Hemoglobin, Cholesterol from text using regex.
        """
        # Normal ranges:
        # Glucose: 70 - 100 mg/dL (fasting)
        # Hemoglobin: 12 - 17.5 g/dL
        # Cholesterol: < 200 mg/dL
        
        glucose = None
        hemoglobin = None
        cholesterol = None

        # Regex patterns
        glucose_pattern = r"(?i)(?:glucose|fasting\s+glucose|sugar|glu)\b[:\s\-]*(\d+(?:\.\d+)?)"
        hemo_pattern = r"(?i)(?:hemoglobin|hb|hgb)\b[:\s\-]*(\d+(?:\.\d+)?)"
        chol_pattern = r"(?i)(?:cholesterol|chol|total\s+cholesterol)\b[:\s\-]*(\d+(?:\.\d+)?)"

        # Search patterns
        glu_match = re.search(glucose_pattern, text)
        if glu_match:
            try:
                glucose = float(glu_match.group(1))
            except ValueError:
                pass

        hemo_match = re.search(hemo_pattern, text)
        if hemo_match:
            try:
                hemoglobin = float(hemo_match.group(1))
            except ValueError:
                pass

        chol_match = re.search(chol_pattern, text)
        if chol_match:
            try:
                cholesterol = float(chol_match.group(1))
            except ValueError:
                pass

        return {
            "glucose": glucose,
            "hemoglobin": hemoglobin,
            "cholesterol": cholesterol
        }

    def generate_report_summary(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".pdf"]:
            text = self.extract_text_from_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            text = self.extract_text_from_image(file_path)
        else:
            text = ""
            
        markers = self.parse_biomarkers(text)
        
        # Build summary explanations
        glucose_val = markers["glucose"]
        hemo_val = markers["hemoglobin"]
        chol_val = markers["cholesterol"]
        
        findings = []
        status = "Normal"
        
        if glucose_val is not None:
            if glucose_val < 70:
                findings.append(f"Low glucose levels ({glucose_val} mg/dL) indicating potential hypoglycemia.")
                status = "Alert"
            elif glucose_val > 100:
                findings.append(f"Elevated glucose levels ({glucose_val} mg/dL) indicating potential hyperglycemia/diabetes risk.")
                status = "Alert"
            else:
                findings.append(f"Glucose level is normal ({glucose_val} mg/dL).")
                
        if hemo_val is not None:
            if hemo_val < 12.0:
                findings.append(f"Low hemoglobin ({hemo_val} g/dL) indicating potential anemia.")
                status = "Alert"
            elif hemo_val > 18.0:
                findings.append(f"High hemoglobin ({hemo_val} g/dL) indicating potential polycythemia.")
            else:
                findings.append(f"Hemoglobin level is normal ({hemo_val} g/dL).")
                
        if chol_val is not None:
            if chol_val >= 240:
                findings.append(f"High total cholesterol ({chol_val} mg/dL) indicating increased cardiovascular risk.")
                status = "Alert"
            elif chol_val >= 200:
                findings.append(f"Borderline high cholesterol ({chol_val} mg/dL).")
            else:
                findings.append(f"Cholesterol level is normal ({chol_val} mg/dL).")

        if not findings:
            findings.append("No common health biomarkers (Glucose, Hemoglobin, Cholesterol) were detected in the report text.")

        return {
            "text_preview": text[:500] + ("..." if len(text) > 500 else ""),
            "biomarkers": markers,
            "findings": findings,
            "status": status
        }

if __name__ == "__main__":
    analyzer = ReportAnalyzer()
    # Test parse_biomarkers
    test_text = "Patient Report: Fasting Glucose: 112 mg/dL. Hemoglobin: 11.5 g/dL. Total Cholesterol: 210 mg/dL"
    res = analyzer.parse_biomarkers(test_text)
    print("Parsed values:", res)
