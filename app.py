import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Set Page Configuration
st.set_page_config(
    page_title="Aegis AI - Clinical Agent Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Inject
st.markdown("""
    <style>
    /* Dark Mode Theme overrides */
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Title styling */
    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00b4db, #0083b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Design (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Metrics display */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat Bubble styling */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        line-height: 1.5;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .chat-user {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: #f8fafc;
        margin-left: 20%;
        border-bottom-right-radius: 2px;
    }
    .chat-assistant {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        color: #f1f5f9;
        margin-right: 20%;
        border-bottom-left-radius: 2px;
    }
    
    /* Alerts and Risks */
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title Section
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown('<div class="app-title">Aegis AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Autonomous Health Agent & Decision Support Portal</div>', unsafe_allow_html=True)

# Sidebar Configuration Panel
st.sidebar.markdown("### ⚙️ Portal Configuration")
api_base_url = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")
model_type = st.sidebar.selectbox("Prediction Model Engine", options=["xgboost", "random_forest"])
openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Enter OpenAI key to unlock conversational GenAI guidance.")

# Session Memory Initializations
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "extracted_biomarkers" not in st.session_state:
    st.session_state.extracted_biomarkers = None
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# Main Portal Navigation
tabs = st.tabs(["🩺 Clinical Chatbot", "📊 Analytics Dashboard", "📋 Diagnostics & API Details"])

# Tab 1: Clinical Chatbot
with tabs[0]:
    col_chat, col_report = st.columns([3, 2])
    
    with col_report:
        st.markdown('<div class="glass-card"><h4>📋 Upload Medical Report</h4>'
                    '<p style="color: #94a3b8; font-size: 0.85rem;">Upload medical PDFs or lab result images (JPG/PNG). Aegis will extract biochemical markers (Glucose, Hemoglobin, Cholesterol) and summarize report metrics.</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose report file...", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            if st.button("🔍 Process Report"):
                with st.spinner("Executing OCR and parsing biochemical values..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    try:
                        res = requests.post(f"{api_base_url}/analyze_report", files=files)
                        if res.status_code == 200:
                            report_summary = res.json()
                            st.session_state.extracted_biomarkers = report_summary.get("biomarkers")
                            
                            st.success("Report analysis completed successfully!")
                            
                            # Display extracted markers
                            st.markdown("##### Extracted Biomarkers")
                            b_cols = st.columns(3)
                            markers = report_summary.get("biomarkers", {})
                            
                            with b_cols[0]:
                                val = markers.get("glucose")
                                st.metric("Glucose", f"{val} mg/dL" if val else "Not Detected", delta="Abnormal" if val and (val > 100 or val < 70) else None, delta_color="inverse")
                            with b_cols[1]:
                                val = markers.get("hemoglobin")
                                st.metric("Hemoglobin", f"{val} g/dL" if val else "Not Detected", delta="Abnormal" if val and val < 12.0 else None, delta_color="inverse")
                            with b_cols[2]:
                                val = markers.get("cholesterol")
                                st.metric("Cholesterol", f"{val} mg/dL" if val else "Not Detected", delta="Abnormal" if val and val >= 200 else None, delta_color="inverse")
                                
                            # Display findings
                            st.markdown("##### Lab Summary Findings")
                            for finding in report_summary.get("findings", []):
                                st.info(finding)
                                
                            # Automatically append report summary to chat context
                            biomarker_str = ", ".join([f"{k}: {v}" for k, v in markers.items() if v is not None])
                            st.info(f"Biomarkers automatically loaded into your active chat session.")
                        else:
                            st.error(f"Failed to analyze report: {res.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {e}")
                        
        if st.session_state.extracted_biomarkers:
            if st.button("🧹 Clear Extracted Lab Values"):
                st.session_state.extracted_biomarkers = None
                st.rerun()

    with col_chat:
        st.markdown("#### Clinical Conversation")
        
        # Display Message History
        for msg in st.session_state.messages:
            bubble_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f'<div class="chat-bubble {bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)
            
        # Chat input
        user_input = st.chat_input("Describe symptoms, ask questions, or discuss report results...")
        
        if user_input:
            # Display user bubble
            st.markdown(f'<div class="chat-bubble chat-user">{user_input}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Aegis clinical agent is analyzing..."):
                payload = {
                    "user_text": user_input,
                    "session_id": st.session_state.session_id,
                    "openai_api_key": openai_key if openai_key != "" else None,
                    "biomarkers": st.session_state.extracted_biomarkers
                }
                
                try:
                    res = requests.post(f"{api_base_url}/chat", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        response = data.get("response", "")
                        
                        # Add predictions to analytics history
                        if data.get("predictions"):
                            pred = data["predictions"][0]
                            st.session_state.predictions_history.append({
                                "disease": pred["disease"],
                                "confidence": pred["confidence"],
                                "risk": data.get("risk_assessment", {}).get("level", "Low")
                            })
                            
                        # Refresh page to show assistant bubble
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    else:
                        st.error(f"Error from health agent: {res.text}")
                except Exception as e:
                    st.error(f"Could not reach clinical backend: {e}")

# Tab 2: Analytics Dashboard
with tabs[1]:
    st.markdown("#### 📊 Diagnostic Analytics & Performance Indicators")
    
    if not st.session_state.predictions_history:
        st.info("No diagnostic predictions registered in this session yet. Interact with the Chatbot above to populate clinical metrics.")
    else:
        # Load prediction history as DataFrame
        df = pd.DataFrame(st.session_state.predictions_history)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Total Predictions</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with col_m2:
            top_disease = df["disease"].mode()[0] if not df.empty else "N/A"
            st.markdown(f'<div class="glass-card"><div class="metric-label">Top Diagnosis</div><div class="metric-value" style="font-size:1.8rem;">{top_disease}</div></div>', unsafe_allow_html=True)
        with col_m3:
            high_risk_pct = (df["risk"] == "High").sum() / len(df) * 100
            st.markdown(f'<div class="glass-card"><div class="metric-label">High Risk Cases %</div><div class="metric-value">{high_risk_pct:.1f}%</div></div>', unsafe_allow_html=True)
            
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("##### Disease Classification Frequencies")
            fig_bar = px.bar(
                df["disease"].value_counts().reset_index(),
                x="disease",
                y="count",
                labels={"disease": "Predicted Condition", "count": "Frequency"},
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_g2:
            st.markdown("##### Case Risk Severity Distribution")
            fig_pie = px.pie(
                df,
                names="risk",
                template="plotly_dark",
                color="risk",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # General Model Metrics comparison
    st.markdown("##### ⚙️ Machine Learning Model Comparison")
    
    # Load model training metrics summary
    metrics_path = r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models\metrics_summary.joblib"
    if os.path.exists(metrics_path):
        import joblib
        metrics = joblib.load(metrics_path)
        
        model_names = ["Random Forest Baseline", "XGBoost Optimized"]
        accs = [metrics["random_forest"]["accuracy"], metrics["xgboost"]["accuracy"]]
        f1s = [metrics["random_forest"]["f1"], metrics["xgboost"]["f1"]]
        cvs = [metrics["random_forest"]["cv_mean"], metrics["xgboost"]["cv_mean"]]
        
        comp_df = pd.DataFrame({
            "Model": model_names,
            "Accuracy": accs,
            "F1-Score": f1s,
            "5-Fold CV Accuracy": cvs
        })
        st.dataframe(comp_df, hide_index=True)
    else:
        st.warning("Model training metrics summary not found. Run model training first.")

# Tab 3: Diagnostics & API Details
with tabs[2]:
    st.markdown("#### 📋 API Configuration & Infrastructure Diagnostics")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("##### Backend API Health Status")
        try:
            health_res = requests.get(f"{api_base_url}/health")
            if health_res.status_code == 200:
                health_data = health_res.json()
                st.success("Connected to Backend API successfully!")
                st.json(health_data)
            else:
                st.error(f"Backend returned unhealthy: {health_res.text}")
        except Exception as e:
            st.error(f"Could not connect to backend API at {api_base_url}: {e}")
            
    with col_d2:
        st.markdown("##### Clinical Feature Map & Properties")
        symptoms_path = r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models\symptoms.joblib"
        if os.path.exists(symptoms_path):
            import joblib
            symptoms = joblib.load(symptoms_path)
            st.info(f"Total clinical features (symptoms) supported: {len(symptoms)}")
            
            with st.expander("Show Supported Symptoms List"):
                # Display 3-column list of clean symptoms
                clean_symptoms = [s.replace("_", " ").capitalize() for s in symptoms]
                st.write(", ".join(clean_symptoms))
        else:
            st.warning("Feature symptoms library not found.")
