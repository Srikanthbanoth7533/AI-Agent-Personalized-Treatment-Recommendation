#!/bin/bash
# Start FastAPI backend in the background on port 8000
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground on port 8501
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
