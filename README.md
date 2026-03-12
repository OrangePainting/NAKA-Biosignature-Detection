# NAKA-Biosignature-Detection
A respository for our project for the HACK-4-SAGES hackathon.

# How to Run:

*First, install all dependencies*

## 1. Download TESS data, generate flare catalog, initialize twin_state.json
python main.py

## 2. Compute photochemistry + biosphere model, update twin_state.json
python stage2_atmospheric_response.py

## 3. Generate transmission spectra + biosphere-on/off spectra
python stage3_spectral_generation.py

## 4. Launch the dashboard
streamlit run stage4_dashboard.py
