# NAKA Biosignature Detection — TRAPPIST-1e Digital Twin

A digital twin pipeline for modelling stellar flare interference with biosignature detection on TRAPPIST-1e, built for the HACK-4-SAGES hackathon.

The system ingests real TESS observational data, simulates UV-driven atmospheric photochemistry across three candidate atmospheres, models a living biosphere's response to flare events, generates synthetic JWST transmission spectra, and provides an interactive Streamlit dashboard for exploring the results.

---

## Prerequisites

- Python **3.10 or higher** (3.11 recommended)
- Internet connection for Stage 1 (downloads TESS data from MAST)
- ~500 MB free disk space

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/NAKA-Biosignature-Detection.git
cd NAKA-Biosignature-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each stage in order. Each stage reads outputs from the previous one.

```bash
# Stage 1 — Downloads TESS light curve, generates flare catalog,
#            builds stellar spectrum, initializes twin_state.json
#            (requires internet connection)
python main.py

# Stage 2 — Computes UV photochemistry for 3 atmospheric scenarios,
#            runs biosphere model, updates twin_state.json
python stage2_atmospheric_response.py

# Stage 3 — Generates synthetic transmission spectra at JWST resolution,
#            computes biosphere-on vs biosphere-off spectra
python stage3_spectral_generation.py

# Stage 4 — Launch the interactive Streamlit dashboard
streamlit run stage4_dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## Notes

- Stages 2–4 are fully **offline** after Stage 1 has run once. The TESS data is cached locally by `lightkurve`.
- If you only want the **static PNG summary** without the browser dashboard, run `python stage4_dashboard.py` instead.
- Output files are written to `Stage 1 Files/`, `Stage 2 Files/`, `Stage 3 Files/`, and `Stage 4 Files/` respectively.
- `twin_state.json` in the root directory is the shared digital twin state — do not delete it between stages.

---

## Project Structure

```text
NAKA-Biosignature-Detection/
├── main.py                          Stage 1: Data ingestion
├── stage2_atmospheric_response.py   Stage 2: Photochemistry + biosphere
├── stage3_spectral_generation.py    Stage 3: Spectra + biosphere-on/off
├── stage4_dashboard.py              Stage 4: Calibration + digital twin dashboard
├── twin_core.py                     Shared: digital twin state engine (all stages import this)
├── twin_state.json                  Live twin state (written by S1, updated by S2; read by S3/S4)
├── requirements.txt
│
├── Stage 1 Files/
│   ├── trappist1_tess_lightcurve.csv
│   ├── trappist1_flare_catalog.csv
│   ├── trappist1_stellar_spectrum.csv
│   ├── davenport_template_demo.csv
│   └── stage1_diagnostic_plots.png
│
├── Stage 2 Files/
│   ├── atmospheric_compositions.csv
│   ├── flare_uv_enhancement.csv
│   ├── photochem_response_atm_a/b/c.csv
│   ├── cumulative_biosignature_state.csv
│   ├── vulcan_analog_timeseries.csv
│   └── biosphere_state.csv
│
├── Stage 3 Files/
│   ├── spectrum_A_quiescent.csv (+ 6 more)
│   ├── all_spectra_combined.csv
│   ├── feature_amplitudes.csv
│   ├── spectrum_atmA_biosphere_on.csv
│   ├── spectrum_atmA_biosphere_off.csv
│   ├── biosphere_spectral_diff.csv
│   └── stage3_spectra_plot.png
│
└── Stage 4 Files/
    ├── jwst_dreams_synthetic.csv
    ├── calibration_results.csv
    └── stage4_summary.png
```

---

## Documentation

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) — Full explanation of every stage, every dashboard tab, and the digital twin architecture
- [SOURCES.md](SOURCES.md) — All scientific literature and data sources, with specific usage notes
