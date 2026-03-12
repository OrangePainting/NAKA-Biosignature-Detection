# NAKA Biosignature Detection — TRAPPIST-1e Digital Twin
## Comprehensive Pipeline & Dashboard Guide

---

## What Is a Digital Twin?

A **digital twin** is a dynamic, real-time virtual replica of a physical system — in this case, the atmosphere and biosphere of TRAPPIST-1e. Unlike a one-pass analysis pipeline, a digital twin:

- Maintains a **living state** that evolves continuously as new flares accumulate
- Has **feedback loops** between systems (flares → atmosphere → biosphere → observable spectra → JWST observations → updated state)
- Can be **projected forward** to predict future atmospheric and biological conditions
- **Assimilates new observations** — when JWST data arrives, it updates the twin's state estimate using Bayesian inference

The twin's persistent state is serialized to `twin_state.json` and shared across all pipeline stages. Each stage reads the current state, updates it, and writes it back. The dashboard reads it with a 30-second cache so it always reflects the latest run.

## Pipeline Architecture

```
twin_state.json  ←──────────────────────────────────────────┐
     │                                                       │
     ▼                                                       │
Stage 1 (main.py)                 → Stage 1 Files/          │
  Real TESS data + flare catalog                            │
  Initializes twin_state.json                               │
     │                                                       │
     ▼                                                       │
Stage 2 (stage2_atmospheric_response.py) → Stage 2 Files/  │
  UV doses + photochemical response                         │
  Biosphere model (Eager-Nash+2024)                         │
  Updates twin_state.json                                   │
     │                                                       │
     ▼                                                       │
Stage 3 (stage3_spectral_generation.py)  → Stage 3 Files/  │
  Transmission spectra                                      │
  Biosphere-on vs biosphere-off spectra                    │
     │                                                       │
     ▼                                                       │
Stage 4 (stage4_dashboard.py)    → Stage 4 Files/           │
  Live twin state monitor                                   │
  JWST calibration + prediction                             │
  JWST data assimilation ──────────────────────────────────►┘
```

**Shared module:** `twin_core.py` — the digital twin's nervous system, imported by all stages. Contains `TwinState`, biosphere model, forward prediction engine, and Bayesian assimilation.

**Core scientific question:** When TRAPPIST-1 flares, the star temporarily becomes hundreds to millions of times brighter in the ultraviolet. Does this UV burst destroy the atmospheric biosignatures (O₃, CH₄, N₂O) we might detect with JWST? Does the biosphere survive? Can JWST distinguish a living world from an abiotic one? What will the atmosphere look like in a year?

---

## Stage 1: Data Ingestion (`main.py`)

**Goal:** Acquire and structure all raw observational inputs needed by later stages.

### What it does

#### Digital Twin Initialization

At the end of Stage 1, `initialize_twin_state()` from `twin_core.py` creates `twin_state.json` with:

- All 3 atmospheric compositions seeded with initial VMRs (including CO for Atm A — the M-dwarf-enhanced abiotic baseline from Eager-Nash+2024)
- Biosphere initialized at full health (population_factor = 1.0)
- Flare event log from the catalog
- Empty prediction and observation slots ready for Stages 2–4

#### Step 1 — TESS Light Curve Download
Downloads the TESS Sector 10 light curve for TRAPPIST-1 (TIC 267519185) using the `lightkurve` package. This gives ~79 days of photometry at 1800-second cadence. The light curve is used to establish the observing baseline and to visualise the stellar variability.

> **Limitation noted in code:** The 1800-second cadence is too coarse to directly detect individual flares, which last 1–30 minutes. Published flare statistics from dedicated higher-cadence campaigns are therefore used instead (Steps 2–3).

#### Step 2 — Flare Catalog Generation
Rather than detecting flares from the coarse TESS data, the code generates a statistically realistic catalog by sampling from the **Vida et al. 2017 flare frequency distribution (FFD)**:

- **Rate:** 0.53 flares per day (Poisson-sampled over 79 days → ~42 flares)
- **Energies:** Power-law distribution, dN/dE ∝ E^−1.72, ranging from 10³⁰ to 5×10³³ erg
- **Durations:** Scaled by E^0.39 (Davenport 2014 empirical relation)
- **Peak amplitudes:** Derived from energy / (stellar luminosity × duration)
- **Temperatures:** Uniformly sampled 5300–8600 K (Ducrot et al. 2022, TRAPPIST-1 specific)

Flares are classified into four categories:
| Class | Energy range |
|---|---|
| Micro | < 10³¹ erg |
| Moderate | 10³¹ – 10³² erg |
| Large | 10³² – 10³³ erg |
| Superflare | > 10³³ erg |

#### Step 3 — Davenport (2014) Flare Template
Builds the empirical white-light flare shape model used to convolve individual flare pulses with the stellar spectrum. The template has two components:

- **Rise phase** (−1 to 0 in normalised time): 4th-order polynomial — `1 + 1.941p − 0.175p² − 2.246p³ − 1.125p⁴`
- **Decay phase** (0 → ∞): Double exponential — `0.689·exp(−1.60p) + 0.303·exp(−0.278p)`

This shape is physically motivated: the rise reflects magnetic reconnection energy release, the fast decay is impulsive phase emission, and the slow tail is gradual phase cooling.

#### Step 4 — Stellar Spectrum
Constructs the TRAPPIST-1 spectral energy distribution using a **Planck blackbody** at T_eff = 2566 K, calibrated against the **MegaMUSCLES** UV flux measurements (France et al. 2020, Wilson et al. 2025):

- Wavelength range: 0.1 – 30,000 nm (log-spaced, 5000 points)
- Quiescent spectrum: Planck at 2566 K, scaled by (R_star/a_planet)²
- Flare spectrum: Additional Planck component at 7000 K covering 1% of the stellar surface

The NUV (100–320 nm) flux increases by ~1.5 million times during a flare compared to quiescence, which is what drives the photochemistry in Stage 2.

#### Step 5 — Diagnostic Plots
Generates a 4-panel figure: TESS light curve, flare energy histogram, template shapes at three energies, and quiescent vs. flare spectrum.

### Output files (`Stage 1 Files/`)
| File | Contents |
|---|---|
| `trappist1_tess_lightcurve.csv` | BTJD time, normalised flux, flux uncertainty |
| `trappist1_flare_catalog.csv` | 47 flares: time, energy, duration, amplitude, temperature, class |
| `trappist1_stellar_spectrum.csv` | Wavelength grid with quiescent and flare flux |
| `davenport_template_demo.csv` | Single flare template time series (5% amplitude) |
| `stage1_diagnostic_plots.png` | 4-panel diagnostic figure |

---

## Stage 2: Atmospheric Photochemical Response (`stage2_atmospheric_response.py`)

**Goal:** Calculate how the UV radiation from each flare modifies the atmospheric chemistry of TRAPPIST-1e, for three candidate atmospheric compositions.

### What it does

#### Section 1 — Atmospheric Compositions
Defines three candidate atmospheres consistent with the JWST DREAMS NIRSpec/PRISM constraints (Espinoza et al. 2025):

**Atmosphere A — N₂-dominated (Earth-like)**
The scientifically most interesting case. Contains genuine biosignature gases:
- 79% N₂, 21% O₂, 420 ppm CO₂, 1.8 ppm CH₄, 300 ppb O₃, 320 ppb N₂O
- Now also includes **100 ppb CO** as an abiotic M-dwarf baseline (Eager-Nash+2024: M-dwarf UV drives 5–10× more CO₂ photolysis than the Sun, making CO the dominant atmospheric feature unless a biosphere consumes it)
- Has a strong O₃ UV shield (equivalent to ~300 Dobson Units)
- Status: *Permitted* by DREAMS data
- Key tension: 1D models predict O₃ destruction; Ridgway et al. 2023 (3D GCM) predicts O₃ *increases* 20× — an unresolved scientific conflict explicitly flagged throughout the pipeline.

**Atmosphere B — CO₂-dominated (Venus-analog)**
A high-pressure, hot Venus-like scenario with a critical **false-positive pathway**:
- 96% CO₂, 3.5% N₂, 150 ppm SO₂, trace O₃
- Flare UV drives CO₂ photolysis → CO + O → traces of abiotic O₃ (Miranda-Rosete et al. 2024)
- Demonstrates how O₃ detection alone cannot confirm biology
- Status: *Disfavoured but not ruled out* by DREAMS

**Atmosphere C — Thin remnant / bare rock**
Represents the airless or near-airless scenario seen in other TRAPPIST-1 siblings (b, c, d):
- 10 mbar surface pressure, 98% N₂, 2% CO₂, no biosignature gases
- UV transmittance of 97%: flares hit the surface almost unattenuated
- Status: *Possible*, consistent with featureless DREAMS spectrum

#### Section 2 — UV Enhancement Calculator
For each flare in the Stage 1 catalog, computes the NUV and FUV flux enhancement at TRAPPIST-1e using **Planck blackbody ratios**:

```
Enhancement(λ) = B(λ, T_flare=7000K) / B(λ, T_eff=2566K) × (flare area fraction)
```

Key results from the flare catalog:
- TESS band (800 nm): ~91.7× enhancement
- NUV (250 nm): ~1.48 million× enhancement
- NUV fluence range across all 47 flares: 24× to 4484× above quiescent

UV energy fractions calibrated from M-dwarf flare spectroscopy (Kowalski et al. 2013):
- NUV (200–320 nm): 2.5% of bolometric flare energy
- FUV (100–200 nm): 0.2% of bolometric flare energy

#### Section 3 — Photochemical Lookup Tables
Instead of running a full photochemical model (VULCAN, ATMO) live, the code interpolates from peer-reviewed results. This is seen in published literature:

| Source | Key result used |
|---|---|
| Segura et al. 2010 | 1D: UV-only flare → ~5% O₃ loss; UV+stellar energetic particles (SEPs) → 94% loss over 2 years |
| Tilley et al. 2019 | Cumulative repeated flaring reduces O₃ to 6% of initial value over 10 years |
| Chen et al. 2021 (3D WACCM) | NO₂, N₂O, HNO₃ *increase* from flares; new chemical equilibria established |
| Miranda-Rosete et al. 2024 | Abiotic O₃ production in CO₂ atmospheres from superflares |

For each flare, per-species fractional changes are computed by interpolating these LUT results against flare NUV fluence.

#### Section 4 — Cumulative 79-Day State
Accumulates all per-flare chemical changes over the full 79-day observation window, with **exponential recovery** between events:

```
state(t) = state(t_prev) × exp(−Δt / τ_recovery) + Δ_flare
```

Recovery timescales (from photochemical modelling literature):
| Species | τ_recovery |
|---|---|
| O₃ | 30 days |
| CH₄ | 1825 days (5 years) |
| N₂O | 14 days |
| NO₂ | 0.5 days |
| HNO₃ | 3 days |
| OH | ~0.0001 days (seconds) |

**Cumulative results for Atmosphere A over 79 days:**
- O₃: −12.7% (partially offset by recovery)
- CH₄: −2.5% (slow recovery so losses accumulate)
- NO₂/HNO₃: increase (consistent with Chen et al. 2021)

#### Section 5 — VULCAN-Analog ODE (Stretch Goal)
A simplified photochemical ODE system implementing Chapman cycle + HOx chemistry, solved with `scipy.solve_ivp` for Atmosphere A during the largest observed flare (E = 1.55×10³² erg):

- Species tracked: O₃, OH, CH₄
- Rate constants from JPL/NASA Sander et al. 2011 at T = 250 K
- Result: O₃ −3.34% and OH +4290× during the 24-hour post-flare window

#### Section 6 — Biosphere Model & Digital Twin State Update (NEW — Eager-Nash+2024)

This is the core of the digital twin transformation. The biosphere is not a passive observer — it actively responds to and modifies the atmospheric chemistry through two coupled feedback loops:

##### Feedback loop 1: CO consumption

M-dwarf UV drives CO₂ photolysis at 5–10× the rate of Sun-like stars (Eager-Nash+2024). The resulting abiotic CO (VMR ~1×10⁻⁷ in Atm A) would dominate the transmission spectrum. A living biosphere consumes this CO via:

```
4CO + 2H₂O → 2CO₂ + CH₃COOH
```

A healthy biosphere (population_factor = 1.0) suppresses CO by ~97%, reducing it from ~1×10⁻⁷ to ~3×10⁻⁹ — a 30× difference that is the clearest spectral signature of life on TRAPPIST-1e.

##### Feedback loop 2: Methanogenesis

H₂/CO-consuming archaea (the biosphere model) produce CH₄ to partially offset UV-driven OH oxidation:

```
4H₂ + CO₂ → 2H₂O + CH₄
```

Biogenic CH₄ replenishment rate scales with population_factor and the geological H₂ outgassing supply (default: 5 Tmol/yr).

##### UV stress model

Each high-UV flare (EF_NUV > 100) degrades the biosphere. Population decays as:

```
population_factor *= exp(-UV_dose / stress_scale)
```

where `stress_scale = 5×10⁹ erg/cm²` (Eager-Nash+2024 calibrated). Recovery between events follows:

```
population_factor → 1 with τ_recovery = 730 days
```

At the end of Stage 2, `update_twin_state()` writes all accumulated chemistry and biosphere state back to `twin_state.json`, making it available to Stages 3 and 4.

At the end of Stage 2, `update_twin_state()` writes all accumulated chemistry and biosphere state back to `twin_state.json`, making it available to Stages 3 and 4.

### Output files (`Stage 2 Files/`)

| File | Contents |
|---|---|
| `atmospheric_compositions.csv` | VMRs, pressures, temperatures for all 3 atmospheres |
| `flare_uv_enhancement.csv` | Per-flare NUV/FUV fluence and enhancement factors |
| `photochem_response_atm_a.csv` | Per-flare dO₃, dCH₄, dN₂O, dNO₂, dHNO₃ for Atm A |
| `photochem_response_atm_b.csv` | Per-flare response for Atm B (includes abiotic O₃) |
| `photochem_response_atm_c.csv` | Per-flare response for Atm C (UV surface dose only) |
| `cumulative_biosignature_state.csv` | 79-day net changes for all 3 atmospheres |
| `vulcan_analog_timeseries.csv` | O₃/OH/CH₄ time series from ODE integration |
| `biosphere_state.csv` | Per-flare biosphere evolution: population factor, UV dose, biotic CO, biogenic CH₄ |

---

## Stage 3: Transmission Spectra Generation (`stage3_spectral_generation.py`)

**Goal:** Convert the atmospheric compositions from Stage 2 into synthetic transmission spectra at JWST NIRSpec/PRISM resolution, both before and after the flare sequence.

### What it does

#### Section 1 — Load Stage 2 Outputs
Reads the three atmospheric compositions and the cumulative flare-induced changes. NaN entries (species absent in a given atmosphere) are safely skipped.

#### Section 2 — Wavelength Grid
Builds a logarithmically-spaced wavelength grid at resolving power R = 100 (matching JWST NIRSpec/PRISM):

- Range: 0.5 – 14.0 µm
- Number of points: ~334
- Spacing: Δλ/λ = 1/100 = 1% per bin

#### Section 3 — Molecular Cross-Sections
Parameterises absorption cross-sections as **sums of Gaussians** centred at published HITRAN2020 band centres. This avoids needing to download or store the full HITRAN line-by-line database while capturing the key spectral features at R=100 resolution.

Species included with their key bands:
| Species | Key JWST-observable feature | Peak cross-section (cm²) |
|---|---|---|
| H₂O | 1.38, 1.87, 2.7, 6.3 µm | up to 6×10⁻²¹ |
| CO₂ | **4.3 µm** (dominant), 2.7, 15 µm | up to 8×10⁻²⁰ |
| CH₄ | **3.3 µm**, 2.3, 7.7 µm | up to 4×10⁻²¹ |
| O₃ | **9.6 µm** (Hartley band), 0.6 µm (Chappuis) | up to 3.5×10⁻²⁰ |
| N₂O | **4.5 µm**, 7.8 µm | up to 1.5×10⁻²¹ |
| NO₂ | 6.2 µm | — |
| CO | 2.35, 4.7 µm | — |
| O₂ | 0.76 µm (A-band), 1.27 µm | — |

Additional continuum opacity:
- **Rayleigh scattering:** σ_R = 4.6×10⁻²⁷ × (0.55/λ_µm)⁴ cm² (blue-scattering slope)
- **N₂–N₂ CIA:** collision-induced absorption at long wavelengths
- **CO₂–CO₂ CIA:** relevant for the Venus-analog atmosphere

#### Section 4 — Transmission Spectrum Calculation
Implements the **Lecavelier des Etangs (2008)** analytic radiative transfer formalism. This is the same underlying physics used by petitRADTRANS and PICASO, but self-contained:

```
delta(λ) = (R_p/R_star)² + 2·R_p·H / R_star² × ln(τ₀(λ) × 2/3)

where:
  H = k_B·T / (μ·g)           — atmospheric scale height
  τ₀(λ) = Σᵢ [Xᵢ · σᵢ(λ) · n₀ · H · √(2π·R_p/H)]  — chord optical depth
  Xᵢ = volume mixing ratio of species i
  n₀ = number density at reference pressure level
```

Key planetary parameters:
- R_p = 0.920 Earth Radius = 5.861×10⁸ cm
- R_star = 0.1192 * Our Sun's Radius = 8.295×10⁹ cm
- Surface gravity g = 803 cm/s²
- Continuum transit depth (no atmosphere): ~4,996 ppm

Scale heights for each atmosphere:
- Atm A (N₂/O₂, T=270K, μ≈29 amu): H ≈ 8.4 km
- Atm B (CO₂, T=380K, μ≈43 amu): H ≈ 4.0 km
- Atm C (thin N₂/CO₂, T=250K): H ≈ 7.5 km (but very low pressure)

#### Section 5 — Seven Spectra Computed
For each atmosphere and scenario, a full spectrum is computed:

| Atmosphere | Scenario | Description |
|---|---|---|
| A | Quiescent | N₂-Earth with all biosignatures at initial VMRs |
| A | Post-flare (cumul) | After 79-day flare sequence: O₃ −12.7%, CH₄ −2.5% |
| A | VULCAN single flare | After largest single flare: O₃ −3.34% |
| B | Quiescent | CO₂-Venus at initial VMRs |
| B | Post-flare (cumul) | Includes abiotic O₃ addition from UV photolysis |
| C | Quiescent | Thin rock at initial VMRs |
| C | Post-flare (cumul) | Minimal change (no photochemistry) |

**Key feature amplitudes (above continuum):**
| Feature | Atm A | Atm B | Atm C |
|---|---|---|---|
| O₃ at 9.6 µm | 12.7 ppm | ~0 ppm | ~0 ppm |
| CH₄ at 3.3 µm | 33.4 ppm | ~0 ppm | ~0 ppm |
| CO₂ at 4.3 µm | 35.2 ppm | ~800 ppm | ~2 ppm |

Note: O₃ at 12.7 ppm is **below the JWST detection horizon of ~20 ppm** for 10 transits of TRAPPIST-1e.

#### Section 7 — Biosphere-On vs Biosphere-Off Spectra (NEW — Eager-Nash+2024)

This section generates the most direct spectral test for life on TRAPPIST-1e. Using the biosphere state from `twin_state.json`, it computes two versions of Atmosphere A:

**Biosphere-off (abiotic baseline):**

- CO VMR: ~1×10⁻⁷ (M-dwarf UV drives sustained CO₂ photolysis — Eager-Nash+2024)
- CH₄ VMR: no biogenic replenishment beyond photochemical equilibrium

**Biosphere-on (living world):**

- CO VMR: ~3×10⁻⁹ (97% suppressed by H₂/CO-consuming archaea)
- CH₄ VMR: enhanced by biogenic production scaled to current population_factor

The spectra are subtracted (`biosphere_spectral_diff.csv`) to isolate the pure biological signal. Key band amplitudes:

| Band | Biosphere-off depth | Biosphere-on depth | Δ (biological signal) |
|---|---|---|---|
| CO 4.67 µm | ~0.8 ppm | ~0.025 ppm | ~0.78 ppm (30× suppression) |
| CH₄ 3.3 µm | baseline | enhanced | depends on population_factor |

The CO band difference (~0.78 ppm) is below the current JWST systematic floor (~20 ppm per transit), but would accumulate to ~3.5σ confidence with ~100 stacked transits — the science case for long-term TRAPPIST-1e monitoring programmes.

`load_twin_biosphere()` reads the biosphere state from `twin_state.json` (population_factor, biotic CO VMR, biogenic CH₄ rate) with a fallback to hardcoded defaults so Stage 3 can run standalone if Stage 2 hasn't yet been executed.

### Output files (`Stage 3 Files/`)

| File | Contents |
|---|---|
| `spectrum_A_quiescent.csv`, etc. | Individual spectrum files (7 total) |
| `all_spectra_combined.csv` | All spectra in one table: wavelength_um, transit_depth_ppm, atmosphere, scenario |
| `feature_amplitudes.csv` | Peak amplitude of each molecular feature for each atmosphere/scenario |
| `spectrum_atmA_biosphere_on.csv` | Atmosphere A with living biosphere (suppressed CO, biogenic CH₄) |
| `spectrum_atmA_biosphere_off.csv` | Atmosphere A abiotic baseline (full CO, no biogenic CH₄) |
| `biosphere_spectral_diff.csv` | Difference spectrum (biosphere-on minus biosphere-off) |
| `stage3_spectra_plot.png` | Multi-panel diagnostic figure |

---

## Shared Module: Digital Twin Engine (`twin_core.py`)

`twin_core.py` is imported by all four pipeline stages. It is the nervous system of the digital twin — the only file that reads and writes `twin_state.json`.

### Functions

| Function | Called by | Purpose |
|---|---|---|
| `initialize_twin_state(flare_rows)` | Stage 1 | Creates a fresh `twin_state.json` with seeded VMRs, healthy biosphere, and the Stage 1 flare catalog |
| `load_twin_state(path)` | Stages 2–4 | Reads and deserialises `twin_state.json` |
| `save_twin_state(state, path)` | Stages 2–4 | Atomic write (`.tmp` → `os.replace()`) to prevent corruption if a stage crashes mid-run |
| `update_atmosphere(state, atm_label, species_changes)` | Stage 2 | Applies fractional VMR changes to a named atmosphere in the state |
| `biosphere_co_consumption(state, co_vmr_abiotic)` | Stage 2 | Suppresses CO by 97% × population_factor |
| `biosphere_ch4_production(state, dt_days)` | Stage 2 | Adds biogenic CH₄ at 2×10⁻¹¹ VMR/day × population_factor |
| `update_biosphere_stress(state, ef_nuv, duration_sec)` | Stage 2 | `population_factor *= exp(−UV_dose / 5×10⁹)` |
| `biosphere_recovery(state, dt_days, tau=730)` | Stage 2 | Exponential recovery between flare events |
| `forward_predict(state, horizon_days, ...)` | Stage 4 | Monte Carlo future projection: samples new flares from Vida+2017 FFD, propagates photochemistry and biosphere forward in time |
| `assimilate_observation(state, lam_jwst, obs_ppm, sigma_ppm, spectra_dict, obs_id)` | Stage 4 | Bayesian posterior update: chi-squared likelihoods `L_i = exp(−χ²_i / 2)` for each atmosphere scenario |
| `compute_detection_horizon(state, species, feature_amp_ppm, noise_per_transit_ppm)` | Stage 4 | Computes transits needed for 5-sigma detection of a given spectral feature |

### State structure (`twin_state.json`)

```json
{
  "timestamp": "ISO datetime of last update",
  "stage": "which pipeline stage last wrote this",
  "flare_catalog": [...],
  "atmospheres": {
    "A": { "O3": 3e-7, "CH4": 1.8e-6, "CO": 3e-9, ... },
    "B": { ... },
    "C": { ... }
  },
  "biosphere": {
    "population_factor": 0.94,
    "cumulative_uv_dose": 1.2e10,
    "biotic_co_vmr": 3e-9,
    "biogenic_ch4_rate": 1.88e-11
  },
  "history": [ { "day": 0.0, "O3": ..., "CH4": ... }, ... ],
  "observations": [...],
  "posterior_weights": { "A_quiescent": 0.12, "A_postflare": 0.08, ... },
  "prediction": { "days": [...], "O3": [...], "population_factor": [...] }
}
```

---

## Stage 4: JWST Calibration & Streamlit Dashboard (`stage4_dashboard.py`)

**Goal:** Compare the predicted spectra against the real JWST DREAMS result and provide an interactive interface for exploring the full pipeline output.

### Running the dashboard

```bash
# Interactive Streamlit dashboard (opens in browser)
streamlit run stage4_dashboard.py

# Static PNG summary only
python stage4_dashboard.py
```

### Synthetic JWST DREAMS Spectrum

Before the dashboard can compare models to data, it constructs a synthetic version of the DREAMS (TRAPPIST-1 Evaluation of Atmosphere in Multiple Systematic) NIRSpec/PRISM spectrum based on Espinoza et al. 2025:

- **Shape:** Featureless — flat transit depth consistent with no thick atmosphere
- **Wavelength range:** 0.6 – 5.3 µm (NIRSpec/PRISM coverage)
- **Baseline depth:** ~4,996 ppm ((R_p/R_star)²)
- **Stellar contamination:** Weak blue slope (+0.5 ppm/µm) from unocculted star spots (Rackham et al. 2018)
- **Noise model:** σ_phot ≈ (30 + 15·(λ−0.6)) / √N_transits ppm, plus 20 ppm systematic floor
- **Adjustable:** The sidebar N_transits slider changes the noise level, simulating deeper observations

### Chi-Squared Calibration

Quantifies how well each predicted spectrum matches the synthetic DREAMS spectrum:

```
χ²_reduced = (1/N) × Σ [(model(λ) − obs(λ))² / σ(λ)²]
```

Evaluated across all ~200 wavelength bins in the 0.6–5.3 µm NIRSpec range.

**Typical results:**
| Atmosphere | Scenario | χ²_reduced | Interpretation |
|---|---|---|---|
| C (thin rock) | Quiescent | ~2.1 | Best fit — featureless matches featureless |
| A (N₂-Earth) | Quiescent | ~6.6 | Tension — CH₄/H₂O features inconsistent |
| B (CO₂-Venus) | Quiescent | ~13.4 | Poor fit — strong CO₂ features ruled out |

---

## Dashboard: Sidebar Controls

The sidebar controls affect the entire dashboard in real time.

| Control | What it does |
|---|---|
| **Atmospheric composition** | Selects which atmosphere (A, B, or C) is highlighted in spectral plots and used for detail views in the biosignature and animation tabs |
| **JWST transits** | Sets the number of transit observations in the noise model. More transits → lower noise → tighter constraints. Range: 1–20 |
| **Show flare classes** | Filters which flare energy classes appear in the timeline scatter plots |
| **Min flare log₁₀(E/erg)** | Additional energy floor filter for flares shown (range: 30–33) |
| **Flare sequence: day** | Position in the 79-day observation window for the animation tab. Drag to step through the flare sequence |
| **Upload JWST observation (CSV)** | File uploader for real or synthetic JWST spectra. Columns expected: `wavelength_um`, `depth_ppm`, `sigma_ppm`. On upload, calls `assimilate_observation()` in `twin_core.py` to update the Bayesian posterior weights and adds the observation to the twin state record |

---

## Dashboard: Metric Cards

Four summary numbers displayed at the top of the page:

| Metric | What it shows |
|---|---|
| **Flares (79-day window)** | Total number of flares in the Stage 1 catalog, with superflare count as delta |
| **Cumul. O₃ change (Atm A)** | Net 79-day O₃ depletion from 1D model, with the 3D model counterpoint (Ridgway+2023: +2000%) as a note |
| **Best-fit atmosphere (chi2)** | Which atmosphere is most consistent with DREAMS, with its χ² value |
| **O₃ feature (9.6 µm, Atm A)** | The amplitude of the O₃ biosignature feature, versus the ~20 ppm JWST detection horizon |

---

## Dashboard: Tab 0 — Twin State Monitor

**What it shows:** The live state of the digital twin — the most recently written `twin_state.json`, refreshed every 30 seconds.

**Left column — Biosphere health gauge:**
A large circular metric showing the current `population_factor` (0.0 = extinct, 1.0 = healthy). Below it, three sub-metrics: cumulative UV dose received, biotic CO VMR (suppressed by the biosphere), and biogenic CH₄ production rate. A colour-coded status label (green/yellow/red) indicates whether the biosphere is thriving, stressed, or critically depleted.

**Centre column — Atmospheric VMR state:**
A horizontal bar chart showing the current VMR of each tracked species in Atmosphere A, compared against the Stage 1 initial values. Red bars indicate depletion from flare photochemistry; blue bars indicate enhancement. The CO bar visually contrasts the biotic (low) vs. abiotic (high) scenarios.

**Right column — Twin state metadata:**
Stage that last wrote the state, ISO timestamp, number of flare events logged, number of JWST observations assimilated, and the current posterior weights for each atmosphere scenario. If no `twin_state.json` is found, a warning with instructions to run Stage 1 is displayed instead.

**79-day history time series:**
If Stage 2 has run, a multi-line plot shows the evolution of O₃, CH₄, and biosphere population_factor over the 79-day observation window, with flare events marked as vertical lines coloured by energy class.

---

## Dashboard: Tab 1 — Flare Timeline

**What it shows:** The 79-day history of stellar flares on TRAPPIST-1, as seen from the Stage 1 catalog.

**Left panel — Scatter plot:**
Each dot is one flare. The x-axis is the day of occurrence (0–79), the y-axis is log₁₀(energy in erg). Dot size scales with energy. Colours code flare class: blue (micro), orange (moderate), red (large), purple (superflare). Horizontal dashed lines mark the large (10³²) and superflare (10³³) energy boundaries. If the sidebar day slider is non-zero, a red vertical line marks the current animation position.

**Right panel — NUV fluence histogram:**
Distribution of log₁₀(total NUV fluence at TRAPPIST-1e surface) across all flares. Two reference lines mark:
- Green: The Segura et al. 2010 O₃ depletion threshold
- Red: The fluence of the AD Leonis 1985 superflare (a well-studied benchmark)

**Table:** Top 10 strongest flares with their time, energy, duration, temperature, and class.

---

## Dashboard: Tab 2 — Spectral Comparison

**What it shows:** How the three predicted atmospheric spectra compare against the synthetic JWST DREAMS measurement.

**Main spectral panel:**
- Black dots with error bars: synthetic DREAMS spectrum (featureless, ~5000 ppm)
- Blue line: Atm A (N₂-Earth) — shows H₂O bumps, CH₄ at 3.3 µm, CO₂ at 4.3 µm
- Red line: Atm B (CO₂-Venus) — shows dominant CO₂ at 4.3 µm
- Grey line: Atm C (thin rock) — nearly flat
- The atmosphere selected in the sidebar is drawn thicker and more opaque; its dashed line shows the post-flare version
- Vertical purple dotted lines annotate key molecular bands

**Residuals subplot:** Model minus DREAMS for each atmosphere. The grey band is the 1-sigma JWST noise envelope. Points within the band are statistically consistent with the data.

**Right panel:** Metadata for the selected atmosphere (pressure, temperature, JWST status) plus a chi-squared summary for all three atmospheres with context (chi² < 1.5 acceptable; chi² > 3.0 in tension).

---

## Dashboard: Tab 3 — Biosignature Vulnerability

**What it shows:** How individual flares and the cumulative flare sequence modify key biosignature mixing ratios.

**Four sub-panels:**

- **O₃ depletion per flare (Atm A):** Scatter plot of per-event O₃ fractional change vs. flare energy, coloured by NUV enhancement factor. Shows the power-law scaling between flare energy and photochemical impact. The red warning label flags the 1D vs. 3D model tension (Ridgway et al. 2023 3D result is opposite).

- **CH₄ depletion per flare:** Same structure. CH₄ is destroyed by OH radicals produced in the UV photolysis chain. Losses accumulate more than O₃ because CH₄ has a 5-year recovery timescale.

- **NO₂ enhancement per flare:** Shows that flares *increase* NO₂. This is the Chen et al. 2021 finding: high-energy UV drives N₂ fixation chemistry that produces NO₂, N₂O, and HNO₃ as new equilibrium products. If true at 3D scale, flares might actually enhance some biosignature-adjacent molecules.

- **79-day cumulative state (bar chart):** Net fractional change for each species after the full observation window. Blue bars = depletion; red bars = enhancement.

**Right panel:** For the selected atmosphere, a metric card for each key feature (O₃, CH₄, CO₂, H₂O, N₂O) showing the quiescent amplitude and percentage change after flares. A progress bar shows O₃ at 9.6 µm relative to the ~20 ppm JWST detection horizon (currently ~64% of that threshold).

---

## Dashboard: Tab 4 — Calibration Residuals

**What it shows:** A detailed residual analysis for all three atmospheres, both before and after the flare sequence.

Three vertically stacked panels (one per atmosphere). Each panel shows:
- **Solid line:** Residuals for the quiescent spectrum (model minus DREAMS)
- **Dashed line:** Residuals for the post-79-day-flare spectrum
- **Grey band:** ±1σ JWST noise envelope
- **Legend χ²:** The reduced chi-squared for each scenario

This lets you see *where* the models agree or disagree with the data. For Atm B (CO₂-Venus), large positive residuals appear at 4.3 µm, showing exactly where the CO₂ feature is inconsistent with the featureless DREAMS result.

**Chi-squared summary table** below the figure: All atmosphere/scenario combinations ranked by χ², with RMS residual and mean offset.

**Interpretation text box:** Scientific context explaining that the featureless DREAMS spectrum best constrains Atm C (thin rock), but an N₂-dominated atmosphere with sub-ppb biosignature concentrations remains *permitted* — the features may simply be below current JWST sensitivity.

---

## Dashboard: Tab 5 — Chi-Squared Grid

**What it shows:** A colour-coded matrix of model-data agreement across all combinations of atmosphere (rows) and scenario (columns).

The heatmap uses a green-to-red scale (green = better fit). Each cell shows the numerical χ² value. The columns correspond to:
- **Quiescent:** Unperturbed atmosphere
- **Post-flare cumul:** After 79-day flare sequence
- **VULCAN single flare:** After the single largest flare (Atm A only)

This is the "digital twin feedback loop" panel: as new JWST TRAPPIST-1e data arrives from the ongoing DREAMS and GO 6456/9256 observing programmes, this matrix would be updated with real data to progressively constrain which atmospheric compositions are viable.

---

## Dashboard: Tab 6 — Flare Animation

**What it shows:** A step-through animation of how the transmission spectrum of Atmosphere A evolves as flares accumulate, driven by the sidebar day slider.

**Main spectral panel:**
- Black line: Quiescent reference spectrum (day 0)
- Grey dots: Synthetic JWST DREAMS spectrum for comparison
- Blue line: Spectrum at the current day (post-flares up to that point)
- Red fill: The difference between quiescent and current spectrum — visually shows the spectral change accumulated from flares

**Timeline progress panel** (below the spectrum):
- Light grey dots: Flares not yet occurred at the current day
- Coloured dots: Flares that have occurred, coloured by energy (hot colormap)
- Red vertical line: Current day marker

**Right panel — Atmospheric state:**
- Number of flares that have occurred vs. remaining
- Cumulative fractional change in O₃, CH₄, N₂O, NO₂ at the selected day (with partial recovery accounted for)

**Scientific takeaway shown in this tab:** The spectral change per step is sub-ppm — below the current JWST systematic noise floor of ~20 ppm — meaning that individual flare events do not produce detectable spectral changes at the current observing programme depth. Only the cumulative effect over years-to-decades of monitoring would become measurable.

---

## Dashboard: Tab 7 — Biosphere Engine

**What it shows:** The biosphere model outputs from Stage 2 and Stage 3 — the CO consumption and methanogenesis feedback loops and their spectral consequences.

**Top panel — Biosphere-on vs biosphere-off spectra:**
Two transmission spectra for Atmosphere A overlaid:

- Orange line: Biosphere-off (abiotic CO ~1×10⁻⁷ VMR, no biogenic CH₄)
- Green line: Biosphere-on (biotic CO ~3×10⁻⁹ VMR, biogenic CH₄ enhanced)
- Shaded difference: The biological signal — the region between the two spectra that JWST would need to detect to distinguish life from no life

Key band annotations: CO 4.67 µm, CH₄ 3.3 µm, CO₂ 4.3 µm.

**Middle panel — CO suppression history:**
If `biosphere_state.csv` from Stage 2 is available, a line plot shows the biotic CO VMR over the 79-day window alongside the abiotic baseline. The gap between the two lines represents the biological CO drawdown at each point in time, which narrows temporarily after large UV flares that stress the biosphere.

**Bottom panel — Detection horizon:**
A bar chart showing how many JWST transits of TRAPPIST-1e would be needed to detect each key biosignature feature at 5-sigma confidence, computed by `compute_detection_horizon()` in `twin_core.py`. For each species the two bars compare current (post-79-day) amplitude versus the required amplitude. The CO biosphere suppression signal requires ~100 transits; O₃ at 9.6 µm requires ~120 transits with the 3D model caveat.

**Right panel — Biosphere stress event log:**
A table of all flares with EF_NUV > 100 that caused measurable biosphere stress, showing UV dose, population_factor before/after, and recovery timeline.

---

## Dashboard: Tab 8 — Forward Prediction

**What it shows:** A Monte Carlo projection of where the twin's atmosphere and biosphere will be in the future, given continued stellar flaring at the observed rate.

**Controls (within-tab):**

- Prediction horizon slider: 1 month to 10 years
- Flare rate multiplier: scale the Vida+2017 rate (0.1× to 3×) to explore optimistic/pessimistic scenarios
- Number of Monte Carlo runs: 10 to 500 (more = smoother uncertainty bands, slower)

**Main prediction panel:**
Four sub-panels showing ensemble mean ± 1σ over the prediction horizon for:

1. O₃ VMR (with 1D prediction band and the Ridgway+2023 3D alternative shown as a dashed upper bound)
2. CH₄ VMR (biotic vs. abiotic trajectories)
3. Biosphere population_factor (showing survival probability vs. time)
4. Biotic CO VMR (key spectral indicator)

Each sub-panel also marks the JWST 5-sigma detection threshold as a horizontal reference line so you can see whether the ensemble crosses into detectable territory.

**Right panel — Scenario comparison:**
Two frozen scenarios can be locked (e.g., "high flare rate" vs. "low flare rate") and their O₃ trajectories overlaid for comparison. A summary table shows the 1-year and 5-year predicted atmospheric state for each scenario, with the posterior-weighted most-likely scenario highlighted.

**Bayesian posterior display:**
If JWST observations have been assimilated via the sidebar uploader, the posterior weights for each atmosphere scenario are shown as a bar chart. As more observations are added, the posterior narrows toward the most consistent scenario — this is the twin's state-estimation loop in action.

---

## Key Scientific Caveats

All results carry important model-dependent uncertainties:

1. **1D vs. 3D tension:** The most important caveat in the entire pipeline. Ridgway et al. 2023 used a 3D Global Climate Model (GCM) and found that flares *increase* O₃ by up to 20× on Proxima Cen b-like planets — the exact opposite of what 1D models predict. The 1D result presented here should be treated as a lower bound on O₃ survivability and an upper bound on O₃ destruction.

2. **Stellar energetic particles (SEPs) not modelled:** Segura et al. 2010 showed that the *particle* component of flares dominates the chemistry. UV-only gives ~5% O₃ loss; UV + SEPs gives 94% loss. SEPs require magnetohydrodynamic modelling of TRAPPIST-1's coronal mass ejections.

3. **No magnetic field shielding:** TRAPPIST-1e's intrinsic magnetic field (unknown) would partially deflect SEPs.

4. **Photochemical LUTs are approximations:** The lookup tables interpolated from Segura et al. 2010 and Tilley et al. 2019 were derived for different stellar and planetary systems. TRAPPIST-1e's specific parameters may shift the response curves.

5. **DREAMS spectrum is synthetic:** The JWST comparison spectrum is generated from a noise model, not real data. The chi-squared results should be interpreted qualitatively, not as definitive atmospheric constraints.

---

## File Structure Summary

```
NAKA-Biosignature-Detection/
├── main.py                          Stage 1: Data ingestion
├── stage2_atmospheric_response.py   Stage 2: Photochemistry + biosphere
├── stage3_spectral_generation.py    Stage 3: Spectra + biosphere-on/off
├── stage4_dashboard.py              Stage 4: Calibration + digital twin dashboard
├── twin_core.py                     Shared: digital twin state engine (all stages import this)
├── twin_state.json                  Live twin state (written by S1, updated by S2; read by S3/S4)
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
│   └── biosphere_state.csv          Per-flare: population_factor, UV dose, biotic CO, biogenic CH4
│
├── Stage 3 Files/
│   ├── spectrum_A_quiescent.csv (+ 6 more individual spectra)
│   ├── all_spectra_combined.csv
│   ├── feature_amplitudes.csv
│   ├── spectrum_atmA_biosphere_on.csv   Living-world spectrum (suppressed CO, biogenic CH4)
│   ├── spectrum_atmA_biosphere_off.csv  Abiotic baseline spectrum (full CO, no biogenic CH4)
│   ├── biosphere_spectral_diff.csv      Difference = pure biological signal
│   └── stage3_spectra_plot.png
│
└── Stage 4 Files/
    ├── jwst_dreams_synthetic.csv
    ├── calibration_results.csv
    └── stage4_summary.png
```
