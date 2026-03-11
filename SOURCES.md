# NAKA Biosignature Detection — Sources & Usage

All literature and data sources used in the TRAPPIST-1e Digital Twin pipeline, with specific notes on what data, values, or methods were taken from each source.

---

## Observational Data

### NASA Exoplanet Archive
**Used in:** Stage 1 (`main.py`), Stage 3 (`stage3_spectral_generation.py`)

All fundamental system parameters for TRAPPIST-1 and TRAPPIST-1e are drawn from the Archive:

| Parameter | Value | Used for |
|---|---|---|
| TRAPPIST-1 TIC ID | 267519185 | TESS data query |
| Spectral type | M8V | Classification |
| T_eff | 2566 K | Stellar blackbody; UV enhancement ratios |
| Stellar luminosity | 5.22×10⁻⁴ L☉ | Flare peak amplitude scaling |
| R_star | 0.1192 R☉ | Transit depth continuum; scale height |
| M_star | 0.0898 M☉ | System parameter |
| Distance | 12.43 pc | Context |
| Rotation period | 3.30 days | Context |
| Stellar age | 7.6 Gyr | Context |
| TRAPPIST-1e radius | 0.920 R⊕ | R_p in transit depth formula |
| TRAPPIST-1e mass | 0.692 M⊕ | Surface gravity calculation |
| Orbital period | 6.101013 days | Context |
| Semi-major axis | 0.02928 AU | Flux scaling (R_star/a)² |
| T_eq | 251 K | Atmospheric temperature |
| Insolation | 0.662 S⊕ | Context |

---

### TESS / lightkurve (NASA)
**Used in:** Stage 1 (`main.py`) — `download_tess_lightcurve()`

The Stage 1 code uses the `lightkurve` Python package to download the TESS Sector 10 light curve for TRAPPIST-1 (TIC 267519185), specifically the TGLC (TESS-SPOC) product. This gives ~79 days of photometry at 1800-second cadence. The light curve is stored and plotted as a diagnostic, but its cadence is explicitly noted as too coarse for direct flare detection on TRAPPIST-1 (flares last 1–30 minutes). The 79-day observing baseline from this dataset is carried through to set the time window for the synthetic flare catalog.

---

## Flare Statistics & Templates

### Vida et al. 2017
**Full citation:** Vida, K., Kővári, Zs., Pál, A., Oláh, K., & Kriskovics, L. (2017). *Frequent flaring in the TRAPPIST-1 system — unsuited for life?* ApJ, 841, 124.

**Used in:** Stage 1 (`main.py`) — `generate_flare_catalog()`; Stage 2 (`stage2_atmospheric_response.py`) — flare energy range

Provides the **flare frequency distribution (FFD)** for TRAPPIST-1:
- Flare rate: **0.53 flares per day** (adopted as the Poisson rate parameter)
- Power-law index: **α = 1.72** (dN/dE ∝ E^−α)
- Energy range: 10³⁰ to 5×10³³ erg

These three numbers drive the entire synthetic flare catalog. The number of flares over 79 days is Poisson-sampled from rate × time, and individual energies are drawn from the inverse CDF of the power-law distribution using the formula: `E = (u × (E_max^b − E_min^b) + E_min^b)^(1/b)` where `b = 1 − α`.

---

### Ducrot et al. 2022
**Full citation:** Ducrot, E., et al. (2022). *The Flaring Activity of TRAPPIST-1 with ESPRESSO, TESS, and CHEOPS.* A&A, 664, A95.

**Used in:** Stage 1 (`main.py`) — `generate_flare_catalog()`; Stage 2 (`stage2_atmospheric_response.py`) — UV enhancement; throughout dashboard

Provides **TRAPPIST-1-specific flare blackbody temperatures** measured from multi-band photometry:
- Temperature range: **5300–8600 K** (adopted as the uniform sampling range)
- Adopted median: **T_flare = 7000 K** (used for all per-flare UV enhancement calculations)
- Superflare rate: ~5 per year (context)

The temperature of 7000 K is critical because it sets the UV-to-optical colour ratio of flares. All NUV and FUV enhancement factors in Stage 2 are computed as the Planck blackbody ratio B(λ, 7000K) / B(λ, 2566K), making this temperature choice the most important single parameter in the photochemistry chain.

---

### Davenport 2014
**Full citation:** Davenport, J.R.A. (2014). *The Kepler Catalog of Stellar Flares.* ApJ, 797, 122.

**Used in:** Stage 1 (`main.py`) — `build_flare_template()`; Stage 2 (`stage2_atmospheric_response.py`) — `davenport_profile()`; Stage 3 — flare template context

Provides two things:

**1. Empirical white-light flare profile (Table 1):**
The shape of a flare light curve as a function of normalised time φ = (t − t_peak) / t_half:
- Rise phase (−1 ≤ φ < 0): `F(φ) = 1 + 1.941φ − 0.175φ² − 2.246φ³ − 1.125φ⁴`
- Decay phase (φ ≥ 0): `F(φ) = 0.6890 × exp(−1.600φ) + 0.3030 × exp(−0.2783φ)`

These exact coefficients are used in both `build_flare_template()` (Stage 1) and `davenport_profile()` (Stage 2) to compute the time-varying UV exposure during each flare event.

**2. Duration–energy scaling relation:**
`τ_flare ∝ E^0.39`

Used to derive flare durations from sampled energies: `duration = C × E^0.39` where the constant C is normalised so that E = 10³² erg gives a 10-minute duration. This appears in `generate_flare_catalog()` (Stage 1, line 91).

---

## Stellar UV Observations

### France et al. 2020 (MegaMUSCLES)
**Full citation:** France, K., et al. (2020). *The MUSCLES Treasury Survey V: Scaling Relations and Predictions for Broadly Characterizing the UV and X-ray Environments of Exoplanets.* ApJS, 247, 25.

**Used in:** Stage 2 (`stage2_atmospheric_response.py`) — `MEGA` dict; UV enhancement calibration

Provides the **quiescent UV flux baseline at TRAPPIST-1e** from the Measurements of the Ultraviolet Spectral Characteristics of Low-mass Exoplanetary Systems (MUSCLES) survey:
- F_NUV (200–320 nm): **0.070 erg s⁻¹ cm⁻²**
- F_FUV (100–200 nm): **0.003 erg s⁻¹ cm⁻²**
- F_Lyα (121.6 nm): **0.025 erg s⁻¹ cm⁻²**

These values anchor the quiescent photochemical baseline. Per-flare NUV fluences are computed as `F_NUV × EF_NUV × duration`, where EF_NUV is the Planck ratio enhancement factor. The cumulative NUV dose over 79 days is then compared against the photochemical LUT thresholds from Segura et al. 2010 and Tilley et al. 2019.

---

### Wilson et al. 2025 (MegaMUSCLES)
**Full citation:** Wilson, D.J., et al. (2025). *The MegaMUSCLES Survey: Panchromatic Spectral Energy Distributions of M and K Dwarf Planet Hosts.* (in prep / arXiv)

**Used in:** Stage 2 (`stage2_atmospheric_response.py`) — `MEGA` dict (cited alongside France+2020)

Co-cited with France et al. 2020 as the updated MegaMUSCLES SED catalogue. The same UV flux values are adopted from this combined dataset.

---

### Peacock et al. 2019
**Full citation:** Peacock, S., et al. (2019). *Predicting the Extreme Ultraviolet Environment of Exoplanets around Low-Mass Stars.* ApJ, 886, 77.

**Used in:** Stage 2 (`stage2_atmospheric_response.py`) — `MEGA` dict (cited alongside France+2020 and Wilson+2025)

Co-cited as part of the MegaMUSCLES UV flux measurement ensemble for TRAPPIST-1. Cross-validates the FUV and NUV flux values used in the quiescent baseline.

---

### Kowalski et al. 2013
**Full citation:** Kowalski, A.F., et al. (2013). *Time-Resolved Properties and Global Trends in dMe Flares from Simultaneous Photometry and Spectra.* ApJS, 207, 15.

**Used in:** Stage 2 (`stage2_atmospheric_response.py`) — `F_UV_NUV`, `F_UV_FUV` constants

Provides the **UV energy fractions of the bolometric flare energy** from M-dwarf flare spectroscopy:
- NUV fraction (200–320 nm): **f_NUV = 2.5%** of E_bolometric
- FUV fraction (100–200 nm): **f_FUV = 0.2%** of E_bolometric

These fractions are used to partition a flare's total bolometric energy (sampled from the Vida+2017 FFD) into the individual UV bands that drive specific photochemical reactions. Cross-validated against Hawley et al. 2003 and Segura et al. 2010.

---

### Hawley et al. 2003
**Full citation:** Hawley, S.L., et al. (2003). *Flares in Late-Type Stars: Constraints on the Stellar Atmosphere, Coronal Structure, and Particle Acceleration.* ApJ, 597, 535.

**Used in:** Stage 2 (`stage2_atmospheric_response.py`) — cited alongside Kowalski+2013 for UV energy fractions

Co-cited with Kowalski et al. 2013 as an earlier observational basis for M-dwarf flare UV energy partitioning. The Kowalski+2013 values supersede these but Hawley+2003 provides the historical context for the adopted fractions.

---

## Atmospheric Compositions & JWST Constraints

### Espinoza et al. 2025 (DREAMS)
**Full citation:** Espinoza, N., et al. (2025). *A Search for Biosignatures in TRAPPIST-1e's Atmosphere with JWST NIRSpec/PRISM (DREAMS).* (Nature / arXiv)

**Used in:** Stage 2 — atmospheric composition definitions; Stage 4 — synthetic DREAMS spectrum, chi-squared calibration; throughout dashboard

The single most important observational anchor in the pipeline. Used in three distinct ways:

1. **Atmospheric composition definitions (Stage 2):** The DREAMS result that rules out H₂-dominated primary atmospheres constrains which of the three candidate compositions are viable. Atm A (N₂-Earth) is labelled "Permitted", Atm B (CO₂-Venus) "Disfavoured", and Atm C (thin rock) "Possible" based on the DREAMS data.

2. **Synthetic DREAMS spectrum (Stage 4):** The observed spectrum is featureless and flat, consistent with no thick atmosphere. This is modelled as a flat transit depth (~5000 ppm) with a weak stellar contamination slope (+0.5 ppm/µm) and photon noise scaled to 5 transit observations.

3. **Chi-squared calibration target:** All three predicted atmospheric spectra are compared against this synthetic spectrum, quantifying which model best matches the real JWST result.

---

### Schwieterman et al. 2018
**Full citation:** Schwieterman, E.W., et al. (2018). *Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life.* Astrobiology, 18(6), 663–708.

**Used in:** Stage 2 — Atm A reference composition; Stage 3 — cross-section calibration targets

Two uses:

1. **Atmosphere A composition:** The N₂-dominated Earth-like VMRs (O₃ 300 ppb, CH₄ 1.8 ppm, N₂O 320 ppb) are drawn from the Earth-analog reference atmospheres in this review.

2. **Cross-section calibration:** The Stage 3 `BAND_PARAMS` Gaussian amplitudes for H₂O, CO₂, CH₄, O₃, and N₂O are cross-referenced against Table 2 and associated figures in this review to ensure simulated feature depths match Earth/Venus observations at R=100 resolution.

---

### Miranda-Rosete et al. 2024
**Full citation:** Miranda-Rosete, A., et al. (2024). *Abiotic Production of Ozone by Stellar Flares in CO₂ Atmospheres.* arXiv:2308.01880.

**Used in:** Stage 2 — Atm B false-positive pathway; `photochem_response_atm_b.csv` (`dO3_abiotic` column)

The scientific basis for the **abiotic O₃ false-positive scenario** in Atmosphere B (CO₂-Venus). The paper shows that TRAPPIST-1-type superflares can drive CO₂ photolysis: CO₂ → CO + O, with subsequent O + O₂ → O₃. This creates detectable O₃ signatures without biology.

The pipeline implements this as a separate `dO3_abiotic` column (absolute VMR addition, not a fractional change of existing O₃) in Stage 2's photochemical response for Atm B. In Stage 3, `apply_flare_changes()` handles the `_abiotic` suffix with addition rather than multiplication, reflecting that this is newly synthesised O₃, not modification of an existing reservoir.

---

### Turbet et al. 2018
**Full citation:** Turbet, M., et al. (2018). *Modeling Climate and Habitability on TRAPPIST-1 Planets.* A&A, 612, A86.

**Used in:** Stage 2 — Atm C (thin remnant) definition

Provides theoretical context for the bare-rock/thin-atmosphere scenario for inner TRAPPIST-1 planets. The Stage 2 Atm C definition (10 mbar, 98% N₂, 97% UV surface transmittance) is based on the atmospheric escape and runaway greenhouse scenarios described here.

---

### Johnstone et al. 2021
**Full citation:** Johnstone, C.P., et al. (2021). *Upper Atmospheres and Stellar Winds of Cool Stars.* A&A, 649, A96.

**Used in:** Stage 2 — Atm C (thin remnant) definition, cited alongside Turbet+2018

Co-cited as an additional reference for the atmospheric loss/thin-remnant scenario, specifically addressing how M-dwarf stellar winds and high-energy radiation can erode secondary atmospheres over gigayear timescales.

---

### Grimm et al. 2018
**Full citation:** Grimm, S.L., et al. (2018). *The Nature of the TRAPPIST-1 Exoplanets.* A&A, 613, A68.

**Used in:** Stage 3 (`stage3_spectral_generation.py`) — surface gravity

The code comment on line 96 cites Grimm+2018 for the TRAPPIST-1e surface gravity of **8.0 m/s² (803 cm/s² in CGS)**, derived from transit timing variations. This value appears in the scale height computation `H = k_B T / (μ g)` which sets the amplitude of all atmospheric spectral features.

---

## Photochemical Models

### Segura et al. 2010
**Full citation:** Segura, A., et al. (2010). *The Effect of a Strong Stellar Flare on the Atmospheric Chemistry of an Earth-like Planet Orbiting an M dwarf.* Astrobiology, 10(7), 751–771.

**Used in:** Stage 2 — primary O₃ depletion LUT; Stage 4 — NUV fluence threshold marker on dashboard; throughout as key 1D reference

The primary source for the **O₃ photochemical response lookup table**. Specific results extracted:
- UV-only flare scenario: **~5% O₃ depletion** (AD Leo 1985 flare equivalent)
- UV + stellar energetic particles (SEPs): **up to 94% O₃ loss** over 2 years
- NUV fluence threshold for significant O₃ response: marked as reference line in Stage 4 Tab 1 (right panel)

The Stage 2 LUT interpolates between these benchmarks as a function of per-flare NUV fluence. The AD Leo 1985 superflare (E ≈ 10³⁴ erg) serves as the upper anchor of the interpolation.

Important caveat: This is a **1D model** that does not capture latitudinal transport or GCM dynamics.

---

### Tilley et al. 2019
**Full citation:** Tilley, M.A., et al. (2019). *Modeling Repeated M Dwarf Flaring at an Earth-like Planet in the Habitable Zone: Atmospheric Effects for an Unmagnetized Planet.* Astrobiology, 19(1), 64–86.

**Used in:** Stage 2 — cumulative O₃ depletion LUT

Provides the **cumulative flaring scenario** context. Key result used in the pipeline:
- Repeated flaring over 10 years reduces O₃ to **~6% of its initial value** (assuming no magnetic shielding)
- Recovery timescale between events: the Stage 2 exponential recovery τ_O3 = 30 days is calibrated to be broadly consistent with the recovery periods implied by this work

This source sets the upper bound on cumulative damage used when scaling per-flare depletions from Segura+2010 to multi-event scenarios.

---

### Chen et al. 2021
**Full citation:** Chen, H., et al. (2021). *Persistence of Flare-driven Atmospheric Chemistry on Rocky Habitable Zone Worlds.* Nature Astronomy, 5, 298–310.

**Used in:** Stage 2 — NO₂/N₂O/HNO₃ enhancement LUT; Stage 4 Tab 3 (NO₂ enhancement subplot and caption)

Provides the **3D CESM1/WACCM result** that flares can *enhance* certain atmospheric species — a critical counterpoint to the O₃-depletion narrative. Specific results used:
- **NO₂, N₂O, and HNO₃ increase** after repeated flaring events (not decrease)
- New chemical equilibria are established that differ fundamentally from a pre-flare state
- These increases are captured in the Stage 2 `dNO2` and `dHNO3` columns being positive values in `photochem_response_atm_a.csv`

The Stage 4 biosignature vulnerability tab explicitly labels the NO₂ subplot with the Chen+2021 finding. This is used to argue that flares do not simply damage biosignature detectability — they may actually create new detectable nitrogen-bearing signatures.

---

### Ridgway et al. 2023
**Full citation:** Ridgway, R.J., et al. (2023). *3D Effects of Stellar Flares on the Atmospheric Chemistry of Habitable Zone Rocky Planets.* MNRAS, 518(2), 2472–2491.

**Used in:** Stage 2 — flagged throughout as key 1D vs. 3D tension; Stage 4 — warning labels on every O₃ panel

Not used to derive numerical values, but cited as **the most important scientific caveat** in the entire pipeline. Key result:
- The Met Office Unified Model 3D GCM predicts that flares **INCREASE O₃ by up to 20×** on Proxima Cen b-like planets
- This is the exact opposite of what 1D models (Segura+2010, Tilley+2019) predict

The pipeline explicitly flags this tension at every point where O₃ depletion results are shown. The Stage 4 dashboard displays the warning: *"[!] Ridgway+2023 3D: O3 increases 20×"* as a red annotation on the O₃ vulnerability scatter plot, and the metric card shows "1D model (Ridgway+2023: +2000%)" as a delta. This is treated as an unresolved model-dependent discrepancy, not a settled question.

---

### Sander et al. 2011 (JPL Publication 10-6)
**Full citation:** Sander, S.P., et al. (2011). *Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation 17.* JPL Publication 10-6, Jet Propulsion Laboratory.

**Used in:** Stage 2 — VULCAN-analog ODE rate constants

The rate constants for the Chapman cycle and HOx chemistry implemented in the Stage 2 `solve_ivp` ODE system are taken from this JPL compilation, evaluated at **T = 250 K** (TRAPPIST-1e equilibrium temperature). Reactions implemented:
- Chapman cycle: O₃ + hν → O + O₂; O + O₂ → O₃; O + O₃ → 2O₂
- HOx: OH + O₃ → HO₂ + O₂; HO₂ + O₃ → OH + 2O₂
- CH₄ oxidation: OH + CH₄ → CH₃ + H₂O

---

## Transmission Spectroscopy

### Lecavelier des Etangs et al. 2008
**Full citation:** Lecavelier des Etangs, A., et al. (2008). *Rayleigh Scattering in the Transit Spectrum of HD 189733b.* A&A, 481, L83–L86.

**Used in:** Stage 3 (`stage3_spectral_generation.py`) — core radiative transfer formula; Stage 4 docstring

The entire Stage 3 radiative transfer implementation is built on the **chord optical depth formalism** from this paper. The key equation used:

```
delta(λ) = (R_p / R_star)²  +  2 R_p H / R_star²  ×  ln(τ₀(λ) × 2/3)
```

where τ₀(λ) is the reference chord optical depth:
```
τ₀(λ) = Σᵢ [ Xᵢ × σᵢ(λ) × n₀ × H × √(2π R_p / H) ]
```

This formalism allows computation of the transit depth spectrum from first principles given VMRs, cross-sections, scale height, and planetary geometry, without needing to run a full line-by-line radiative transfer code. It is the same physics underlying petitRADTRANS and PICASO, but self-contained.

---

### Fortney 2005
**Full citation:** Fortney, J.J. (2005). *The Effect of Condensates on the Characterization of Transiting Planet Atmospheres with Transmission Spectroscopy.* MNRAS, 364, 649–653.

**Used in:** Stage 3 — analytic transit spectrum formalism context

Cited as a foundational reference for analytic transmission spectra. The Lecavelier+2008 formula used in Stage 3 builds directly on the scale-height parameterisation introduced here. Not used for specific numerical values.

---

### Seager & Sasselov 2000
**Full citation:** Seager, S. & Sasselov, D.D. (2000). *Theoretical Transmission Spectra during Extrasolar Giant Planet Transits.* ApJ, 537, 916–921.

**Used in:** Stage 3 — cited as foundational reference for transmission spectroscopy

The originating paper for the concept of measuring atmospheric composition through transit depth as a function of wavelength. The framework developed here underlies all subsequent work including Fortney 2005 and Lecavelier+2008. Not used for specific numerical values in the pipeline.

---

## Molecular Spectroscopy

### Gordon et al. 2022 (HITRAN2020)
**Full citation:** Gordon, I.E., et al. (2022). *The HITRAN2020 Molecular Spectroscopic Database.* JQSRT, 277, 107949.

**Used in:** Stage 3 (`stage3_spectral_generation.py`) — `BAND_PARAMS` dict; all cross-section band centres and peak values

The primary database for all molecular cross-section parameterisations in Stage 3. The `BAND_PARAMS` dictionary lists Gaussian bands centred at HITRAN2020 line-group centres for eight species: H₂O, CO₂, CH₄, O₃, N₂O, NO₂, CO, O₂, plus SO₂ and HCl for Atm B. Specifically:
- All band centre wavelengths (e.g., CH₄ 3.3 µm, O₃ 9.6 µm, CO₂ 4.3 µm) come from HITRAN2020
- Peak cross-section values are calibrated so that simulated feature amplitudes at R=100 match known Earth/Venus depths
- The Gaussian widths are set to match R=100 PRISM resolution elements

---

### Rothman et al. 2010
**Full citation:** Rothman, L.S., et al. (2010). *HITEMP, the High-Temperature Molecular Spectroscopic Database.* JQSRT, 111(15), 2139–2150.

**Used in:** Stage 3 (`stage3_spectral_generation.py`) — H₂O cross-section bands (cited in comments)

Cited specifically for the H₂O cross-section parameters used in the `BAND_PARAMS` H₂O entry. The strong H₂O features at 1.38 µm, 1.87 µm, 2.7 µm and 6.27 µm have peak values cross-referenced against HITEMP line-by-line integrals at T~270 K.

---

## JWST Calibration & Stellar Contamination

### Rackham et al. 2018
**Full citation:** Rackham, B.V., Apai, D., & Giampapa, M.S. (2018). *The Transit Light Source Effect: False Spectral Features and Incorrect Densities for M-Dwarf Transiting Planets.* ApJ, 853, 122.

**Used in:** Stage 4 (`stage4_dashboard.py`) — synthetic DREAMS spectrum construction (`build_dreams_spectrum()`)

Provides the theoretical basis for the **stellar contamination correction** applied to the synthetic DREAMS spectrum. When TRAPPIST-1 has unocculted star spots or faculae, the transit depth becomes wavelength-dependent in a way that mimics atmospheric features. The Stage 4 synthetic spectrum adds a linear contamination slope of **+0.5 ppm/µm** from 0.6 to 5.3 µm, representing the contribution of unocculted cool star spots (T_spot ~ 2400 K, f_spot ~ 50%) on TRAPPIST-1's highly spotted surface. This is the dominant systematic in real TRAPPIST-1 JWST spectra.

---

## Summary Table

| Source | Stage(s) | Specific values / functions extracted |
|---|---|---|
| NASA Exoplanet Archive | 1, 3, 4 | All TRAPPIST-1/1e system parameters |
| TESS / lightkurve | 1 | Sector 10 light curve download |
| Vida et al. 2017 | 1 | FFD rate 0.53/day, α=1.72, E range |
| Ducrot et al. 2022 | 1, 2 | T_flare = 7000 K, range 5300–8600 K |
| Davenport 2014 | 1, 2 | Flare template coefficients; τ ∝ E^0.39 |
| France et al. 2020 | 2 | F_NUV = 0.070, F_FUV = 0.003 erg/s/cm² |
| Wilson et al. 2025 | 2 | MegaMUSCLES UV SED (co-cited) |
| Peacock et al. 2019 | 2 | MegaMUSCLES UV flux (co-cited) |
| Kowalski et al. 2013 | 2 | f_NUV = 2.5%, f_FUV = 0.2% of E_bol |
| Hawley et al. 2003 | 2 | UV fraction cross-validation (co-cited) |
| Espinoza et al. 2025 | 2, 4 | Atmospheric constraints; synthetic DREAMS |
| Schwieterman et al. 2018 | 2, 3 | Atm A VMRs; cross-section calibration |
| Miranda-Rosete et al. 2024 | 2, 3 | Abiotic O₃ pathway in CO₂ atmospheres |
| Turbet et al. 2018 | 2 | Thin atmosphere / bare rock scenario |
| Johnstone et al. 2021 | 2 | Atmospheric escape / thin remnant |
| Grimm et al. 2018 | 3 | g = 8.0 m/s² (803 cm/s²) |
| Segura et al. 2010 | 2, 4 | O₃ depletion LUT; 5% UV-only, 94% UV+SEP |
| Tilley et al. 2019 | 2 | Cumulative O₃ reduction to 6%; recovery τ |
| Chen et al. 2021 | 2, 4 | NO₂/N₂O/HNO₃ increase from flares |
| Ridgway et al. 2023 | 2, 4 | 3D: O₃ increases 20× (key caveat) |
| Sander et al. 2011 (JPL) | 2 | Rate constants at T=250K for ODE |
| Lecavelier des Etangs+2008 | 3 | Core RT formula (chord optical depth) |
| Fortney 2005 | 3 | Analytic transit spectrum formalism |
| Seager & Sasselov 2000 | 3 | Foundational transmission spectroscopy |
| Gordon et al. 2022 (HITRAN2020) | 3 | All band centres and cross-sections |
| Rothman et al. 2010 (HITEMP) | 3 | H₂O cross-section parameters |
| Rackham et al. 2018 | 4 | Stellar contamination slope model |
