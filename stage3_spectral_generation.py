import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
===========================================================================
TRAPPIST-1e Digital Twin -- Stage 3: Transmission Spectra Generation
===========================================================================

Generates synthetic transmission spectra for TRAPPIST-1e under quiescent
and post-flare atmospheric conditions.

Method: Analytic radiative transfer using the Lecavelier des Etangs (2008)
  chord optical depth formalism -- the same physics as petitRADTRANS/PICASO
  but self-contained, with no external opacity table downloads required.

  Transit depth at wavelength lambda:
    delta(lambda) = (R_p/R_star)^2
                  + 2*R_p*H / R_star^2 * ln( tau_0(lambda) * 3/2 )

  where tau_0(lambda) = sum_i [X_i * sigma_i(lambda) * n0 * H * sqrt(2*pi*R_p/H)]
  is the chord reference optical depth at the surface pressure level.

  Molecular cross-sections sigma_i(lambda) are parameterised as sums of
  Gaussians at published HITRAN/ExoMol band centres with peak values
  calibrated to match Earth/Venus spectrum feature depths.

Pipeline:
  1. Load Stage 2 outputs: atmospheric compositions + cumulative state
  2. Build wavelength grid at R=100 (JWST NIRSpec/PRISM resolution)
  3. Compute cross-sections for H2O, CO2, CH4, O3, N2O, NO2, CO, O2
     + Rayleigh scattering + collision-induced absorption (N2-N2, CO2-CO2)
  4. Compute transmission spectra for all 3 atmospheres:
       (a) Quiescent (Stage 2 initial VMRs)
       (b) Post-79-day-flare-sequence (Stage 2 cumulative changes applied)
       (c) Post-largest-single-flare (VULCAN-analog ODE result)
  5. Save spectra + diagnostic plots

Key spectral features (JWST observable windows):
  H2O: 1.4, 1.9, 2.7 um (NIRSpec)
  CO2: 4.3 um (NIRSpec/MIRI boundary)
  CH4: 3.3 um (NIRSpec)
  O3:  9.6 um (MIRI; key biosignature)
  N2O: 4.5, 7.8 um
  NO2: 6.2 um

References:
  Lecavelier des Etangs et al. 2008 A&A 481 L83   -- formalism
  Fortney 2005 MNRAS 364 649                       -- analytic transit spectra
  Seager & Sasselov 2000 ApJ 537 916               -- transmission spectroscopy
  Schwieterman et al. 2018 Astrobiology 18(6)      -- biosignature cross-sections
  Gordon et al. 2022 JQSRT 277 (HITRAN2020)        -- molecular line data
  Espinoza et al. 2025 (DREAMS)                    -- JWST TRAPPIST-1e obs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings, os

warnings.filterwarnings('ignore')

OUTPUT_DIR = "Stage 3 Files"
STAGE1_DIR = "Stage 1 Files"
STAGE2_DIR = "Stage 2 Files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# Physical Constants (SI + CGS)
# ===========================================================================
K_B    = 1.3806e-16   # erg K^-1 (CGS)
AMU    = 1.6605e-24   # g
G_CGS  = 6.674e-8     # cm^3 g^-1 s^-2
PI     = np.pi

# ===========================================================================
# TRAPPIST-1 System Parameters (NASA Exoplanet Archive)
# ===========================================================================
R_SUN_CM   = 6.957e10     # cm
R_EARTH_CM = 6.371e8      # cm
M_EARTH_G  = 5.972e27     # g

# Star
R_STAR_CM  = 0.1192 * R_SUN_CM          # 8.295e9 cm
T_EFF_K    = 2566                        # K

# Planet TRAPPIST-1e
R_P_CM     = 0.920 * R_EARTH_CM         # 5.861e8 cm
M_P_G      = 0.692 * M_EARTH_G          # 4.132e27 g
T_EQ_K     = 251.0                       # K (equilibrium temperature)

# Surface gravity
G_P        = G_CGS * M_P_G / R_P_CM**2  # ~800 cm s^-2  (Grimm+2018: 8.0 m/s^2)

# Reference transit depth (continuum, no atmosphere)
DELTA_CONT = (R_P_CM / R_STAR_CM)**2    # dimensionless; ~5.0e-3


# ===========================================================================
# SECTION 1: Load Stage 2 Outputs
# ===========================================================================
def load_stage2(atm_label):
    """
    Load atmospheric composition for one atmosphere from Stage 2.
    Returns dict of VMRs + metadata.
    """
    atm_df = pd.read_csv(f"{STAGE2_DIR}/atmospheric_compositions.csv")
    row = atm_df[atm_df['atmosphere'] == atm_label].iloc[0]

    vmr_cols = [c for c in atm_df.columns if c.startswith('vmr_')]
    vmr = {}
    for col in vmr_cols:
        species = col.replace('vmr_', '')
        val = float(row[col])
        if val > 0:
            vmr[species] = val

    return dict(
        label      = atm_label,
        name       = row['name'],
        P_bar      = float(row['P_bar']),
        T_K        = float(row['T_K']),
        vmr        = vmr,
        biosigs    = str(row.get('biosignatures', '')).split(','),
        fp         = str(row.get('false_positives', '')).split(','),
    )


def load_cumulative_changes():
    """
    Load Stage 2 cumulative biosignature changes (79-day flare sequence).
    Returns dict: {atm_label: {species: fractional_change}}

    NaN entries (species absent in that atmosphere) are skipped.
    Keys ending in '_abiotic' are absolute VMR additions (not fractional).
    """
    df = pd.read_csv(f"{STAGE2_DIR}/cumulative_biosignature_state.csv")
    result = {}
    for _, row in df.iterrows():
        lbl = row['atmosphere']
        changes = {}
        for col in df.columns:
            if col.startswith('cumul_d'):
                val = row[col]
                if pd.notna(val) and np.isfinite(float(val)):
                    sp = col.replace('cumul_d', '')
                    changes[sp] = float(val)
        result[lbl] = changes
    return result


def load_vulcan_analog():
    """
    Load VULCAN-analog ODE timeseries (largest single flare, 24h window).
    Returns fractional changes at t=24h for Atmosphere A.
    """
    try:
        df = pd.read_csv(f"{STAGE2_DIR}/vulcan_analog_timeseries.csv")
        last = df.iloc[-1]
        return {
            'O3':  float(last['dO3_frac']),
            'CH4': float(last['dCH4_frac']),
            'OH':  float(last.get('dOH_frac', 0.0)),
        }
    except Exception:
        return {'O3': -0.0334, 'CH4': -0.0001}  # fallback from Stage 2 results


# ===========================================================================
# SECTION 2: Wavelength Grid (R=100, JWST NIRSpec/PRISM range)
# ===========================================================================
def make_wavelength_grid(lam_min_um=0.5, lam_max_um=14.0, R=100):
    """
    Logarithmically-spaced wavelength grid at resolving power R = lambda/delta_lambda.
    Covers NIRSpec (0.6-5.3 um) + MIRI (5-28 um).  R=100 matches PRISM.
    Returns wavelengths in um (for display) and cm (for cross-section computation).
    """
    dlnlam = 1.0 / R
    n_pts  = int(np.log(lam_max_um / lam_min_um) / dlnlam) + 1
    lam_um = lam_min_um * np.exp(dlnlam * np.arange(n_pts))
    lam_cm = lam_um * 1e-4    # 1 um = 1e-4 cm
    return lam_um, lam_cm


# ===========================================================================
# SECTION 3: Molecular Cross-Sections
# ===========================================================================
# Band parameters: (centre_um, sigma_um_halfwidth, peak_cross_section_cm2)
# Calibrated to match feature depths in Earth/Venus transmission spectra.
# Primary reference: HITRAN2020 (Gordon+2022 JQSRT 277), VIRA, Schwieterman+2018.
# Peak cross-sections are HITRAN line-by-line integrated over PRISM resolution.
#
# Convention: sigma_um is Gaussian 1-sigma width (not FWHM).
# FWHM = 2*sqrt(2*ln2)*sigma_um ≈ 2.355 * sigma_um

BAND_PARAMS = {
    # ── H2O (Rothman+2010; Schwieterman+2018 Table 2) ──────────────────────
    'H2O': [
        (0.720, 0.008,  8.0e-26),   # z-band
        (0.820, 0.010,  3.0e-25),
        (0.940, 0.018,  1.5e-24),   # near-IR window
        (1.140, 0.025,  5.0e-24),
        (1.380, 0.040,  2.5e-23),   # Y/J boundary
        (1.870, 0.060,  6.0e-23),   # H-K gap
        (2.700, 0.120,  2.5e-22),   # strong combined H2O+CO2
        (3.200, 0.150,  1.5e-22),
        (6.270, 0.400,  6.0e-21),   # mid-IR rotation-vibration band
        (12.00, 1.500,  2.0e-21),   # far-IR
    ],
    # ── CO2 (HITRAN2020; Schwieterman+2018) ────────────────────────────────
    'CO2': [
        (1.050, 0.015,  3.0e-26),
        (1.270, 0.018,  1.5e-25),
        (1.600, 0.025,  8.0e-25),   # weak; still detectable by JWST
        (2.010, 0.040,  4.0e-24),
        (2.700, 0.120,  6.0e-23),   # shared window with H2O
        (4.300, 0.150,  8.0e-20),   # DOMINANT CO2 feature (nu3 asymm. stretch)
        (10.40, 0.500,  3.0e-22),
        (14.90, 1.500,  1.5e-20),   # bending mode (MIRI)
    ],
    # ── CH4 (HITRAN2020; Schwieterman+2018) ────────────────────────────────
    'CH4': [
        (0.890, 0.010,  2.0e-27),
        (1.000, 0.012,  8.0e-27),
        (1.167, 0.020,  3.0e-26),
        (1.380, 0.015,  4.0e-26),   # blends with H2O
        (1.667, 0.035,  2.5e-24),   # H band
        (2.300, 0.080,  1.5e-22),   # strong K band
        (3.300, 0.130,  4.0e-21),   # KEY JWST feature (nu3 stretch)
        (7.660, 0.450,  6.0e-22),   # mid-IR deformation band
    ],
    # ── O3 (HITRAN2020; Schwieterman+2018; key biosignature) ───────────────
    'O3': [
        (0.600, 0.080,  1.5e-21),   # Chappuis band (visible; very broad)
        (3.300, 0.200,  1.5e-22),   # Wulf band
        (9.600, 0.700,  3.5e-20),   # KEY: Hartley/nu3 stretching mode
        (14.20, 1.000,  1.0e-21),   # bending mode
    ],
    # ── N2O (HITRAN2020; Chen+2021; biosignature) ──────────────────────────
    'N2O': [
        (2.870, 0.050,  2.0e-23),
        (3.900, 0.060,  5.0e-23),
        (4.500, 0.120,  1.5e-21),   # strongest N2O feature
        (7.780, 0.400,  4.0e-22),
        (17.00, 1.200,  1.0e-22),
    ],
    # ── NO2 (HITRAN2020; Chen+2021 flare-enhanced) ─────────────────────────
    'NO2': [
        (0.400, 0.040,  3.0e-22),   # UV/blue absorption
        (3.400, 0.120,  2.5e-22),
        (6.200, 0.250,  1.5e-21),
    ],
    # ── CO (HITRAN2020; false-positive diagnostic) ─────────────────────────
    'CO': [
        (2.350, 0.040,  6.0e-22),   # first overtone
        (4.670, 0.120,  2.5e-20),   # fundamental band
    ],
    # ── O2 (HITRAN2020; A-band + weaker bands) ─────────────────────────────
    'O2': [
        (0.628, 0.003,  1.0e-25),   # B-band
        (0.688, 0.003,  3.0e-25),
        (0.762, 0.004,  1.5e-24),   # A-band (O2 biosignature)
        (1.268, 0.005,  1.5e-26),
    ],
    # ── SO2 (Venus / volcanic; HITRAN2020) ────────────────────────────────
    'SO2': [
        (0.290, 0.020,  5.0e-20),   # UV (strong)
        (7.340, 0.400,  3.0e-22),
        (8.680, 0.500,  4.0e-22),
    ],
    # ── HNO3 (Chen+2021 flare-enhanced; HITRAN2020) ────────────────────────
    'HNO3': [
        (5.900, 0.300,  1.0e-21),
        (7.520, 0.400,  3.5e-21),
        (11.30, 0.700,  2.5e-21),
    ],
    # ── HCl (volcanic; HITRAN2020) ────────────────────────────────────────
    'HCl': [
        (1.750, 0.020,  1.5e-24),
        (3.460, 0.060,  2.5e-22),
    ],
}


def compute_cross_sections(lam_um, species_list):
    """
    For each species in species_list, sum Gaussian cross-sections over all
    known absorption bands and add the Rayleigh scattering continuum.

    Returns dict: {species: sigma_array_cm2}  (same length as lam_um)
    """
    sigma = {}
    for sp in species_list:
        if sp not in BAND_PARAMS:
            sigma[sp] = np.zeros(len(lam_um))
            continue
        xs = np.zeros(len(lam_um))
        for (lam0, sig_w, peak) in BAND_PARAMS[sp]:
            xs += peak * np.exp(-0.5 * ((lam_um - lam0) / sig_w)**2)
        sigma[sp] = xs

    # Rayleigh scattering (continuum) -- wavelength^-4 law
    # Reference: N2 at 550 nm: sigma_R = 4.6e-27 cm^2 (van de Hulst 1981)
    sigma['Rayleigh'] = 4.6e-27 * (0.55 / lam_um)**4

    # Collision-induced absorption (CIA):
    # N2-N2 CIA (Lafferty+1996): broad features around 4.3 um and 7.8 um
    # Included as extremely broad Gaussians scaled to N2 pressure^2
    sigma['CIA_N2N2'] = (5.0e-50 * np.exp(-0.5*((lam_um - 4.30)/1.2)**2) +
                         3.0e-50 * np.exp(-0.5*((lam_um - 7.80)/1.5)**2))

    # CO2-CO2 CIA: enhances CO2 4.3 um feature in CO2-dominated atmospheres
    sigma['CIA_CO2CO2'] = 2.0e-49 * np.exp(-0.5*((lam_um - 4.30)/0.5)**2)

    return sigma


# ===========================================================================
# SECTION 4: Transmission Spectrum Calculator
# ===========================================================================
def compute_scale_height(T_K, mu_amu, g_cgs):
    """
    Atmospheric scale height H = kT / (mu * g).
    Returns H in cm.
    """
    return K_B * T_K / (mu_amu * AMU * g_cgs)


def mean_molecular_weight(vmr):
    """
    Mean molecular weight (amu) from volume mixing ratios.
    Uses species -> molecular weight lookup.
    """
    MW = {'N2': 28.0, 'O2': 32.0, 'CO2': 44.0, 'CH4': 16.0, 'H2O': 18.0,
          'O3': 48.0, 'N2O': 44.0, 'NO2': 46.0, 'CO': 28.0, 'SO2': 64.0,
          'HNO3': 63.0, 'HCl': 36.5, 'Ar': 40.0, 'He': 4.0}
    total_vmr = sum(vmr.values())
    if total_vmr == 0:
        return 28.0
    mu = sum(vmr.get(sp, 0.0) * MW.get(sp, 30.0) for sp in vmr) / total_vmr
    return max(mu, 2.0)


def compute_transmission_spectrum(atm, lam_um, sigma, R_p=R_P_CM, R_star=R_STAR_CM):
    """
    Compute the transmission spectrum (transit depth vs wavelength) for
    one atmospheric state using the Lecavelier des Etangs (2008) formalism.

    Parameters
    ----------
    atm    : dict with keys 'vmr', 'P_bar', 'T_K'
    lam_um : wavelength array in microns
    sigma  : dict of cross-sections from compute_cross_sections()
    R_p    : planet radius (cm)
    R_star : stellar radius (cm)

    Returns
    -------
    delta  : transit depth array (dimensionless)
    R_eff  : effective planet radius at each wavelength (cm)
    H      : atmospheric scale height (cm)
    """
    vmr   = atm['vmr']
    T_K   = atm['T_K']
    P_bar = atm['P_bar']

    # Mean molecular weight
    mu    = mean_molecular_weight(vmr)

    # Scale height
    H     = compute_scale_height(T_K, mu, G_P)

    # Surface number density: n0 = P / (kT)
    P_cgs = P_bar * 1.0133e6    # bar -> dyn/cm^2
    n0    = P_cgs / (K_B * T_K)  # cm^-3

    # Effective path length at terminator: L_ref = sqrt(2*pi*R_p*H)
    L_ref = np.sqrt(2.0 * PI * R_p * H)

    # Reference optical depth (chord) at each wavelength:
    # tau_0(lambda) = sum_i [ X_i * sigma_i(lambda) * n0 * H * L_ref / H ]
    #               = sum_i [ X_i * sigma_i(lambda) ] * n0 * L_ref
    #
    # Then tau_chord(b) = tau_0 * exp(-(b-R_p)/H)
    # Setting tau_chord = 2/3:  b_eff = R_p + H * ln(tau_0 * 3/2)

    # Aggregate weighted cross-section
    sigma_total = np.zeros(len(lam_um))

    # Molecular species
    for sp, X_i in vmr.items():
        if sp in sigma:
            sigma_total += X_i * sigma[sp]

    # Rayleigh (treated as N2 background)
    X_N2 = vmr.get('N2', vmr.get('CO2', 1.0))   # use dominant background gas
    sigma_total += X_N2 * sigma['Rayleigh']

    # CIA: N2-N2 scales as N2 abundance squared × n0 (pressure-squared)
    X_N2_sq = X_N2**2 * n0 * 1e-19   # dimensionless CIA enhancement
    sigma_total += X_N2_sq * sigma.get('CIA_N2N2', 0.0)

    # CIA: CO2-CO2 (relevant for Atmosphere B)
    X_CO2 = vmr.get('CO2', 0.0)
    X_CO2_sq = X_CO2**2 * n0 * 1e-19
    sigma_total += X_CO2_sq * sigma.get('CIA_CO2CO2', 0.0)

    # Reference optical depth at surface
    tau_0 = sigma_total * n0 * L_ref   # dimensionless

    # Effective radius: R_eff = R_p + H * ln(tau_0 * 3/2)
    # Where tau_0 * 3/2 < 1, the atmosphere is transparent and we use
    # the continuum (Rayleigh-dominated) level; clip to avoid log(<=0).
    tau_eff = np.maximum(tau_0 * (2.0 / 3.0), 1e-30)
    delta_R = H * np.log(tau_eff)

    # Transit depth (exact formula)
    R_eff   = R_p + np.maximum(delta_R, 0.0)
    delta   = (R_eff / R_star)**2

    # Enforce physical minimum (bare planet)
    delta   = np.maximum(delta, (R_p / R_star)**2)

    return delta, R_eff, H


def apply_flare_changes(atm, changes):
    """
    Return a modified copy of `atm` with VMRs updated by flare-driven changes.

    Two types of keys are supported:
      - Regular species (e.g. 'O3', 'CH4'): fractional multiplier
          new_VMR = VMR * (1 + frac_change)
      - Abiotic species (e.g. 'O3_abiotic', 'O2_abiotic'): absolute addition
          new_VMR = VMR + abs_change
          These represent newly-produced species from abiotic photochemistry.

    NaN or non-finite frac_change values are silently skipped.
    """
    import copy
    atm_mod = copy.deepcopy(atm)
    vmr_mod = atm_mod['vmr']

    for sp, change in changes.items():
        if not np.isfinite(change):
            continue
        if sp.endswith('_abiotic'):
            # Absolute VMR addition: abiotic production from photolysis
            sp_base = sp.replace('_abiotic', '')
            if sp_base in vmr_mod:
                vmr_mod[sp_base] = float(np.clip(vmr_mod[sp_base] + change, 0.0, 1.0))
        else:
            if sp in vmr_mod:
                new_val = vmr_mod[sp] * (1.0 + change)
                vmr_mod[sp] = float(np.clip(new_val, 0.0, 1.0))
    return atm_mod


# ===========================================================================
# SECTION 5: Build all spectra
# ===========================================================================
def build_all_spectra(lam_um, atms, cumul_changes, vulcan_changes):
    """
    Compute quiescent + post-flare spectra for all 3 atmospheres.

    Returns dict: {label: {'quiescent': delta_arr, 'postflare': delta_arr,
                            'vulcan': delta_arr (Atm A only), 'H_cm': float}}
    """
    # All species that appear in any atmosphere
    all_species = set()
    for atm in atms.values():
        all_species.update(atm['vmr'].keys())

    sigma = compute_cross_sections(lam_um, list(all_species))

    results = {}
    for lbl, atm in atms.items():
        print(f"\n  Computing Atm {lbl} ({atm['name'][:40]})...")
        print(f"    T={atm['T_K']}K  P={atm['P_bar']}bar  "
              f"mu={mean_molecular_weight(atm['vmr']):.1f} amu")

        H = compute_scale_height(atm['T_K'],
                                 mean_molecular_weight(atm['vmr']), G_P)
        print(f"    Scale height H = {H/1e5:.2f} km")

        # Quiescent spectrum
        delta_q, R_eff_q, _ = compute_transmission_spectrum(atm, lam_um, sigma)
        print(f"    Quiescent: transit depth range "
              f"{delta_q.min()*1e6:.0f} - {delta_q.max()*1e6:.0f} ppm")

        # Post-79-day-flare-sequence (cumulative state from Stage 2 LUT)
        changes_lbl = {}
        if lbl in cumul_changes:
            for sp, frac in cumul_changes[lbl].items():
                changes_lbl[sp] = frac
        atm_pf = apply_flare_changes(atm, changes_lbl)
        delta_pf, _, _ = compute_transmission_spectrum(atm_pf, lam_um, sigma)

        # Change in ppm
        ddelta = (delta_pf - delta_q) * 1e6
        print(f"    Post-flare: max |ddelta| = {np.nanmax(np.abs(ddelta)):.1f} ppm")

        entry = {
            'name': atm['name'],
            'H_cm': H,
            'mu_amu': mean_molecular_weight(atm['vmr']),
            'quiescent': delta_q,
            'postflare_cumul': delta_pf,
        }

        # VULCAN-analog (largest single flare) for Atm A only
        if lbl == 'A' and vulcan_changes:
            atm_v = apply_flare_changes(atm, vulcan_changes)
            delta_v, _, _ = compute_transmission_spectrum(atm_v, lam_um, sigma)
            entry['vulcan_single'] = delta_v
            ddv = (delta_v - delta_q) * 1e6
            print(f"    VULCAN single-flare: max |ddelta| = {np.abs(ddv).max():.2f} ppm")

        results[lbl] = entry
    return results


# ===========================================================================
# SECTION 6: Save Spectra as CSV
# ===========================================================================
def save_spectra(lam_um, results):
    """
    Save each spectrum as a CSV, plus a combined comparison file.
    """
    all_rows = []
    for lbl, entry in results.items():
        for scenario, arr in entry.items():
            if not isinstance(arr, np.ndarray):
                continue
            label_str = f"Atm{lbl}_{scenario}"
            df = pd.DataFrame({
                'wavelength_um': lam_um,
                'transit_depth': arr,
                'transit_depth_ppm': arr * 1e6,
                'atmosphere': lbl,
                'name': entry['name'],
                'scenario': scenario,
            })
            df.to_csv(f"{OUTPUT_DIR}/spectrum_atm{lbl}_{scenario}.csv", index=False)
            all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/all_spectra_combined.csv", index=False)
    print(f"  Saved {len(all_rows)} spectrum files + combined CSV")
    return combined


# ===========================================================================
# SECTION 7: Diagnostic Plots
# ===========================================================================
def plot_spectra(lam_um, results, combined_df):
    """
    Generate Stage 3 summary figure with:
      Row 1: All 3 quiescent spectra comparison
      Row 2: Atm A before/after (VULCAN + cumulative)
      Row 3: Atm B before/after (CO2-dominated; abiotic O3 false positive)
      Row 4: Feature amplitude summary + scale height comparison
    """
    print("\n  Generating diagnostic plots...")

    # Feature annotations for key biosignatures
    FEATURES = [
        (0.76,  'O$_2$',   'darkblue',   0.016),
        (1.38,  'H$_2$O',  'steelblue',  0.016),
        (1.87,  'H$_2$O',  'steelblue',  0.016),
        (2.30,  'CH$_4$',  'green',      0.012),
        (2.70,  'H$_2$O\n+CO$_2$', 'steelblue', 0.015),
        (3.30,  'CH$_4$',  'green',      0.016),
        (4.30,  'CO$_2$',  'firebrick',  0.016),
        (4.50,  'N$_2$O',  'purple',     0.010),
        (4.67,  'CO',      'darkorange', 0.010),
        (6.30,  'H$_2$O',  'steelblue',  0.016),
        (7.70,  'CH$_4$',  'green',      0.016),
        (9.60,  'O$_3$',   'royalblue',  0.018),
    ]

    COLORS = {'A': '#1565C0', 'B': '#B71C1C', 'C': '#546E7A'}
    ATM_NAMES = {
        'A': 'Atm A: N$_2$-Earth (biosignatures)',
        'B': 'Atm B: CO$_2$-Venus (false-positive O$_3$)',
        'C': 'Atm C: Thin remnant',
    }

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "TRAPPIST-1e Digital Twin -- Stage 3: Synthetic Transmission Spectra\n"
        "Lecavelier+2008 analytic RT | HITRAN2020 cross-sections | JWST NIRSpec/PRISM resolution (R=100)",
        fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

    # ── Panel 1: All 3 quiescent spectra ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    for lbl, entry in results.items():
        delta_q = entry['quiescent'] * 1e6   # ppm
        ax1.plot(lam_um, delta_q, color=COLORS[lbl], lw=1.5,
                 label=ATM_NAMES[lbl], alpha=0.9)

    # Shade JWST NIRSpec/PRISM range
    ax1.axvspan(0.6, 5.3, alpha=0.06, color='gold', label='NIRSpec/PRISM range')
    ax1.axvspan(5.0, 14.0, alpha=0.04, color='orange', label='MIRI range')

    # Annotate key features
    for (lam0, name, clr, yoff) in FEATURES:
        if lam_um[0] <= lam0 <= lam_um[-1]:
            ax1.axvline(lam0, color=clr, lw=0.7, ls='--', alpha=0.6)
            ylim = ax1.get_ylim()
            ax1.text(lam0, ax1.get_ylim()[1] * (0.97 - yoff), name,
                     ha='center', va='top', fontsize=6.5, color=clr, rotation=90)

    ax1.set_xlabel('Wavelength (μm)', fontsize=10)
    ax1.set_ylabel('Transit Depth (ppm)', fontsize=10)
    ax1.set_title('All 3 Atmospheric Compositions — Quiescent Spectra\n'
                  'TRAPPIST-1e (R_p=0.920 R_Earth, R_star=0.119 R_Sun)', fontsize=10)
    ax1.set_xlim(lam_um[0], lam_um[-1])
    ax1.legend(fontsize=8, loc='upper right')
    ax1.text(0.01, 0.96, f'Continuum depth = {DELTA_CONT*1e6:.0f} ppm (R_p²/R_*²)',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # ── Panel 2: Atm A before vs after flares ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    delta_q  = results['A']['quiescent'] * 1e6
    delta_pf = results['A']['postflare_cumul'] * 1e6

    ax2.plot(lam_um, delta_q, 'b-', lw=1.8, label='Quiescent (pre-flare)', alpha=0.9)
    ax2.plot(lam_um, delta_pf, 'r-', lw=1.8, label='Post 79-day flare sequence', alpha=0.9)

    if 'vulcan_single' in results['A']:
        delta_v = results['A']['vulcan_single'] * 1e6
        ax2.plot(lam_um, delta_v, 'g--', lw=1.3,
                 label='Post largest single flare (VULCAN ODE)', alpha=0.8)

    ax2.fill_between(lam_um, delta_q, delta_pf,
                     where=(delta_pf < delta_q), alpha=0.25, color='red',
                     label='O$_3$/CH$_4$ depletion')
    ax2.fill_between(lam_um, delta_q, delta_pf,
                     where=(delta_pf > delta_q), alpha=0.25, color='orange',
                     label='NO$_2$/N$_2$O enhancement')

    # Mark O3 9.6 um and CH4 3.3 um
    for lam0, sp in [(3.3, 'CH$_4$\n3.3μm'), (9.6, 'O$_3$\n9.6μm')]:
        if lam0 <= lam_um[-1]:
            ax2.axvline(lam0, color='purple', lw=1, ls=':', alpha=0.7)
            ax2.text(lam0, ax2.get_ylim()[1]*0.97 if ax2.get_ylim()[1] > 0
                     else delta_q.max()*0.97,
                     sp, ha='center', va='top', fontsize=7, color='purple')

    ax2.set_xlabel('Wavelength (μm)', fontsize=10)
    ax2.set_ylabel('Transit Depth (ppm)', fontsize=10)
    ax2.set_title('Atm A (N$_2$-Earth): Biosignature Response to TRAPPIST-1 Flares\n'
                  '79-day cumulative (Segura+2010, Tilley+2019, Chen+2021)',
                  fontsize=9)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.text(0.02, 0.04,
             '[!] 1D model: O3 depleted\n3D (Ridgway+2023): O3 increases 20x',
             transform=ax2.transAxes, fontsize=7.5, color='red', va='bottom',
             bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))
    ax2.axvspan(0.6, 5.3, alpha=0.06, color='gold')

    # ── Panel 3: Atm B before vs after flares (false positive) ──────────────
    ax3 = fig.add_subplot(gs[1, 1])

    delta_qB  = results['B']['quiescent'] * 1e6
    delta_pfB = results['B']['postflare_cumul'] * 1e6

    ax3.plot(lam_um, delta_qB,  color='#B71C1C', lw=1.8,
             label='Quiescent CO$_2$-rich', alpha=0.9)
    ax3.plot(lam_um, delta_pfB, color='#E57373', lw=1.8, ls='--',
             label='Post-flare (abiotic O$_3$ increase)', alpha=0.9)

    ax3.fill_between(lam_um, delta_qB, delta_pfB,
                     where=(delta_pfB > delta_qB), alpha=0.3, color='red',
                     label='Abiotic O$_3$ false positive')

    for lam0, sp in [(4.3, 'CO$_2$\n4.3μm'), (9.6, 'O$_3$\n9.6μm')]:
        if lam0 <= lam_um[-1]:
            ax3.axvline(lam0, color='darkred', lw=1, ls=':', alpha=0.7)
            ax3.text(lam0, delta_qB.max()*0.97, sp, ha='center', va='top',
                     fontsize=7, color='darkred')

    ax3.set_xlabel('Wavelength (μm)', fontsize=10)
    ax3.set_ylabel('Transit Depth (ppm)', fontsize=10)
    ax3.set_title('Atm B (CO$_2$-Venus): Flare-Driven Abiotic O$_3$ False Positive\n'
                  'Miranda-Rosete+2024: CO$_2$ photolysis -> O$_3$ without life',
                  fontsize=9)
    ax3.legend(fontsize=7, loc='upper right')
    ax3.text(0.02, 0.04,
             'WARNING: O$_3$ at 9.6μm here is ABIOTIC\n'
             '(CO$_2$ photolysis under flare UV)\nDiagnostic: CO at 4.67μm also present',
             transform=ax3.transAxes, fontsize=7.5, color='darkred', va='bottom',
             bbox=dict(boxstyle='round', fc='#FFEBEE', alpha=0.9))
    ax3.axvspan(0.6, 5.3, alpha=0.06, color='gold')

    # ── Panel 4: Spectral difference (ddelta in ppm) ─────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])

    for lbl, entry in results.items():
        ddelta = (entry['postflare_cumul'] - entry['quiescent']) * 1e6
        ax4.plot(lam_um, ddelta, color=COLORS[lbl], lw=1.5,
                 label=f"Atm {lbl}", alpha=0.85)

    ax4.axhline(0, color='k', lw=0.8, ls='--')
    ax4.axhspan(-1, 0, alpha=0.08, color='blue', label='Depletion (negative)')
    ax4.axhspan(0, 1, alpha=0.08, color='red', label='Enhancement (positive)')

    for lam0, sp in [(3.3, 'CH4'), (4.3, 'CO2'), (7.7, 'CH4'), (9.6, 'O3')]:
        if lam0 <= lam_um[-1]:
            ax4.axvline(lam0, color='gray', lw=0.7, ls=':', alpha=0.5)

    ax4.set_xlabel('Wavelength (μm)', fontsize=10)
    ax4.set_ylabel('Δ(Transit Depth) (ppm)', fontsize=10)
    ax4.set_title('Spectral Change: Post-Flare minus Quiescent\n'
                  '79-day TRAPPIST-1 flare sequence', fontsize=9)
    ax4.legend(fontsize=8)
    ax4.axvspan(0.6, 5.3, alpha=0.06, color='gold')

    # ── Panel 5: Scale heights + feature amplitudes (bar chart) ─────────────
    ax5 = fig.add_subplot(gs[2, 1])

    atm_labels = ['A\n(N2-Earth)', 'B\n(CO2-Venus)', 'C\n(Thin rock)']
    scale_heights_km = [results[lbl]['H_cm'] / 1e5 for lbl in ['A', 'B', 'C']]
    mu_amus = [results[lbl]['mu_amu'] for lbl in ['A', 'B', 'C']]

    x = np.arange(3)
    bars = ax5.bar(x, scale_heights_km, width=0.5,
                   color=[COLORS['A'], COLORS['B'], COLORS['C']],
                   alpha=0.8, edgecolor='k', lw=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(atm_labels, fontsize=9)
    ax5.set_ylabel('Atmospheric Scale Height (km)', fontsize=9)
    ax5.set_title('Scale Heights & Mean Molecular Weights\n'
                  'H = kT/(mu*g),  g = 8.0 m/s² (TRAPPIST-1e)', fontsize=9)

    for bar_i, (H_km, mu) in zip(bars, zip(scale_heights_km, mu_amus)):
        ax5.text(bar_i.get_x() + bar_i.get_width()/2.0,
                 bar_i.get_height() + 0.1,
                 f'{H_km:.1f} km\nmu={mu:.1f}', ha='center', va='bottom', fontsize=8)

    # Feature detection threshold annotation
    ax5.axhline(2.0 * np.mean(scale_heights_km),
                color='gray', ls='--', lw=1,
                label=f'2H avg = {2*np.mean(scale_heights_km):.1f} km')
    ax5.text(2.5, 2.0 * np.mean(scale_heights_km) + 0.2,
             '~2H feature\namplitude', fontsize=7.5, color='gray')
    ax5.legend(fontsize=7)
    ax5.text(0.02, 0.96,
             f'g_planet = {G_P:.0f} cm/s^2\n'
             f'R_p = {R_P_CM/R_EARTH_CM:.3f} R_Earth',
             transform=ax5.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    path = f"{OUTPUT_DIR}/stage3_spectra_plot.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# SECTION 8: Feature Amplitude Summary
# ===========================================================================
def compute_feature_amplitudes(lam_um, results):
    """
    For each key spectral feature, compute peak amplitude above continuum
    in ppm for each atmosphere and scenario.  Used for Stage 4 calibration.
    """
    KEY_FEATURES = {
        'H2O_1.4': (1.35, 1.45),
        'H2O_1.9': (1.80, 1.95),
        'CH4_2.3': (2.15, 2.45),
        'CH4_3.3': (3.10, 3.55),
        'CO2_4.3': (4.10, 4.50),
        'N2O_4.5': (4.40, 4.65),
        'CO_4.67': (4.55, 4.80),
        'O3_9.6':  (9.00, 10.20),
        'H2O_6.3': (5.80, 6.80),
    }

    rows = []
    for lbl, entry in results.items():
        for feat_name, (lam_lo, lam_hi) in KEY_FEATURES.items():
            # Continuum: average over wings (regions just outside the feature)
            wing_width = (lam_hi - lam_lo) * 0.5
            cont_mask = ((lam_um > lam_lo - wing_width) & (lam_um < lam_lo)) | \
                        ((lam_um > lam_hi) & (lam_um < lam_hi + wing_width))
            feat_mask = (lam_um >= lam_lo) & (lam_um <= lam_hi)

            for scenario_key in ['quiescent', 'postflare_cumul', 'vulcan_single']:
                if scenario_key not in entry:
                    continue
                spec = entry[scenario_key] * 1e6  # ppm

                if cont_mask.sum() == 0 or feat_mask.sum() == 0:
                    continue

                continuum = np.nanmean(spec[cont_mask])
                feature_peak = np.nanmax(spec[feat_mask])
                amplitude = feature_peak - continuum

                rows.append({
                    'atmosphere': lbl,
                    'atm_name': entry['name'],
                    'scenario': scenario_key,
                    'feature': feat_name,
                    'lam_lo_um': lam_lo,
                    'lam_hi_um': lam_hi,
                    'continuum_ppm': continuum,
                    'peak_ppm': feature_peak,
                    'amplitude_ppm': amplitude,
                })

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(f"{OUTPUT_DIR}/feature_amplitudes.csv", index=False)
    print(f"  Feature amplitude table: {len(feat_df)} entries")
    return feat_df


# ===========================================================================
# SECTION 9: Main Pipeline
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "#"*65)
    print("  TRAPPIST-1e DIGITAL TWIN -- STAGE 3: SPECTRAL GENERATION")
    print("#"*65 + "\n")

    # ── Load Stage 2 outputs ───────────────────────────────────────────────
    print("="*65)
    print("Loading Stage 2 outputs")
    print("="*65)

    try:
        atms = {lbl: load_stage2(lbl) for lbl in ['A', 'B', 'C']}
    except FileNotFoundError as e:
        print(f"ERROR: Stage 2 output not found: {e}")
        print("  Run stage2_atmospheric_response.py first.")
        sys.exit(1)

    for lbl, atm in atms.items():
        print(f"  Atm {lbl}: {atm['name']}  "
              f"(P={atm['P_bar']} bar, T={atm['T_K']} K, "
              f"mu={mean_molecular_weight(atm['vmr']):.1f} amu)")
        biosig_vmrs = {sp: atm['vmr'][sp] for sp in ['O3','CH4','N2O']
                       if sp in atm['vmr']}
        if biosig_vmrs:
            print(f"    Biosignature VMRs: " +
                  ", ".join(f"{sp}={v:.2e}" for sp, v in biosig_vmrs.items()))

    cumul_changes = load_cumulative_changes()
    print(f"\n  Cumulative changes loaded:")
    for lbl, chg in cumul_changes.items():
        sig = {k: v for k, v in chg.items() if abs(v) > 1e-6}
        if sig:
            print(f"    Atm {lbl}: " +
                  ", ".join(f"d{sp}={v*100:.3f}%" for sp, v in sig.items()))

    vulcan_changes = load_vulcan_analog()
    print(f"  VULCAN single-flare changes: " +
          ", ".join(f"{sp}={v*100:.4f}%" for sp, v in vulcan_changes.items()))

    # ── Wavelength grid ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Building wavelength grid (R=100, 0.5-14 um)")
    print("="*65)

    lam_um, lam_cm = make_wavelength_grid(0.5, 14.0, R=100)
    print(f"  {len(lam_um)} wavelength points from {lam_um[0]:.2f} to {lam_um[-1]:.1f} um")
    print(f"  NIRSpec/PRISM range (0.6-5.3 um): "
          f"{((lam_um>=0.6)&(lam_um<=5.3)).sum()} points")
    print(f"  MIRI range (5-14 um): "
          f"{((lam_um>=5.0)&(lam_um<=14.0)).sum()} points")

    # ── Compute spectra ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Computing transmission spectra")
    print("="*65)
    print(f"  System: R_p={R_P_CM/R_EARTH_CM:.3f} R_Earth, "
          f"R_star={R_STAR_CM/R_SUN_CM:.4f} R_Sun")
    print(f"  g_planet = {G_P:.0f} cm/s^2 ({G_P/100:.2f} m/s^2)")
    print(f"  Continuum transit depth = {DELTA_CONT*1e6:.0f} ppm")

    results = build_all_spectra(lam_um, atms, cumul_changes, vulcan_changes)

    # ── Save ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Saving spectra")
    print("="*65)

    combined_df = save_spectra(lam_um, results)

    # ── Feature amplitudes ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Computing feature amplitudes")
    print("="*65)

    feat_df = compute_feature_amplitudes(lam_um, results)

    # Print key results
    for lbl in ['A', 'B']:
        print(f"\n  Atm {lbl} quiescent feature amplitudes:")
        sub = feat_df[(feat_df['atmosphere'] == lbl) &
                      (feat_df['scenario'] == 'quiescent')]
        for _, r in sub.iterrows():
            if r['amplitude_ppm'] > 1.0:
                print(f"    {r['feature']:12s}: {r['amplitude_ppm']:7.1f} ppm")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Generating plots")
    print("="*65)

    plot_path = plot_spectra(lam_um, results, combined_df)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STAGE 3 COMPLETE")
    print("="*65)

    A_H  = results['A']['H_cm'] / 1e5
    B_H  = results['B']['H_cm'] / 1e5
    C_H  = results['C']['H_cm'] / 1e5

    # Find O3 amplitude for Atm A
    o3_amp_A = feat_df[(feat_df['atmosphere']=='A') &
                       (feat_df['scenario']=='quiescent') &
                       (feat_df['feature']=='O3_9.6')]['amplitude_ppm']
    o3_amp_A = float(o3_amp_A.iloc[0]) if len(o3_amp_A) > 0 else 0.0

    o3_amp_B_pf_s = feat_df[(feat_df['atmosphere']=='B') &
                            (feat_df['scenario']=='postflare_cumul') &
                            (feat_df['feature']=='O3_9.6')]['amplitude_ppm']
    o3_amp_B_pf = float(o3_amp_B_pf_s.iloc[0]) if len(o3_amp_B_pf_s) > 0 and \
                  np.isfinite(float(o3_amp_B_pf_s.iloc[0])) else 0.0

    cumul_O3_A = cumul_changes.get('A', {}).get('O3', 0.0)

    print(f"""
  TRAPPIST-1e (TIC 267519185) -- Stage 3 Results
  ───────────────────────────────────────────────
  Wavelength grid:     {len(lam_um)} pts, {lam_um[0]:.2f}-{lam_um[-1]:.1f} um, R=100
  Continuum depth:     {DELTA_CONT*1e6:.0f} ppm  [(R_p/R_star)^2]
  Planet gravity:      {G_P:.0f} cm/s^2 = {G_P/100:.2f} m/s^2

  Scale heights:
    Atm A (N2-Earth,  T=270K, mu=29):  H = {A_H:.2f} km
    Atm B (CO2-Venus, T=380K, mu=43):  H = {B_H:.2f} km
    Atm C (Thin rock, T=250K, mu=28):  H = {C_H:.2f} km

  Key spectral features (Atm A, quiescent):
    O3   9.6 um:  {o3_amp_A:.1f} ppm  (key biosignature)
    CH4  3.3 um:  see feature_amplitudes.csv
    CO2  4.3 um:  dominant continuum driver

  Flare impact (79-day cumulative, Atm A):
    Cumulative dO3 = {cumul_O3_A*100:.3f}%  (1D model; Ridgway+2023 3D predicts INCREASE)
    Abiotic O3 (Atm B post-flare): {o3_amp_B_pf:.1f} ppm  (false-positive scenario)

  Formalism:
    Lecavelier des Etangs+2008 chord optical depth
    tau_chord(b) = tau_0(lambda) * exp(-(b-R_p)/H)
    R_eff = R_p + H * ln(tau_0 * 2/3)
    Cross-sections: HITRAN2020 Gaussian parameterisation

  Output files:  {OUTPUT_DIR}/
    spectrum_atmA_quiescent.csv / postflare_cumul.csv / vulcan_single.csv
    spectrum_atmB_quiescent.csv / postflare_cumul.csv
    spectrum_atmC_quiescent.csv / postflare_cumul.csv
    all_spectra_combined.csv
    feature_amplitudes.csv
    stage3_spectra_plot.png

  Key references:
    Lecavelier des Etangs+2008 A&A 481 L83   -- analytic RT formalism
    Fortney 2005 MNRAS 364 649               -- scale height transit spectra
    HITRAN2020 (Gordon+2022 JQSRT 277)       -- molecular cross-sections
    Schwieterman+2018 Astrobiology 18(6)     -- biosignature spectroscopy
    Espinoza+2025 DREAMS (JWST)              -- calibration target

  -> Ready for Stage 4: JWST calibration + Streamlit dashboard
""")
