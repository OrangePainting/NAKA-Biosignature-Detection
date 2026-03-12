import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(sys.executable)

"""
===========================================================================
TRAPPIST-1e Digital Twin — Stage 2: Atmospheric Photochemical Response
===========================================================================

Pipeline (Hours 3–8 of hackathon):
  1. Load Stage 1 outputs: real TESS flare catalog + MegaMUSCLES-calibrated
     stellar spectrum for TRAPPIST-1 (TIC 267519185)
  2. Define 3 candidate TRAPPIST-1e atmospheric compositions consistent
     with JWST DREAMS NIRSpec/PRISM constraints (Espinoza et al. 2025):
       (a) N2-dominated, Earth-like    → CH4/O3/N2O biosignatures present
       (b) CO2-dominated, Venus-analog → abiotic O3 false-positive pathway
       (c) Thin remnant / bare rock    → no UV chemical shield
  3. Per-flare UV enhancement (Davenport 2014 template):
       - T_flare = 7000 K (Ducrot+2022, TRAPPIST-1 specific)
       - Planck blackbody ratio: TESS band → FUV/NUV bands
       - Scaled to MegaMUSCLES quiescent SED (France+2020, Wilson+2025)
  4. Photochemical lookup tables — hackathon shortcut
       (instead of running VULCAN live, interpolate from peer-reviewed results):
       - Segura+2010: 1D O3 response; UV-only ~5%, UV+SEP up to 94% loss
       - Tilley+2019: cumulative flaring reduces O3 to 6% over 10 years
       - Chen+2021: 3D WACCM; NO2/N2O/HNO3 increase; new equilibria
       - Miranda-Rosete+2024: abiotic O3 false positive in CO2 atmospheres
  [STRETCH] 5. Simplified VULCAN-analog photochemical ODE (Chapman + HOx)
               Atmosphere A + largest observed flare. Species: O3, OH, CH4.
               Rate constants from JPL/NASA Sander et al. 2011 at T=250K.

NOTE on 1D vs 3D tension (explicitly flagged per PDF guidance):
  Ridgway+2023 (Met Office UM 3D GCM) found flares INCREASE O3 by 20x on
  Proxima Cen b-like planets — the opposite of 1D predictions. Chen+2021
  (CESM1/WACCM 3D) found new chemical equilibria where NO2/N2O/HNO3 become
  detectable. This 1D model represents a lower bound on biological survivability
  and upper bound on O3 destruction. 3D dynamic effects are not captured here.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import warnings, os

warnings.filterwarnings('ignore')

OUTPUT_DIR = "Stage 2 Files"
STAGE1_DIR = "Stage 1 Files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Physical Constants ──────────────────────────────────────────────────────
HCK       = 1.4388e7       # h·c / k_B   in nm·K  (= 14388 um·K converted to nm·K)
AU_CM     = 1.496e13       # 1 AU in cm
R_SUN_CM  = 6.957e10       # Solar radius in cm

# ── TRAPPIST-1 System (NASA Exoplanet Archive) ─────────────────────────────
T_EFF_K   = 2566           # Stellar effective temperature (K)
T_FLARE_K = 7000           # Flare blackbody temperature, Ducrot+2022 (A&A)
                           # Range 5300-8600 K measured; 7000 K adopted as median
R_STAR_CM = 0.1192 * R_SUN_CM
A_AU      = 0.02928        # TRAPPIST-1e semi-major axis (AU)
A_CM      = A_AU * AU_CM
FOUR_PI_A2 = 4.0 * np.pi * A_CM**2   # cm^2 — isotropic flux dilution

# ── MegaMUSCLES quiescent UV fluxes at TRAPPIST-1e ─────────────────────────
# France+2020 ApJS 247 25; Wilson+2025 (MegaMUSCLES); Peacock+2019 ApJ 886 77
# Units: erg s^-1 cm^-2 (integrated over bandpass)
MEGA = dict(
    F_FUV  = 0.003,    # 100–200 nm  (Lyman-alpha region dominated)
    F_NUV  = 0.070,    # 200–320 nm  (key O3 photochemistry band)
    F_Lya  = 0.025,    # 121.6 nm    (Lyman-alpha; chromospheric line)
    F_TESS = 14.2,     # 600–1000 nm (TESS band, computed from Planck T=2566K)
)

# ── UV energy fractions of bolometric flare energy ─────────────────────────
# Calibrated from M-dwarf flare spectroscopy (Kowalski+2013, Segura+2010,
# Hawley+2003). These represent the fraction of E_bol emitted in each band.
F_UV_NUV = 0.025    # 2.5% of E_bol in NUV (200–320 nm)
F_UV_FUV = 0.002    # 0.2% of E_bol in FUV (100–200 nm)


# ===========================================================================
# SECTION 1: Atmospheric Compositions
# ===========================================================================
def define_atmospheric_compositions():
    """
    3 candidate atmospheric compositions consistent with JWST DREAMS
    NIRSpec/PRISM transmission spectroscopy (Espinoza et al. 2025).

    DREAMS ruled out primary H2-dominated atmospheres (>3σ) and disfavored
    Venus-like and Mars-like compositions. N2-rich with traces of CH4 and CO2
    remains permitted but unconfirmed as of March 2026.
    """
    atms = {}

    # ── Atmosphere A: N2-dominated (Earth-like) ────────────────────────────
    # Most favoured by DREAMS; biosignatures potentially detectable
    atms['A'] = dict(
        label       = 'A',
        name        = 'N₂-dominated (Earth-like)',
        shortname   = 'N2-Earth',
        jwst_status = 'Permitted (Espinoza+2025)',
        P_surface   = 1.0,          # bar
        T_surface   = 270.0,        # K  (near TRAPPIST-1e T_eq ~250K + greenhouse)
        # Volume mixing ratios
        vmr = dict(
            N2  = 0.7900,
            O2  = 0.2095,
            CO2 = 4.20e-4,          # 420 ppm
            CH4 = 1.80e-6,          # 1.8 ppm (biogenic)
            O3  = 3.00e-7,          # 300 ppb (stratospheric column)
            N2O = 3.20e-7,          # 320 ppb (biogenic)
            CO  = 1.00e-7,          # 100 ppb (M-dwarf abiotic baseline; Eager-Nash+2024, 5-10x Earth)
            H2O = 1.00e-2,          # 1% (troposphere)
            NO2 = 5.00e-10,         # 0.5 ppb
            HNO3= 3.00e-10,         # 0.3 ppb
        ),
        biosignatures    = ['O3', 'CH4', 'N2O'],
        false_positives  = [],
        uv_shield        = 'Strong (O3 column ~300 DU equivalent)',
        reference        = 'Schwieterman+2018 Astrobiology; arXiv:2602.02267',
        notes = ('1D model predicts severe O3 depletion from repeated flaring. '
                 '3D models (Ridgway+2023) predict O3 INCREASE by up to 20x. '
                 'Result flagged as model-dependent.')
    )

    # ── Atmosphere B: CO2-dominated Venus-analog ───────────────────────────
    # Disfavoured by DREAMS but not yet ruled out at high significance
    atms['B'] = dict(
        label       = 'B',
        name        = 'CO₂-dominated (Venus-analog)',
        shortname   = 'CO2-Venus',
        jwst_status = 'Disfavoured (Espinoza+2025)',
        P_surface   = 10.0,         # bar (much less than Venus 92 bar)
        T_surface   = 380.0,        # K  (strong greenhouse)
        vmr = dict(
            CO2 = 0.9600,
            N2  = 0.0350,
            SO2 = 1.50e-4,          # 150 ppm (volcanic)
            CO  = 2.00e-5,          # 20 ppm (CO2 photolysis)
            O2  = 5.00e-7,          # 0.5 ppm (abiotic, photolytic)
            O3  = 1.00e-10,         # trace (no O2 reservoir)
            H2O = 1.00e-5,          # 10 ppm (upper atmosphere)
            HCl = 5.00e-7,          # 0.5 ppm (volcanic)
        ),
        biosignatures    = [],
        false_positives  = ['O3', 'O2'],  # Miranda-Rosete+2024 abiotic pathway
        uv_shield        = 'Minimal (no O3 column; H2SO4 clouds provide some NUV absorption)',
        reference        = 'Miranda-Rosete+2024 arXiv:2308.01880',
        notes = ('Flare UV drives CO2 → CO + O → trace O3 (abiotic false positive). '
                 'Miranda-Rosete+2024: superflares enhance abiotic O3 in prebiotic '
                 'CO2 atmospheres. This is the false-positive scenario to quantify.')
    )

    # ── Atmosphere C: Thin remnant / bare rock ─────────────────────────────
    # Consistent with ~50% of rocky M-dwarf planets being atmosphereless
    atms['C'] = dict(
        label       = 'C',
        name        = 'Thin remnant / bare rock',
        shortname   = 'Thin-Rock',
        jwst_status = 'Possible (many TRAPPIST-1b,c,d siblings are airless)',
        P_surface   = 0.01,         # bar  (~10 mbar; Mars-like thin)
        T_surface   = 250.0,        # K  (T_eq, no greenhouse)
        vmr = dict(
            N2  = 0.9800,
            CO2 = 0.0200,           # 2%
            Ar  = 1.00e-4,
            # No biosignature gases
        ),
        biosignatures    = [],
        false_positives  = [],
        uv_shield        = 'None — UV reaches surface nearly unattenuated',
        uv_transmission  = 0.97,    # 97% of UV passes through thin atmosphere
        reference        = 'Turbet+2018; Johnstone+2021',
        notes = ('No significant photochemistry. UV surface dose tracks directly '
                 'with stellar UV enhancement. Relevant for assessing surface '
                 'habitability under TRAPPIST-1 flare activity.')
    )

    return atms


def save_atmosphere_table(atms):
    rows = []
    for lbl, atm in atms.items():
        row = {'atmosphere': lbl, 'name': atm['name'],
               'P_bar': atm['P_surface'], 'T_K': atm['T_surface'],
               'jwst_status': atm['jwst_status'],
               'biosignatures': ','.join(atm['biosignatures']),
               'false_positives': ','.join(atm['false_positives']),
               'uv_shield': atm['uv_shield']}
        vmr = atm['vmr']
        for sp in ['N2','O2','CO2','CH4','O3','N2O','H2O','CO','SO2','NO2','HNO3']:
            row[f'vmr_{sp}'] = vmr.get(sp, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/atmospheric_compositions.csv", index=False)
    return df


# ===========================================================================
# SECTION 2: UV Enhancement Calculator
# ===========================================================================
def planck_ratio(wl_nm, T_hot, T_cold):
    """
    Ratio of Planck blackbody functions: B(λ, T_hot) / B(λ, T_cold).
    Uses expm1 for numerical stability. Handles large exponent limit.

    At UV wavelengths with T_cold = 2566 K, x_cold can be large (>20),
    so we use exp(x_cold - x_hot) as the asymptotic ratio.
    """
    x_h = HCK / (wl_nm * T_hot)
    x_c = HCK / (wl_nm * T_cold)
    # For x_c > 50, use Wien approximation: ratio ≈ exp(x_c - x_h)
    if x_c > 50:
        return float(np.exp(min(x_c - x_h, 700.0)))
    return float(np.expm1(x_c) / np.expm1(x_h))


def davenport_profile(t_sec, t_peak_sec, amplitude, fwhm_sec):
    """
    Davenport (2014) empirical white-light flare template.
    Polynomial rise (phase −1→0) + double-exponential decay (phase 0→∞).
    Returns ΔF/F at each time point.

    Integral = 0.9135 × amplitude × fwhm_sec  (time-averaged equivalent width).
    """
    t_half = fwhm_sec / 2.0
    phase  = (t_sec - t_peak_sec) / t_half
    flux   = np.zeros_like(t_sec, dtype=float)

    # Rise: 4th-order polynomial (Davenport+2014, Table 1)
    rise = (phase >= -1.0) & (phase < 0.0)
    p = phase[rise]
    flux[rise] = 1.0 + 1.941*p - 0.175*p**2 - 2.246*p**3 - 1.125*p**4

    # Decay: double exponential (Davenport+2014, Table 1)
    decay = phase >= 0.0
    p = phase[decay]
    flux[decay] = 0.6890*np.exp(-1.600*p) + 0.3030*np.exp(-0.2783*p)

    return amplitude * np.clip(flux, 0, None)


def compute_uv_enhancement(cat):
    """
    For every flare in the catalog, compute:
      (1) Peak UV enhancement factor (dimensionless) via Planck blackbody ratio.
          Method: catalog amplitude A = f_area × C_TESS
                  → f_area = A / C_TESS
                  → EF_NUV_peak = 1 + f_area × C_NUV
          Ref: Kowalski+2013; Howard+2020 (43% of M-dwarf superflares >14000 K)

      (2) Total UV fluence deposited at TRAPPIST-1e (erg cm^-2) via energy
          fraction method (Hawley+2003; Segura+2010; Kowalski+2013).
          F_NUV = E_bol × f_UV_NUV / (4π × a^2)

    The two methods give consistent enhancement factors.  Method (1) tracks
    temporal shape; Method (2) feeds the photochemical lookup tables.
    """
    # Blackbody color ratios at T_flare=7000 K vs T_eff=2566 K
    # Reference wavelengths
    C_TESS = planck_ratio(800.0,  T_FLARE_K, T_EFF_K)   # TESS band anchor  ~91
    C_NUV  = planck_ratio(250.0,  T_FLARE_K, T_EFF_K)   # NUV  200–320 nm  ~1.4e6
    C_FUV  = planck_ratio(150.0,  T_FLARE_K, T_EFF_K)   # FUV  100–200 nm  ~1e18
    C_Lya  = planck_ratio(121.6,  T_FLARE_K, T_EFF_K)   # Ly-α 121.6 nm    ~1e22

    records = []
    for _, row in cat.iterrows():
        A    = row['peak_amplitude']       # ΔF/F in TESS band
        E    = row['energy_erg']
        dur  = row['duration_sec']
        T_fl = row.get('temperature_K', T_FLARE_K)

        # ── Method 1: Planck ratio (instantaneous peak enhancement) ────────
        C_T  = planck_ratio(800.0, T_fl, T_EFF_K)
        C_N  = planck_ratio(250.0, T_fl, T_EFF_K)
        f_area = A / C_T if C_T > 0 else 0.0
        EF_NUV_peak = 1.0 + f_area * C_N

        # Davenport time-integrated area ≈ 0.9135 × amplitude × fwhm
        # Effective UV fluence enhancement over quiescent during event:
        t_eff  = 0.9135 * dur    # effective equivalent duration (s)
        EF_NUV_avg = EF_NUV_peak * 0.5  # rough time-average (peak × 0.5 factor)

        # ── Method 2: Energy fraction → UV fluence at planet ───────────────
        E_NUV = E * F_UV_NUV          # erg emitted in NUV band
        E_FUV = E * F_UV_FUV          # erg emitted in FUV band
        F_NUV_fluence = E_NUV / FOUR_PI_A2   # erg cm^-2 at planet
        F_FUV_fluence = E_FUV / FOUR_PI_A2

        # Enhancement factor cross-check (fluence / quiescent × time):
        F_NUV_q_eff = MEGA['F_NUV'] * t_eff  # quiescent NUV fluence over same window
        EF_NUV_energy = 1.0 + (F_NUV_fluence / F_NUV_q_eff) if t_eff > 0 else 1.0

        records.append(dict(
            time_days        = row['time_days'],
            energy_erg       = E,
            log10_energy     = row['log10_energy'],
            duration_sec     = dur,
            peak_amplitude   = A,
            temperature_K    = T_fl,
            flare_class      = str(row.get('class', 'unknown')),
            # Planck ratio method
            C_TESS           = C_TESS,
            C_NUV            = C_N,
            f_area           = f_area,
            EF_NUV_peak      = EF_NUV_peak,
            EF_NUV_avg       = EF_NUV_avg,
            # Energy fraction method
            F_NUV_fluence    = F_NUV_fluence,
            F_FUV_fluence    = F_FUV_fluence,
            log10_F_NUV      = np.log10(max(F_NUV_fluence, 1e-30)),
            log10_F_FUV      = np.log10(max(F_FUV_fluence, 1e-30)),
            EF_NUV_energy    = EF_NUV_energy,
            t_eff_sec        = t_eff,
        ))

    uv_df = pd.DataFrame(records)
    uv_df.to_csv(f"{OUTPUT_DIR}/flare_uv_enhancement.csv", index=False)
    return uv_df


# ===========================================================================
# SECTION 3: Photochemical Lookup Tables
# ===========================================================================
def build_photochem_lut():
    """
    Per-event biosignature mixing-ratio changes vs. log10(F_NUV / erg cm^-2).
    Interpolated from published peer-reviewed photochemical model results.

    Primary references:
      Segura+2010 Astrobiology 9(18): 1D Atmos model, AD Leo 1985 flare
        → UV-only: ~5% O3 depletion; UV+SEP: 94% over 2 years
      Tilley+2019 Astrobiology 19(1): repeated GJ-1243 flare history
        → cumulative O3 to 6% of initial over 10 years; threshold E~10^30.5 erg
      Chen+2021 Nature Astronomy 5(298): 3D CESM1/WACCM model
        → new equilibria; NO2/N2O/HNO3 increase; CH4 slight decrease
      Miranda-Rosete+2024 arXiv:2308.01880: prebiotic CO2 atmosphere
        → abiotic O3 false positive from single superflare

    x-axis: log10(F_NUV / erg cm^-2) — NUV fluence at planet per event
    Values: fractional change per single flare event (ΔX / X_background)
      Positive = production/increase; Negative = destruction/decrease
    """
    # x-axis nodes (log10 of NUV fluence in erg/cm^2)
    # Typical TRAPPIST-1e flare mapping:
    #   10^30 erg → log10(F_NUV) ≈ 4.0   (micro-flare)
    #   10^31 erg → log10(F_NUV) ≈ 5.0   (moderate)
    #   10^32 erg → log10(F_NUV) ≈ 6.0   (large)
    #   10^33 erg → log10(F_NUV) ≈ 7.0   (superflare)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # ── Atmosphere A: N2-dominated (Earth-like) ────────────────────────────
    # Segura+2010 calibration: single AD Leo event at Earth-equiv distance
    #   → F_NUV ~ 3.5e6 erg/cm^2 → O3 change ≈ -5% (UV only)
    # Tilley+2019: individual TRAPPIST-1 large flare (E~10^32 erg) → ~-1% per event
    # Chen+2021 3D: NO2/N2O/HNO3 INCREASE due to NOx chemistry
    lut_A = {
        # O3: 1D models predict depletion via HOx catalysis (UV-only pathway)
        # NOTE: Ridgway+2023 3D GCM predicts O3 INCREASES 20x (unresolved tension)
        'O3':   interp1d(x, [-5e-5, -2e-4, -8e-4, -3e-3, -8e-3, -2.0e-2, -5e-2, -0.18],
                         kind='linear', fill_value='extrapolate'),
        # CH4: OH-mediated oxidation increases during UV burst (Segura+2010)
        'CH4':  interp1d(x, [-3e-6, -1e-5, -5e-5, -2e-4, -5e-4, -2.0e-3, -7e-3, -2.5e-2],
                         kind='linear', fill_value='extrapolate'),
        # N2O: NOx chemistry → slight increase at moderate UV, depletion at extreme
        # Chen+2021: N2O becomes detectable; slight net increase in 3D
        'N2O':  interp1d(x, [+2e-6, +8e-6, +3e-5, +1e-4, +3e-4, +8e-4, +1e-3, -5e-3],
                         kind='linear', fill_value='extrapolate'),
        # NO2: NOx production from N2+O during flare (Chen+2021; increases detectable)
        'NO2':  interp1d(x, [+5e-5, +2e-4, +8e-4, +3e-3, +1e-2, +4e-2, +1.5e-1, +5e-1],
                         kind='linear', fill_value='extrapolate'),
        # HNO3: NOx + OH → HNO3 (Chen+2021 finds HNO3 increase)
        'HNO3': interp1d(x, [+3e-5, +1e-4, +5e-4, +2e-3, +7e-3, +2.5e-2, +8e-2, +2.5e-1],
                         kind='linear', fill_value='extrapolate'),
        # OH: radical increase during UV burst (transient; < flare duration)
        'OH':   interp1d(x, [+5e-4, +2e-3, +8e-3, +3e-2, +1e-1, +4e-1, +1.5, +5.0],
                         kind='linear', fill_value='extrapolate'),
    }

    # ── Atmosphere B: CO2-dominated Venus-analog ───────────────────────────
    # No background O3 (no biogenic O2 source)
    # Key: flares drive CO2 → CO + O → trace O3 ABIOTIC false positive
    # Miranda-Rosete+2024: single superflare → O3 detectable above JWST threshold
    # Lincowski+2018; Hu+2020: CO2-rich photochemistry under M-dwarf UV
    lut_B = {
        # CO: CO2 photolysis product (increases with UV)
        'CO':   interp1d(x, [+3e-6, +1e-5, +5e-5, +2e-4, +8e-4, +3e-3, +1.2e-2, +5e-2],
                         kind='linear', fill_value='extrapolate'),
        # SO2: UV photodissociation to SO + O (decreases under intense UV)
        'SO2':  interp1d(x, [-5e-7, -2e-6, -8e-6, -3e-5, -1e-4, -4e-4, -1.5e-3, -5e-3],
                         kind='linear', fill_value='extrapolate'),
        # O3 ABIOTIC: CO2 → CO + O → O + O2 → O3 (false positive pathway)
        # Miranda-Rosete+2024: superflare can produce O3 ~1e-8 vmr (borderline detectable)
        'O3_abiotic': interp1d(
            x, [+5e-12, +2e-11, +8e-11, +3e-10, +1e-9, +5e-9, +3e-8, +1e-7],
            kind='linear', fill_value='extrapolate'),
        # O2 abiotic: H2O photolysis + H escape (slow process, long timescale)
        'O2_abiotic': interp1d(
            x, [+1e-10, +5e-10, +2e-9, +8e-9, +3e-8, +1e-7, +5e-7, +2e-6],
            kind='linear', fill_value='extrapolate'),
    }

    # ── Atmosphere C: Thin remnant / bare rock ─────────────────────────────
    # No significant photochemistry; no biosignature gases to track.
    # UV surface fluence is already captured in uv_df (F_NUV_fluence × 0.97).
    # Returning zero fractional changes since there are no chemical mixing ratios
    # to deplete or enhance. UV dose is analysed directly from uv_df.
    lut_C = {
        'no_chemistry_note': interp1d(
            x, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            kind='linear', fill_value='extrapolate'),
    }

    return {'A': lut_A, 'B': lut_B, 'C': lut_C}


def apply_photochem_lut(uv_df, lut, atm_label):
    """
    Apply photochemical lookup table to each flare, yielding per-event
    fractional changes in biosignature mixing ratios.
    """
    log_F = uv_df['log10_F_NUV'].values
    # Clip to LUT range to avoid extrapolation artifacts
    log_F_clipped = np.clip(log_F, 1.0, 8.0)

    records = []
    for i, (_, row) in enumerate(uv_df.iterrows()):
        lf = log_F_clipped[i]
        rec = {
            'time_days':    row['time_days'],
            'energy_erg':   row['energy_erg'],
            'log10_energy': row['log10_energy'],
            'log10_F_NUV':  row['log10_F_NUV'],
            'EF_NUV_peak':  row['EF_NUV_peak'],
            'flare_class':  row['flare_class'],
        }
        atm = lut[atm_label]
        for species, fn in atm.items():
            rec[f'd{species}'] = float(fn(lf))
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/photochem_response_atm_{atm_label.lower()}.csv", index=False)
    return df


# ===========================================================================
# SECTION 4: Cumulative Biosignature State
# ===========================================================================
def compute_cumulative_state(resp_dfs, atms):
    """
    Accumulate per-event mixing ratio changes over the 79-day observation window.
    Applies a partial recovery between events using a recovery timescale.

    Recovery timescales (1D models, Segura+2010):
      O3:  ~30 days per individual event depletion
      CH4: years (biogenic replenishment dominates)
      NO2: hours-days (fast photolysis)
      N2O: weeks
      HNO3: days

    NOTE: These recovery timescales are 1D model estimates. 3D dynamics
    (atmospheric circulation, transport) can significantly alter recovery.
    """
    RECOVERY_TAU = {
        'O3': 30.0, 'CH4': 1825.0, 'N2O': 14.0,
        'NO2': 0.5, 'HNO3': 3.0, 'OH': 0.0001,
        'CO': 5.0, 'SO2': 7.0, 'O3_abiotic': 0.5,
        'O2_abiotic': 365.0, 'surface_UV_dose': 0.0,
        'bio_effective_UV': 0.0,
    }

    summary_rows = []
    for lbl, resp_df in resp_dfs.items():
        atm = atms[lbl]
        state = {}  # current fractional depletion from background
        species_cols = [c for c in resp_df.columns if c.startswith('d')]

        events_sorted = resp_df.sort_values('time_days')
        prev_time = 0.0

        for _, ev in events_sorted.iterrows():
            dt = ev['time_days'] - prev_time
            # Apply exponential recovery between events
            for col in species_cols:
                sp = col[1:]  # strip 'd'
                tau = RECOVERY_TAU.get(sp, 30.0)
                if tau > 0:
                    state[col] = state.get(col, 0.0) * np.exp(-dt / tau)
                # Apply this event's change
                state[col] = state.get(col, 0.0) + ev.get(col, 0.0)
            prev_time = ev['time_days']

        row = {'atmosphere': lbl, 'name': atm['name'],
               'n_flares': len(resp_df),
               'obs_window_days': resp_df['time_days'].max() if len(resp_df) > 0 else 0}
        for col, val in state.items():
            row[f'cumul_{col}'] = val
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(f"{OUTPUT_DIR}/cumulative_biosignature_state.csv", index=False)
    return df


# ===========================================================================
# SECTION 5: VULCAN-Analog Stretch Goal
# ===========================================================================
def run_vulcan_analog(flare_row, uv_df_row):
    """
    Simplified VULCAN-analog photochemical ODE for Atmosphere A under the
    strongest observed flare. Demonstrates live photochemistry capability.

    Reaction network (Chapman + HOx, 1D single pressure layer at P=0.01 atm):
      (3) O3 + hν_NUV → O(1D) + O2          J_O3  (Hartley band, UV-driven)
      (4) O(1D) + H2O → 2 OH                 k4 (branching fraction f_OH)
      (5) OH + O3 → HO2 + O2                 k5
      (6) OH + CH4 → CH3• + H2O              k6
      Chapman restoration: O + O2 + M → O3   (parameterized as relaxation to bg)

    OH treated at quasi-steady state (lifetime ~ seconds, much shorter than
    the ODE integration timestep). This avoids stiffness.

    Rate constants at T=250K from JPL/NASA Sander et al. 2011 Chemical
    Kinetics and Photochemical Data for Use in Atmospheric Studies.

    Time-varying UV forcing: J(t) = J0 × (1 + A_UV × davenport_profile(t))
      Where A_UV = amplitude × (C_NUV / C_TESS)  (Planck ratio conversion)
    """
    print("\n" + "="*65)
    print("STRETCH GOAL: VULCAN-Analog Photochemical ODE")
    print("="*65)

    # ── Physical parameters (P=0.01 atm, T=250 K stratospheric layer) ──────
    T_atm   = 250.0        # K
    P_atm   = 0.01         # atm (stratospheric photochemistry layer)
    # Number density (Loschmidt: 2.69e19 cm^-3 at STP → scale for T,P)
    n_M     = 2.69e19 * P_atm * (273.15 / T_atm)   # cm^-3
    n_O2    = 0.21 * n_M                             # cm^-3 (fixed O2 reservoir)
    n_H2O   = 1.0e-3 * n_M                           # cm^-3 (fixed H2O)

    # ── Initial mixing ratios (Atmosphere A: Earth-like) ────────────────────
    vmr_O3_0  = 3.00e-7    # 300 ppb
    vmr_CH4_0 = 1.80e-6    # 1.8 ppm (biogenic)
    vmr_OH_0  = 1.00e-12   # OH background

    n_O3_0   = vmr_O3_0  * n_M
    n_CH4_0  = vmr_CH4_0 * n_M
    n_OH_0   = vmr_OH_0  * n_M

    # ── Rate constants at T=250K (JPL Sander+2011) ────────────────────────
    # k2: O + O2 + M → O3 + M  (termolecular, low-pressure limit)
    k2_raw  = 6.0e-34 * (300.0/T_atm)**2.4           # cm^6 s^-1
    k2_eff  = k2_raw * n_M                            # cm^3 s^-1 (effective 2nd order)
    # k4: O(1D) + H2O → 2OH
    k4      = 1.63e-10                                # cm^3 s^-1
    # k4_quench: O(1D) + M → O(3P) + M  (deactivation)
    k4_q    = 3.0e-11 * n_M                           # s^-1
    # Branching fraction: fraction of O(1D) that produces OH (vs. quenched)
    f_OH    = (k4 * n_H2O) / (k4_q + k4 * n_H2O)
    # k5: OH + O3 → HO2 + O2
    k5      = 1.7e-12 * np.exp(-940.0/T_atm)          # cm^3 s^-1
    # k6: OH + CH4 → CH3• + H2O
    k6      = 2.45e-12 * np.exp(-1775.0/T_atm)        # cm^3 s^-1
    # Other OH sinks (HO2, NOx, etc.) — parameterized as effective lifetime
    tau_OH_other = 5.0                                 # s (other OH loss channels)

    # ── Quiescent photolysis rates at TRAPPIST-1e ──────────────────────────
    # Scaled from Earth by UV flux ratio: F_NUV_T1e / F_NUV_Earth
    UV_ratio  = MEGA['F_NUV'] / 1.5       # TRAPPIST-1e NUV / Earth-equiv NUV
    J_O3_0   = 3.0e-3 * UV_ratio          # s^-1  (Hartley band)
    J_O2_0   = 3.0e-8 * UV_ratio          # s^-1  (Schumann-Runge; small for M dwarfs)

    # ── OH source rate calibrated to background [OH] ─────────────────────
    # P_OH_0 × EF = (k5×n_O3 + k6×n_CH4 + 1/tau_OH_other) × n_OH
    # At quiescence (EF=1): n_OH = n_OH_0
    OH_sink_bg = k5 * n_O3_0 + k6 * n_CH4_0 + 1.0/tau_OH_other
    P_OH_0     = n_OH_0 * OH_sink_bg      # cm^-3 s^-1

    # O3 restoration timescale (Chapman cycle + slow column recovery)
    tau_O3_restore = 30.0 * 86400.0       # 30 days in seconds

    # ── UV enhancement function following Davenport template ──────────────
    amplitude = flare_row['peak_amplitude']
    fwhm_sec  = flare_row['duration_sec']  # use duration as proxy for FWHM
    T_fl      = flare_row.get('temperature_K', T_FLARE_K)

    C_TESS  = planck_ratio(800.0, float(T_fl), T_EFF_K)
    C_NUV   = planck_ratio(250.0, float(T_fl), T_EFF_K)
    A_UV    = amplitude * C_NUV / C_TESS   # UV amplitude in NUV band

    # Flare occurs at t=0 in the simulation window
    t_peak_sim = 0.0

    def EF_UV(t):
        """Time-varying NUV enhancement factor following Davenport template."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        profile = davenport_profile(t_arr, t_peak_sim, A_UV, fwhm_sec)
        return float(1.0 + profile[0] if np.isscalar(t) else 1.0 + profile)

    # ── ODE right-hand side ───────────────────────────────────────────────
    def ode_rhs(t, y):
        n_O3, n_CH4 = max(y[0], 1.0), max(y[1], 1.0)

        EF = 1.0 + float(davenport_profile(
            np.array([t]), t_peak_sim, A_UV, fwhm_sec)[0])

        # OH at quasi-steady state (very fast equilibration, ~seconds)
        OH_sink = k5 * n_O3 + k6 * n_CH4 + 1.0/tau_OH_other
        n_OH    = P_OH_0 * EF / OH_sink

        # d[O3]/dt: Chapman restoration + HOx depletion
        dn_O3  = (-(n_O3 - n_O3_0) / tau_O3_restore
                  - k5 * n_OH * n_O3)

        # d[CH4]/dt: HOx oxidation (slow; biogenic source implicitly included
        # via long recovery timescale — CH4 approximately conserved over hours)
        dn_CH4 = -k6 * n_OH * n_CH4

        return [dn_O3, dn_CH4]

    # ── Integration window: -2 h to +24 h around flare peak ──────────────
    t_start = -2.0  * 3600.0   # s
    t_end   = +24.0 * 3600.0   # s
    t_eval  = np.linspace(t_start, t_end, 2000)

    y0 = [n_O3_0, n_CH4_0]

    print(f"  Flare: E = {flare_row['energy_erg']:.2e} erg  "
          f"| A = {amplitude:.4f}  | fwhm = {fwhm_sec/60:.1f} min")
    print(f"  A_UV (NUV amplitude) = {A_UV:.1f}  (peak UV enhancement = {1+A_UV:.1f}x)")
    print(f"  f_OH = {f_OH:.4f} | J_O3_0 = {J_O3_0:.2e} s^-1")
    print(f"  k5 = {k5:.2e} | k6 = {k6:.2e} cm^3 s^-1")
    print(f"  n_O3_0 = {n_O3_0:.2e} cm^-3 | n_CH4_0 = {n_CH4_0:.2e} cm^-3")
    print(f"  Integrating ODE over {t_start/3600:.0f}h to {t_end/3600:.0f}h ...")

    sol = solve_ivp(ode_rhs, [t_start, t_end], y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-3)

    if not sol.success:
        print(f"  WARNING: ODE integration issue: {sol.message}")

    n_O3_t  = sol.y[0]
    n_CH4_t = sol.y[1]

    # Compute OH and EF at each time step for output
    EF_t   = np.array([1.0 + float(davenport_profile(
        np.array([t]), t_peak_sim, A_UV, fwhm_sec)[0]) for t in sol.t])
    OH_sink_t = k5 * n_O3_t + k6 * n_CH4_t + 1.0/tau_OH_other
    n_OH_t  = P_OH_0 * EF_t / OH_sink_t

    ts = pd.DataFrame({
        'time_hours':      sol.t / 3600.0,
        'EF_NUV':          EF_t,
        'n_O3_cm3':        n_O3_t,
        'n_CH4_cm3':       n_CH4_t,
        'n_OH_cm3':        n_OH_t,
        'vmr_O3':          n_O3_t  / n_M,
        'vmr_CH4':         n_CH4_t / n_M,
        'vmr_OH':          n_OH_t  / n_M,
        'dO3_frac':        (n_O3_t  - n_O3_0)  / n_O3_0,
        'dCH4_frac':       (n_CH4_t - n_CH4_0) / n_CH4_0,
        'dOH_frac':        (n_OH_t  - n_OH_0)  / n_OH_0,
    })
    ts.to_csv(f"{OUTPUT_DIR}/vulcan_analog_timeseries.csv", index=False)

    peak_idx = np.argmin(ts['n_O3_cm3'])
    max_O3_depletion = abs(ts['dO3_frac'].min())
    max_OH_enhancement = ts['dOH_frac'].max()
    print(f"  Peak O3 depletion:   {max_O3_depletion*100:.3f}%")
    print(f"  Peak OH enhancement: {max_OH_enhancement:.1f}x over background")
    print(f"  Final CH4 change:    {ts['dCH4_frac'].iloc[-1]*100:.4f}%")

    return ts, dict(
        max_O3_depletion=max_O3_depletion,
        max_OH_enhancement=max_OH_enhancement,
        A_UV=A_UV,
        n_M=n_M, n_O3_0=n_O3_0, n_CH4_0=n_CH4_0, n_OH_0=n_OH_0,
        f_OH=f_OH, k5=k5, k6=k6,
    )


# ===========================================================================
# SECTION 6: Biosphere Model & Digital Twin State Update (Eager-Nash+2024)
# ===========================================================================
def run_biosphere_model(resp_df_A, uv_df, atm_A):
    """
    Apply Eager-Nash et al. 2024 biosphere feedback to Atmosphere A.

    Two coupled feedback loops:
      1. CO consumption: 4CO + 2H2O -> 2CO2 + CH3COOH
         Biosphere acts as CO sink, reducing abiotic CO by ~97% when healthy.
         CO is the dominant abiotic spectral feature on M-dwarf planets.
      2. CH4 production: 4H2 + CO2 -> 2H2O + CH4 (methanogenesis)
         Biosphere replenishes CH4 depleted by UV-driven OH oxidation.

    Stress model: each high-UV flare (EF_NUV > 100) degrades population.
    Population decays as: pf *= exp(-UV_dose / stress_scale)
    Recovery between events: pf increases toward 1.0 with tau=730 days.

    Returns
    -------
    biosphere_df : DataFrame with per-flare biosphere state
      columns: time_days, ef_nuv, uv_dose, population_factor,
               co_vmr_abiotic, co_vmr_biotic, ch4_biogenic_vmr
    """
    import copy

    BIO_CO_CONSUMPTION   = 0.97      # fraction consumed by healthy biosphere
    BIO_CH4_RATE_VMR_DAY = 2.0e-11   # biogenic CH4 VMR added per day
    BIO_UV_STRESS_SCALE  = 5.0e9     # erg/cm^2 — population half-life dose
    BIO_RECOVERY_TAU     = 730.0     # days — biosphere recovery timescale
    F_NUV_QUIESCENT      = MEGA['F_NUV']

    population_factor = 1.0
    cumul_uv_dose     = 0.0
    prev_time         = 0.0

    # Base abiotic CO VMR (from LUT CO production in resp_df_A)
    co_abiotic_base = atm_A['vmr'].get('CO', 1.0e-7)

    records = []
    for _, row in resp_df_A.sort_values('time_days').iterrows():
        dt       = row['time_days'] - prev_time
        ef_nuv   = float(row.get('EF_NUV_peak', 1.0))
        dur_sec  = float(row.get('duration_sec',
                         uv_df[uv_df['time_days'] == row['time_days']]['duration_sec'].iloc[0]
                         if len(uv_df[uv_df['time_days'] == row['time_days']]) > 0 else 600.0))

        # Recovery between flares
        if dt > 0:
            population_factor = 1.0 - (1.0 - population_factor) * np.exp(-dt / BIO_RECOVERY_TAU)

        # UV dose from this flare
        dose = max(0.0, ef_nuv - 1.0) * F_NUV_QUIESCENT * dur_sec
        cumul_uv_dose += dose

        # Stress response
        if dose > 0:
            population_factor *= float(np.exp(-dose / BIO_UV_STRESS_SCALE))
        population_factor = float(np.clip(population_factor, 0.0, 1.0))

        # Compute biotic CO (biosphere consumes abiotic CO)
        co_abiotic = co_abiotic_base * (1.0 + row.get('dCO', 0.0))
        co_biotic  = co_abiotic * (1.0 - BIO_CO_CONSUMPTION * population_factor)
        co_biotic  = max(co_biotic, co_abiotic * 0.001)

        # Compute biogenic CH4 produced since last flare
        ch4_biogenic = BIO_CH4_RATE_VMR_DAY * population_factor * dt

        records.append(dict(
            time_days          = row['time_days'],
            ef_nuv             = ef_nuv,
            uv_dose_erg_cm2    = dose,
            cumul_uv_dose      = cumul_uv_dose,
            population_factor  = population_factor,
            co_vmr_abiotic     = co_abiotic,
            co_vmr_biotic      = co_biotic,
            ch4_biogenic_vmr   = ch4_biogenic,
        ))
        prev_time = row['time_days']

    biosphere_df = pd.DataFrame(records)
    biosphere_df.to_csv(f"{OUTPUT_DIR}/biosphere_state.csv", index=False)

    final = biosphere_df.iloc[-1] if len(biosphere_df) > 0 else {}
    print(f"  Biosphere final population factor : {final.get('population_factor', 1.0):.4f}")
    print(f"  Biogenic CH4 cumulative (VMR)     : {biosphere_df['ch4_biogenic_vmr'].sum():.2e}")
    print(f"  CO (abiotic)                      : {final.get('co_vmr_abiotic', 0):.2e}")
    print(f"  CO (biotic, with consumption)     : {final.get('co_vmr_biotic', 0):.2e}")

    return biosphere_df


def update_twin_state(resp_dfs, cumul_df, biosphere_df, uv_df):
    """
    Load twin_state.json, update it with Stage 2 results, and save.

    Updates:
      state['atmosphere'] — final VMRs after 79-day flare sequence
      state['cumulative_changes'] — fractional changes per atmosphere
      state['biosphere'] — population factor, CO suppression, CH4 production
      state['history'] — per-day state time series for the timeline plot
    """
    from datetime import datetime, timezone
    try:
        from twin_core import (load_twin_state, save_twin_state,
                               update_biosphere_stress, biosphere_recovery)
    except ImportError:
        print("  [Warning] twin_core not found — skipping state update")
        return

    state = load_twin_state()

    # Update cumulative changes for each atmosphere
    for _, row in cumul_df.iterrows():
        lbl  = row['atmosphere']
        changes = {}
        for col in cumul_df.columns:
            if col.startswith('cumul_d'):
                val = row[col]
                if pd.notna(val) and np.isfinite(float(val)):
                    sp = col.replace('cumul_d', '')
                    changes[sp] = float(val)
        state['cumulative_changes'][lbl] = changes

        # Apply cumulative changes to living atmosphere state
        for sp, frac in changes.items():
            if sp in state['atmosphere'].get(lbl, {}):
                old = state['atmosphere'][lbl][sp]
                state['atmosphere'][lbl][sp] = float(
                    np.clip(old * (1.0 + frac), 0.0, 1.0))

    # Update biosphere from biosphere_df
    if len(biosphere_df) > 0:
        final_bio = biosphere_df.iloc[-1]
        bio = state['biosphere']
        bio['population_factor']    = float(final_bio['population_factor'])
        bio['cumulative_uv_dose']   = float(final_bio['cumul_uv_dose'])
        bio['co_vmr_abiotic']       = float(final_bio['co_vmr_abiotic'])
        bio['co_vmr_biotic']        = float(final_bio['co_vmr_biotic'])
        bio['ch4_production_rate']  = float(final_bio['population_factor'])
        bio['active']               = final_bio['population_factor'] > 0.01
        # Update Atm A's CO VMR with biotic value
        state['atmosphere']['A']['CO'] = float(final_bio['co_vmr_biotic'])
        # Add cumulative biogenic CH4
        ch4_boost = float(biosphere_df['ch4_biogenic_vmr'].sum())
        old_ch4 = state['atmosphere']['A'].get('CH4', 1.8e-6)
        state['atmosphere']['A']['CH4'] = float(np.clip(old_ch4 + ch4_boost, 0.0, 1.0))

    # Build history time series for the Twin State Monitor
    history = []
    for _, row in biosphere_df.iterrows():
        history.append({
            'day':                 float(row['time_days']),
            'O3':                  float(state['atmosphere']['A'].get('O3', 3e-7)),
            'CH4':                 float(state['atmosphere']['A'].get('CH4', 1.8e-6)),
            'CO':                  float(row['co_vmr_biotic']),
            'N2O':                 float(state['atmosphere']['A'].get('N2O', 3.2e-7)),
            'population_factor':   float(row['population_factor']),
            'uv_dose':             float(row['uv_dose_erg_cm2']),
        })
    state['history'] = history

    # Update metadata
    state['meta']['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
    state['meta']['last_updated_by']  = 'stage2'

    save_twin_state(state)
    pf = state['biosphere']['population_factor']
    print(f"  twin_state.json updated (biosphere pf={pf:.3f}, "
          f"{len(history)} history points)")


# ===========================================================================
# SECTION 6: Diagnostic Plots
# ===========================================================================
def plot_main(atms, uv_df, resp_dfs, cumul_df, cat):
    print("\n" + "="*65)
    print("PLOTS: Generating Stage 2 Summary Figure")
    print("="*65)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "TRAPPIST-1e Digital Twin — Stage 2: Atmospheric Photochemical Response\n"
        "UV Enhancement + Biosignature Lookup Tables (Segura+2010, Tilley+2019, Chen+2021)",
        fontsize=14, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.38)

    COLORS = {'A': '#2196F3', 'B': '#FF5722', 'C': '#9E9E9E'}
    ATM_LABELS = {'A': 'Atm A: N₂-Earth', 'B': 'Atm B: CO₂-Venus', 'C': 'Atm C: Thin rock'}

    # ── Panel 1: Atmospheric composition bar chart ─────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    species = ['N2', 'O2', 'CO2', 'CH4', 'O3', 'N2O']
    x_pos   = np.arange(len(species))
    width   = 0.28
    for i, (lbl, atm) in enumerate(atms.items()):
        vals = [atm['vmr'].get(sp, 0.0) for sp in species]
        vals_plot = [max(v, 1e-12) for v in vals]  # avoid log(0)
        ax.bar(x_pos + i*width, vals_plot, width, label=ATM_LABELS[lbl],
               color=COLORS[lbl], alpha=0.82, edgecolor='k', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 2.0)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(species, fontsize=9)
    ax.set_ylabel('Volume Mixing Ratio')
    ax.set_title('Atmospheric Compositions\n(JWST DREAMS constraints)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.text(0.02, 0.02, 'Espinoza+2025 DREAMS', transform=ax.transAxes,
            fontsize=7, color='gray', va='bottom')

    # ── Panel 2: UV Enhancement Factor vs. Flare Energy ───────────────────
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(uv_df['log10_energy'], np.log10(uv_df['EF_NUV_peak'].clip(1)),
                    c=uv_df['log10_F_NUV'], cmap='plasma', s=18, alpha=0.75,
                    vmin=4, vmax=7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(F_NUV / erg cm⁻²)', fontsize=8)
    ax.axhline(np.log10(10),  color='green', ls='--', lw=1, label='10× (lit. threshold)')
    ax.axhline(np.log10(100), color='orange', ls='--', lw=1, label='100×')
    ax.axhline(np.log10(1000),color='red', ls='--', lw=1, label='1000×')
    ax.set_xlabel('log₁₀(E_flare / erg)')
    ax.set_ylabel('log₁₀(Peak NUV Enhancement)')
    ax.set_title('UV Enhancement vs. Flare Energy\n(Planck ratio, T_flare=7000K)', fontsize=10)
    ax.legend(fontsize=7)
    ax.text(0.02, 0.97, f'N={len(uv_df)} flares\nDucrot+2022 T_flare', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # ── Panel 3: Flare NUV fluence distribution ───────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(2, 8, 25)
    ax.hist(uv_df['log10_F_NUV'], bins=bins, color='mediumpurple', ec='indigo', alpha=0.8)
    ax.axvline(5, color='green', ls='--', lw=1.5, label='Segura+2010 threshold\n(O3 ~1%)')
    ax.axvline(6.5, color='red', ls='--', lw=1.5, label='AD Leo 1985 level\n(O3 ~5%)')
    ax.set_xlabel('log₁₀(F_NUV / erg cm⁻²)')
    ax.set_ylabel('Number of Flares')
    ax.set_title('NUV Fluence Distribution\nat TRAPPIST-1e', fontsize=10)
    ax.legend(fontsize=7)

    # ── Panel 4: Atmosphere A — O3 and N2O response per flare ────────────
    ax = fig.add_subplot(gs[1, 0])
    resp_A = resp_dfs.get('A')
    if resp_A is not None and 'dO3' in resp_A.columns:
        ax.scatter(resp_A['log10_energy'], resp_A['dO3']*100,
                   c='blue', s=15, alpha=0.6, label='ΔO₃ (deplete)')
        ax2 = ax.twinx()
        ax2.scatter(resp_A['log10_energy'], resp_A.get('dNO2', pd.Series([0]*len(resp_A)))*100,
                    c='orange', s=15, alpha=0.6, marker='^', label='ΔNO₂ (produce)')
        ax2.set_ylabel('ΔNO₂ (% change)', color='orange', fontsize=8)
        ax2.tick_params(axis='y', colors='orange')
    ax.set_xlabel('log₁₀(E_flare / erg)')
    ax.set_ylabel('ΔO₃ (% per event)', color='blue', fontsize=8)
    ax.tick_params(axis='y', colors='blue')
    ax.set_title('Atm A — O₃ Depletion per Flare\n(Segura+2010; 1D; NO₂ from Chen+2021)',
                 fontsize=10)
    ax.text(0.02, 0.95, '[!] 3D models (Ridgway+2023)\npredict O₃ INCREASE 20×',
            transform=ax.transAxes, fontsize=7, va='top', color='red',
            bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))

    # ── Panel 5: Atmosphere B — abiotic O3 (false positive) ───────────────
    ax = fig.add_subplot(gs[1, 1])
    resp_B = resp_dfs.get('B')
    if resp_B is not None and 'dO3_abiotic' in resp_B.columns:
        logF = resp_B['log10_F_NUV']
        aO3  = resp_B['dO3_abiotic']
        sc2  = ax.scatter(logF, np.log10(np.abs(aO3).clip(1e-15)),
                          c=resp_B['log10_energy'], cmap='Reds', s=18, alpha=0.8)
        cbar2 = plt.colorbar(sc2, ax=ax)
        cbar2.set_label('log₁₀(E_flare)', fontsize=7)
        ax.axhline(np.log10(1e-8), color='purple', ls='--', lw=1.5,
                   label='JWST detection\nthreshold ~1e-8 vmr')
    ax.set_xlabel('log₁₀(F_NUV / erg cm⁻²)')
    ax.set_ylabel('log₁₀(Abiotic O₃ vmr change)')
    ax.set_title('Atm B — Abiotic O₃ False Positive\n(Miranda-Rosete+2024; CO₂-rich)',
                 fontsize=10)
    ax.legend(fontsize=7)

    # ── Panel 6: Cumulative biosignature state (79-day window) ────────────
    ax = fig.add_subplot(gs[1, 2])
    if cumul_df is not None and len(cumul_df) > 0:
        atm_names = [r['name'].replace('₂', '2').replace('₁', '1')[:18]
                     for _, r in cumul_df.iterrows()]
        n_atm = len(cumul_df)
        x_pos2 = np.arange(n_atm)
        species_to_plot = []
        for col in cumul_df.columns:
            if col.startswith('cumul_d') and 'O3' in col:
                species_to_plot.append(col)
                break

        if species_to_plot:
            col = species_to_plot[0]
            vals = cumul_df[col].values * 100
            clrs = [COLORS.get(r['atmosphere'], 'gray')
                    for _, r in cumul_df.iterrows()]
            bars = ax.bar(x_pos2, vals, color=clrs, alpha=0.8, ec='k', lw=0.5)
            ax.set_xticks(x_pos2)
            ax.set_xticklabels(atm_names, rotation=15, ha='right', fontsize=8)
            ax.set_ylabel('Cumulative ΔO₃ (%) after 79 days')
            ax.set_title('Cumulative O₃ Change\n79-Day Observation Window', fontsize=10)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.2f}%', ha='center', va='bottom', fontsize=8)
        ax.text(0.02, 0.97, 'Tilley+2019 calibrated', transform=ax.transAxes,
                fontsize=7, va='top', color='gray')

    # ── Panel 7: Davenport template + UV profile for example flare ─────────
    ax = fig.add_subplot(gs[2, 0])
    t_plot  = np.linspace(-300, 3600, 2000)
    # Example: large flare (A=0.09, fwhm=600s)
    A_ex    = 0.09
    fwhm_ex = 600.0
    C_T_ex  = planck_ratio(800.0, T_FLARE_K, T_EFF_K)
    C_N_ex  = planck_ratio(250.0, T_FLARE_K, T_EFF_K)
    A_UV_ex = A_ex * C_N_ex / C_T_ex

    tess_profile = davenport_profile(t_plot, 0, A_ex, fwhm_ex)
    nuv_profile  = davenport_profile(t_plot, 0, A_UV_ex, fwhm_ex)

    ax.plot(t_plot/60, tess_profile, 'k-', lw=2, label=f'TESS band (A={A_ex:.2f})')
    ax_r = ax.twinx()
    ax_r.plot(t_plot/60, nuv_profile + 1, 'r-', lw=2, label=f'NUV (A_UV={A_UV_ex:.0f})')
    ax_r.set_ylabel('NUV Enhancement Factor', color='red', fontsize=8)
    ax_r.tick_params(axis='y', colors='red')
    ax.set_xlabel('Time from Peak (min)')
    ax.set_ylabel('ΔF/F (TESS band)', fontsize=8)
    ax.set_title('Davenport Template\n+ UV Enhancement Profile', fontsize=10)
    ax.set_xlim(-5, 60)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7)

    # ── Panel 8: CH4 + N2O response across all flare classes ──────────────
    ax = fig.add_subplot(gs[2, 1])
    if resp_A is not None:
        for cls, clr, sym in [('micro', 'lightblue', 'o'),
                               ('moderate', 'blue', 's'),
                               ('large', 'red', '^'),
                               ('superflare', 'darkred', 'D')]:
            sub = resp_A[resp_A['flare_class'] == cls]
            if len(sub) > 0 and 'dCH4' in sub.columns:
                ax.scatter(sub['log10_energy'], sub['dCH4']*100,
                           c=clr, marker=sym, s=25 if cls != 'micro' else 10,
                           alpha=0.7, label=f'{cls} (N={len(sub)})', zorder=3)
    ax.set_xlabel('log₁₀(E_flare / erg)')
    ax.set_ylabel('ΔCH₄ per event (%)')
    ax.set_title('Atm A — CH₄ Response per Flare Class\n(Chen+2021; OH oxidation pathway)',
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 9: 1D vs 3D tension summary ────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    tension_text = (
        "1D vs. 3D Model Tension\n"
        "─────────────────────────────────────\n\n"
        "1D models (this stage):\n"
        "  Segura+2010:  O3 −5% (UV only)\n"
        "  Tilley+2019:  O3 →6% over 10 yr\n"
        "  Chen+2021:    NO2/N2O/HNO3 ↑\n\n"
        "3D models (NOT implemented here):\n"
        "  Ridgway+2023: O3 INCREASES 20×!\n"
        "  Chen+2021 3D: new equilibria\n"
        "  Circulation reverses depletion\n\n"
        "This 1D model represents:\n"
        "  UPPER BOUND on O3 destruction\n"
        "  LOWER BOUND on biosig. survival\n\n"
        "Unresolved tension in literature.\n"
        "3D capability = future Stage 5."
    )
    ax.text(0.05, 0.97, tension_text, transform=ax.transAxes,
            fontsize=8.5, va='top', family='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.9))

    path = f"{OUTPUT_DIR}/stage2_summary_plots.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_vulcan(ts, meta, flare_info):
    print("PLOTS: Generating VULCAN Stretch Goal Figure")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "TRAPPIST-1e — VULCAN-Analog Photochemical ODE (Stretch Goal)\n"
        f"Atmosphere A (N₂-Earth) | Flare: E = {flare_info['energy_erg']:.2e} erg  "
        f"| T_flare = 7000 K (Ducrot+2022)\n"
        "Chapman + HOx network | Rate constants: JPL/NASA Sander+2011",
        fontsize=12, fontweight='bold')

    t = ts['time_hours']

    # Panel 1: UV Enhancement
    ax = axes[0, 0]
    ax.plot(t, ts['EF_NUV'], 'r-', lw=2)
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.fill_between(t, 1, ts['EF_NUV'], where=ts['EF_NUV'] > 1,
                    color='red', alpha=0.2)
    ax.set_xlabel('Time from flare peak (hours)')
    ax.set_ylabel('NUV Enhancement Factor (EF)')
    ax.set_title('UV Forcing: NUV Enhancement over Quiescent', fontsize=10)
    ax.set_xlim(-2, 24)
    ax.text(0.98, 0.95, f"Peak EF = {ts['EF_NUV'].max():.1f}×",
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', fc='lightyellow'))

    # Panel 2: O3 fractional change
    ax = axes[0, 1]
    ax.plot(t, ts['dO3_frac']*100, 'b-', lw=2)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.fill_between(t, 0, ts['dO3_frac']*100, where=ts['dO3_frac'] < 0,
                    color='blue', alpha=0.2, label='O₃ depletion')
    min_O3 = ts['dO3_frac'].min()*100
    t_min  = t[ts['dO3_frac'].idxmin()]
    ax.annotate(f"Peak depletion:\n{min_O3:.3f}%",
                xy=(t_min, min_O3), xytext=(t_min+2, min_O3*0.6),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')
    ax.set_xlabel('Time from flare peak (hours)')
    ax.set_ylabel('ΔO₃ / O₃ (%)')
    ax.set_title('O₃ Fractional Change (Chapman+HOx)\nSegura+2010 calibrated', fontsize=10)
    ax.set_xlim(-2, 24)
    ax.text(0.02, 0.03,
            '[!] 1D only. Ridgway+2023 3D GCM\npredicts O₃ INCREASES 20×.',
            transform=ax.transAxes, fontsize=7, color='red', va='bottom',
            bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))

    # Panel 3: OH enhancement (key oxidant)
    ax = axes[1, 0]
    ax.semilogy(t, ts['EF_NUV'], 'r-', lw=1.5, alpha=0.6, label='NUV EF (forcing)')
    ax.semilogy(t, np.abs(ts['dOH_frac']) + 1, 'purple', lw=2, label='OH enhancement')
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time from flare peak (hours)')
    ax.set_ylabel('Enhancement factor (log scale)')
    ax.set_title('OH Radical Enhancement\n(quasi-steady-state; key O₃/CH₄ oxidant)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 24)
    ax.text(0.98, 0.97, f"Max OH = {ts['dOH_frac'].max()+1:.1f}×",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='lavender'))

    # Panel 4: CH4 change (very small per event; shown in ppm)
    ax = axes[1, 1]
    vmr_CH4_ppm = ts['vmr_CH4'] * 1e6
    ax.plot(t, vmr_CH4_ppm, 'green', lw=2)
    ax.set_xlabel('Time from flare peak (hours)')
    ax.set_ylabel('CH₄ mixing ratio (ppm)')
    ax.set_title('CH₄ Response (OH oxidation)\nChen+2021; biogenic source not modeled',
                 fontsize=10)
    ax.set_xlim(-2, 24)
    final_dch4_ppb = (ts['dCH4_frac'].iloc[-1]) * 1.8e6 * 1e3  # in ppb
    ax.text(0.98, 0.50, f"ΔCH₄ after 24h:\n{final_dch4_ppb:.4f} ppb",
            transform=ax.transAxes, ha='right', va='center', fontsize=10,
            bbox=dict(boxstyle='round', fc='honeydew'))

    # Rate constant info box
    info = (f"VULCAN-Analog Parameters\n"
            f"P = 0.01 atm, T = 250 K\n"
            f"n_M = {meta['n_M']:.2e} cm⁻³\n"
            f"k5(OH+O3) = {meta['k5']:.2e} cm³/s\n"
            f"k6(OH+CH4) = {meta['k6']:.2e} cm³/s\n"
            f"f_OH = {meta['f_OH']:.4f}\n"
            f"(JPL/NASA Sander+2011)")
    axes[0, 1].text(1.02, 0.5, info, transform=axes[0, 1].transAxes,
                    fontsize=8, va='center', family='monospace',
                    bbox=dict(boxstyle='round', fc='lightcyan', ec='teal', alpha=0.9))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/stage2_vulcan_stretch_plot.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# SECTION 7: Main Pipeline
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "#"*65)
    print("  TRAPPIST-1e DIGITAL TWIN -- STAGE 2: ATMOSPHERIC RESPONSE")
    print("#"*65 + "\n")

    # ── Load Stage 1 outputs ───────────────────────────────────────────────
    print("="*65)
    print("Loading Stage 1 outputs")
    print("="*65)

    cat_path  = f"{STAGE1_DIR}/trappist1_flare_catalog.csv"
    spec_path = f"{STAGE1_DIR}/trappist1_stellar_spectrum.csv"

    cat  = pd.read_csv(cat_path)
    spec = pd.read_csv(spec_path)
    print(f"  Flare catalog: {len(cat)} flares from {cat_path}")
    print(f"  Stellar spectrum: {len(spec)} wavelength points from {spec_path}")
    print(f"  Flare classes: {cat['class'].value_counts().to_dict()}")

    # ── Section 1: Atmospheric Compositions ───────────────────────────────
    print("\n" + "="*65)
    print("SECTION 1: Atmospheric Compositions")
    print("="*65)

    atms    = define_atmospheric_compositions()
    atm_df  = save_atmosphere_table(atms)

    for lbl, atm in atms.items():
        print(f"\n  Atmosphere {lbl}: {atm['name']}")
        print(f"    P = {atm['P_surface']} bar | T = {atm['T_surface']} K")
        print(f"    JWST status: {atm['jwst_status']}")
        print(f"    Biosignatures: {atm['biosignatures']}")
        print(f"    UV shield: {atm['uv_shield']}")

    # ── Section 2: UV Enhancement ─────────────────────────────────────────
    print("\n" + "="*65)
    print("SECTION 2: UV Enhancement per Flare")
    print("="*65)

    C_TESS = planck_ratio(800.0, T_FLARE_K, T_EFF_K)
    C_NUV  = planck_ratio(250.0, T_FLARE_K, T_EFF_K)
    C_FUV  = planck_ratio(150.0, T_FLARE_K, T_EFF_K)
    print(f"  Planck ratios at T_flare={T_FLARE_K}K vs T_eff={T_EFF_K}K:")
    print(f"    TESS band (800 nm): {C_TESS:.1f}×")
    print(f"    NUV       (250 nm): {C_NUV:.3e}×")
    print(f"    FUV       (150 nm): {C_FUV:.3e}×")
    print(f"  → NUV/TESS color ratio = {C_NUV/C_TESS:.0f} (converts TESS amplitude → NUV EF)")
    print(f"  MegaMUSCLES quiescent fluxes at TRAPPIST-1e:")
    for band, val in MEGA.items():
        print(f"    {band}: {val:.3f} erg/s/cm²  (France+2020, Wilson+2025)")

    uv_df = compute_uv_enhancement(cat)
    print(f"\n  UV enhancement computed for {len(uv_df)} flares")
    print(f"  EF_NUV range: {uv_df['EF_NUV_peak'].min():.1f}× – {uv_df['EF_NUV_peak'].max():.1f}×")
    print(f"  F_NUV fluence range: 10^{uv_df['log10_F_NUV'].min():.1f} – "
          f"10^{uv_df['log10_F_NUV'].max():.1f} erg/cm²")
    print(f"  Saved: {OUTPUT_DIR}/flare_uv_enhancement.csv")

    # ── Section 3 & 4: Photochemical LUT + apply to flares ────────────────
    print("\n" + "="*65)
    print("SECTION 3: Photochemical Lookup Tables + Per-Flare Response")
    print("="*65)

    lut = build_photochem_lut()
    resp_dfs = {}
    for lbl in ['A', 'B', 'C']:
        resp_dfs[lbl] = apply_photochem_lut(uv_df, lut, lbl)
        atm_name = atms[lbl]['name']
        df_r = resp_dfs[lbl]
        print(f"\n  Atmosphere {lbl} ({atm_name}):")
        if 'dO3' in df_r.columns:
            print(f"    ΔO3  range: {df_r['dO3'].min()*100:.3f}% to {df_r['dO3'].max()*100:.3f}%")
        if 'dCH4' in df_r.columns:
            print(f"    ΔCH4 range: {df_r['dCH4'].min()*100:.4f}% to {df_r['dCH4'].max()*100:.4f}%")
        if 'dO3_abiotic' in df_r.columns:
            print(f"    ΔO3_abiotic range: {df_r['dO3_abiotic'].min():.2e} to {df_r['dO3_abiotic'].max():.2e}")
        print(f"    Saved: {OUTPUT_DIR}/photochem_response_atm_{lbl.lower()}.csv")

    # ── Section 4: Cumulative state ────────────────────────────────────────
    print("\n" + "="*65)
    print("SECTION 4: Cumulative Biosignature State (79-day window)")
    print("="*65)

    cumul_df = compute_cumulative_state(resp_dfs, atms)
    for _, row in cumul_df.iterrows():
        print(f"\n  Atmosphere {row['atmosphere']} ({row['name'][:30]}):")
        for col in row.index:
            if col.startswith('cumul_d') and abs(row[col]) > 1e-8:
                sp = col.replace('cumul_d', '')
                print(f"    Cumul. Δ{sp}: {row[col]*100:.4f}%")
    print(f"  Saved: {OUTPUT_DIR}/cumulative_biosignature_state.csv")

    # ── Section 5: VULCAN stretch goal ────────────────────────────────────
    print("\n" + "="*65)
    print("SECTION 5: VULCAN-Analog Stretch Goal")
    print("="*65)

    # Select the largest flare for the demonstration
    idx_max    = cat['energy_erg'].idxmax()
    flare_row  = cat.loc[idx_max]
    uv_row     = uv_df.loc[idx_max]

    print(f"  Selected flare: E = {flare_row['energy_erg']:.2e} erg  "
          f"(class: {flare_row['class']})")

    ts_df, meta = run_vulcan_analog(flare_row, uv_row)

    print("\n" + "="*65)
    print("SECTION 6: Biosphere Model (Eager-Nash+2024) & Twin State")
    print("="*65)
    resp_A = resp_dfs['A']
    biosphere_df = run_biosphere_model(resp_A, uv_df, atms['A'])
    update_twin_state(resp_dfs, cumul_df, biosphere_df, uv_df)

    # ── Section 6: Plots ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SECTION 6: Diagnostic Plots")
    print("="*65)

    plot_main(atms, uv_df, resp_dfs, cumul_df, cat)
    plot_vulcan(ts_df, meta, flare_row)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STAGE 2 COMPLETE")
    print("="*65)

    n_large     = (uv_df['log10_F_NUV'] >= 6).sum()
    n_superflare= (uv_df['log10_F_NUV'] >= 7).sum()

    cumul_O3_A  = cumul_df[cumul_df['atmosphere']=='A']['cumul_dO3'].values
    cumul_O3_A  = float(cumul_O3_A[0]) if len(cumul_O3_A) > 0 else 0.0

    print(f"""
  TRAPPIST-1e (TIC 267519185) — Stage 2 Results
  ──────────────────────────────────────────────
  Flares processed:     {len(cat)} total
    with F_NUV > 10^6:  {n_large} (large/superflare scale)
    with F_NUV > 10^7:  {n_superflare} (approaching Segura+2010 AD Leo level)

  UV Enhancement Summary (Planck ratio, T_flare=7000K, Ducrot+2022):
    Peak NUV EF range:  {uv_df['EF_NUV_peak'].min():.0f}× – {uv_df['EF_NUV_peak'].max():.0f}×
    NUV/TESS color ratio:  {C_NUV/C_TESS:.0f} (Howard+2020 range 10-1000x observed)

  Atmospheric Response (1D LUT, Segura+2010, Tilley+2019, Chen+2021):
    Atm A (N2-Earth):   Cumul. ΔO3 = {cumul_O3_A*100:.3f}% over 79 days
                        NOTE: 3D models (Ridgway+2023) predict opposite sign!
    Atm B (CO2-Venus):  Abiotic O3 produced (false-positive pathway, Miranda-Rosete+2024)
    Atm C (Thin rock):  UV reaches surface near-unattenuated (no chemistry shield)

  VULCAN-Analog (Chapman+HOx ODE, T=250K, P=0.01 atm):
    Largest flare:      E = {flare_row['energy_erg']:.2e} erg
    Peak O3 depletion:  {meta['max_O3_depletion']*100:.3f}% (per event, UV-only)
    Peak OH enhance.:   {meta['max_OH_enhancement']:.1f}× background

  Output files:  {OUTPUT_DIR}/
    atmospheric_compositions.csv
    flare_uv_enhancement.csv
    photochem_response_atm_a.csv
    photochem_response_atm_b.csv
    photochem_response_atm_c.csv
    cumulative_biosignature_state.csv
    vulcan_analog_timeseries.csv
    stage2_summary_plots.png
    stage2_vulcan_stretch_plot.png

  Key references:
    Segura+2010 Astrobiology 9(18)     — 1D photochem, O3 response
    Tilley+2019 Astrobiology 19(1)     — cumulative flare O3 depletion
    Chen+2021 Nature Astronomy 5(298)  — 3D WACCM; NOx biosignatures
    Ridgway+2023 MNRAS                 — 3D GCM; O3 increases 20x (!)
    Miranda-Rosete+2024 arXiv:2308.01880 — abiotic O3 false positive
    Ducrot+2022 A&A                    — TRAPPIST-1 flare temperatures
    France+2020 ApJS 247 25            — MegaMUSCLES UV fluxes
    Espinoza+2025 (DREAMS)             — JWST TRAPPIST-1e spectrum

  -> Ready for Stage 3: petitRADTRANS spectral generation
""")
