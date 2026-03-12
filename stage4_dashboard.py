"""
===========================================================================
TRAPPIST-1e Digital Twin -- Stage 4: JWST Calibration + Streamlit Dashboard
===========================================================================

Run as: streamlit run stage4_dashboard.py
        python stage4_dashboard.py  (generates static PNG summary)

Panels:
  1. System Overview    -- key metrics from all pipeline stages
  2. Flare Timeline     -- 79-day TESS flare history, UV enhancement
  3. Spectral Comparison -- predicted spectra vs synthetic JWST DREAMS
  4. Biosignature Vulnerability -- feature amplitude vs flare energy
  5. Calibration Residuals -- chi-squared for all 3 atmospheres
  6. Flare Animation    -- step through the flare sequence

JWST DREAMS mock spectrum:
  Based on Espinoza et al. 2025 (DREAMS): spectrum is featureless/flat,
  consistent with no thick atmosphere. NIRSpec/PRISM 0.6-5.3 um.
  Synthesised as flat transit depth + Gaussian noise (~50 ppm/bin) + a
  weak stellar contamination slope (Rackham+2018 model).
  Chi-squared calibration identifies which model atmosphere best fits.

References:
  Espinoza et al. 2025 (DREAMS): JWST TRAPPIST-1e NIRSpec/PRISM
  Rackham et al. 2018 ApJ 853 122: stellar contamination correction
  Lecavelier des Etangs+2008: transmission spectrum formalism (Stage 3)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Detect whether running inside Streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _STREAMLIT = _get_ctx() is not None
except Exception:
    _STREAMLIT = False

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
STAGE1 = "Stage 1 Files"
STAGE2 = "Stage 2 Files"
STAGE3 = "Stage 3 Files"
STAGE4 = "Stage 4 Files"
os.makedirs(STAGE4, exist_ok=True)

# Physical constants (CGS)
DELTA_CONT  = (0.920 * 6.371e8 / (0.1192 * 6.957e10))**2   # (R_p/R_star)^2
TRAPPIST1E  = dict(R_p_Rearth=0.920, M_p_Mearth=0.692, P_d=6.101,
                   a_AU=0.02928, T_eq_K=251, insolation=0.662)


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_all_data():
    """Load all Stage 1-3 outputs into a single dict. Called once and cached."""
    d = {}

    # Stage 1
    d['cat']  = pd.read_csv(f"{STAGE1}/trappist1_flare_catalog.csv")
    d['spec1']= pd.read_csv(f"{STAGE1}/trappist1_stellar_spectrum.csv")

    # Stage 2
    d['uv']   = pd.read_csv(f"{STAGE2}/flare_uv_enhancement.csv")
    d['atm']  = pd.read_csv(f"{STAGE2}/atmospheric_compositions.csv")
    d['resp_A']= pd.read_csv(f"{STAGE2}/photochem_response_atm_a.csv")
    d['resp_B']= pd.read_csv(f"{STAGE2}/photochem_response_atm_b.csv")
    d['cumul'] = pd.read_csv(f"{STAGE2}/cumulative_biosignature_state.csv")
    d['vulcan']= pd.read_csv(f"{STAGE2}/vulcan_analog_timeseries.csv")

    # Stage 3
    d['spectra']= pd.read_csv(f"{STAGE3}/all_spectra_combined.csv")
    d['feats']  = pd.read_csv(f"{STAGE3}/feature_amplitudes.csv")

    # Stage 3 — Biosphere spectra (Eager-Nash+2024); optional (may not exist yet)
    for key, fname in [('bio_on',  'spectrum_atmA_biosphere_on.csv'),
                       ('bio_off', 'spectrum_atmA_biosphere_off.csv'),
                       ('bio_diff','biosphere_spectral_diff.csv')]:
        path = f"{STAGE3}/{fname}"
        d[key] = pd.read_csv(path) if os.path.exists(path) else None

    # Stage 2 — biosphere state (optional)
    bio_path = f"{STAGE2}/biosphere_state.csv"
    d['biosphere_ts'] = pd.read_csv(bio_path) if os.path.exists(bio_path) else None

    return d


# ===========================================================================
# SYNTHETIC JWST DREAMS SPECTRUM
# ===========================================================================
def build_dreams_spectrum(lam_um, n_transits=5, seed=77):
    """
    Synthetic TRAPPIST-1e NIRSpec/PRISM spectrum based on DREAMS programme
    (Espinoza et al. 2025).  Covers 0.6-5.3 um (NIRSpec/PRISM range).

    The published result shows a featureless spectrum consistent with thin/no
    atmosphere.  We model this as:
      depth(lambda) = DELTA_CONT * correction(lambda) + noise
    where correction captures:
      (a) Stellar contamination slope (Rackham+2018): higher at blue end
      (b) TRAPPIST-1b-calibrated offset (GO 6456/9256 strategy)

    Uncertainty model: sigma = sqrt(sigma_phot^2 + sigma_sys^2)
      sigma_phot ~ 120 ppm / sqrt(n_transits * n_avg)  per bin
      sigma_sys  = 20 ppm floor (JWST systematic noise)

    n_transits: number of transit observations (default 5)
    Returns DataFrame: wavelength_um, transit_depth_ppm, uncertainty_ppm
    """
    rng  = np.random.default_rng(seed)
    mask = (lam_um >= 0.6) & (lam_um <= 5.3)
    lam  = lam_um[mask]

    # ── Stellar contamination slope (Rackham+2018; TRAPPIST-1 specific) ──
    # Unocculted star spots (T_spot~2400 K, f_spot~0.5) produce a
    # wavelength-dependent correction that mimics atmospheric spectral slope.
    # For simplicity: linear slope of +0.5 ppm/um from 0.6 to 5.3 um.
    contamination = 0.5 * (lam - 0.6)   # ppm — weak blue slope

    # ── Base transit depth (featureless, consistent with thin atmosphere) ──
    depth_ppm = DELTA_CONT * 1e6 + contamination

    # ── Photon noise model ────────────────────────────────────────────────
    # TRAPPIST-1 J=11.35 mag; NIRSpec/PRISM throughput varies with lambda
    # Roughly 30 ppm per bin per transit at 1 um, rising to 80 ppm at 5 um
    sigma_phot = (30 + 15 * (lam - 0.6)) / np.sqrt(n_transits)
    sigma_sys  = 20.0  # ppm systematic floor
    sigma      = np.sqrt(sigma_phot**2 + sigma_sys**2)

    # ── Noise realisation ─────────────────────────────────────────────────
    noise = rng.normal(0, sigma)
    depth_ppm += noise

    return pd.DataFrame({
        'wavelength_um':     lam,
        'transit_depth_ppm': depth_ppm,
        'uncertainty_ppm':   sigma,
    })


# ===========================================================================
# CHI-SQUARED CALIBRATION
# ===========================================================================
def compute_calibration(spectra_df, jwst_df):
    """
    Compute chi-squared residuals between each predicted spectrum and the
    synthetic JWST DREAMS spectrum.

    chi2_reduced = 1/N * sum( (model - obs)^2 / sigma^2 )
    evaluated over the NIRSpec/PRISM wavelength range (0.6-5.3 um).

    Returns DataFrame with columns:
      atmosphere, scenario, chi2_reduced, rms_ppm, delta_best_ppm
    """
    lam_jwst = jwst_df['wavelength_um'].values
    obs      = jwst_df['transit_depth_ppm'].values
    sigma    = jwst_df['uncertainty_ppm'].values

    rows = []
    for (atm, scen), grp in spectra_df.groupby(['atmosphere', 'scenario']):
        # Interpolate model onto JWST wavelength grid
        grp_sorted = grp.sort_values('wavelength_um')
        model = np.interp(lam_jwst,
                          grp_sorted['wavelength_um'].values,
                          grp_sorted['transit_depth_ppm'].values)

        residual = model - obs
        chi2     = np.mean((residual / sigma)**2)
        rms      = np.sqrt(np.mean(residual**2))

        rows.append({
            'atmosphere':      atm,
            'scenario':        scen,
            'chi2_reduced':    chi2,
            'rms_ppm':         rms,
            'mean_offset_ppm': np.mean(residual),
        })

    df = pd.DataFrame(rows).sort_values('chi2_reduced')
    df.to_csv(f"{STAGE4}/calibration_results.csv", index=False)
    return df


# ===========================================================================
# CUMULATIVE STATE AT TIME T (for animation)
# ===========================================================================
def compute_cumulative_at_day(resp_A, day):
    """
    Accumulate per-flare biosignature changes up to `day` days,
    with exponential recovery (same logic as Stage 2 Section 4).

    Returns dict: {species: cumulative_fractional_change}
    """
    RECOVERY_TAU = {
        'O3': 30.0, 'CH4': 1825.0, 'N2O': 14.0,
        'NO2': 0.5, 'HNO3': 3.0, 'OH': 0.0001,
    }
    species_cols = ['dO3', 'dCH4', 'dN2O', 'dNO2', 'dHNO3']
    ev_sorted = resp_A[resp_A['time_days'] <= day].sort_values('time_days')

    state = {}
    prev_t = 0.0
    for _, ev in ev_sorted.iterrows():
        dt = ev['time_days'] - prev_t
        for col in species_cols:
            sp = col[1:]  # strip 'd'
            tau = RECOVERY_TAU.get(sp, 30.0)
            state[col] = state.get(col, 0.0) * np.exp(-dt / tau)
            state[col] = state.get(col, 0.0) + ev.get(col, 0.0)
        prev_t = ev['time_days']

    # Return as species-keyed dict
    return {col[1:]: state.get(col, 0.0) for col in species_cols}


def spectrum_at_day(spectra_df, resp_A, day, atm='A'):
    """
    Estimate the transmission spectrum at a given day by linearly scaling
    the feature excess above continuum using the cumulative VMR changes.

    spectrum(t) ≈ continuum + (quiescent_excess) * (1 + cumul_dX(t))
    Valid for |dX| << 1 (small perturbations).
    """
    state = compute_cumulative_at_day(resp_A, day)

    q = spectra_df[(spectra_df['atmosphere'] == atm) &
                   (spectra_df['scenario'] == 'quiescent')].copy()
    q = q.sort_values('wavelength_um')

    # Continuum (Rayleigh floor)
    cont_ppm = DELTA_CONT * 1e6
    excess   = q['transit_depth_ppm'].values - cont_ppm

    # Scale: each wavelength bin has contributions from O3, CH4, N2O, etc.
    # Use a simple global scaling by the dominant biosignature (O3)
    # More rigorous: weight by species cross-section contribution, but
    # that requires full Stage 3 RT — this linear approx is sufficient
    # for the visualisation purpose.
    O3_scale  = 1.0 + state.get('O3', 0.0)
    CH4_scale = 1.0 + state.get('CH4', 0.0)
    N2O_scale = 1.0 + state.get('N2O', 0.0)
    NO2_scale = 1.0 + state.get('NO2', 0.0)

    # Weighted average scale (O3 dominates at 9.6um, CH4 at 3.3um, etc.)
    # Use a wavelength-partitioned scale factor:
    lam = q['wavelength_um'].values
    scale = np.ones_like(lam)
    scale[(lam >= 3.0) & (lam <= 4.0)]  = CH4_scale    # CH4 3.3 um
    scale[(lam >= 8.5) & (lam <= 11.0)] = O3_scale     # O3  9.6 um
    scale[(lam >= 4.3) & (lam <= 4.8)]  = N2O_scale    # N2O 4.5 um
    scale[(lam >= 5.5) & (lam <= 7.5)]  = NO2_scale    # NO2/H2O 6 um

    scaled_depth = cont_ppm + excess * scale

    result = q.copy()
    result['transit_depth_ppm'] = scaled_depth
    return result


# ===========================================================================
# STATIC SUMMARY FIGURE (used when not running as Streamlit)
# ===========================================================================
def generate_static_summary(data, jwst_df, calib_df):
    print("  Generating static Stage 4 summary figure...")

    spectra_df = data['spectra']
    cat        = data['cat']
    uv         = data['uv']
    feats      = data['feats']
    resp_A     = data['resp_A']

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        "TRAPPIST-1e Digital Twin -- Stage 4: JWST Calibration & Biosignature Assessment\n"
        "Synthetic JWST DREAMS spectrum (Espinoza+2025) vs. predicted spectra | "
        "Chi-squared calibration | Flare timeline",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.35)

    COLORS = {'A': '#1565C0', 'B': '#B71C1C', 'C': '#546E7A'}
    ATM_NAMES = {'A': 'Atm A: N2-Earth', 'B': 'Atm B: CO2-Venus', 'C': 'Atm C: Thin rock'}

    # ── Panel 1: Flare timeline ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    classes = {'micro': '#90CAF9', 'moderate': '#FB8C00', 'large': '#E53935', 'superflare': '#6A1A9A'}
    for cls, clr in classes.items():
        sub = cat[cat['class'] == cls]
        if len(sub):
            ax1.scatter(sub['time_days'], sub['log10_energy'],
                        c=clr, s=np.clip(sub['log10_energy']*5, 5, 100),
                        alpha=0.8, label=f"{cls} (N={len(sub)})", zorder=3)
    ax1.set_xlabel('Days since start of TESS Sector 10 observation', fontsize=9)
    ax1.set_ylabel('log10(E_flare / erg)', fontsize=9)
    ax1.set_title(f'TRAPPIST-1 Flare Timeline — {len(cat)} flares over {cat.time_days.max():.1f} days\n'
                  'Davenport (2014) template | Ducrot+2022 temperatures | Vida+2017 FFD', fontsize=9)
    ax1.legend(fontsize=7, ncol=4)
    ax1.axhline(32, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.text(cat.time_days.max()*0.98, 32.05, 'large', ha='right', fontsize=7, color='gray')

    # ── Panel 2: Chi-squared calibration ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    calib_q = calib_df[calib_df['scenario'] == 'quiescent'].sort_values('chi2_reduced')
    atm_lbls = [ATM_NAMES.get(a, a) for a in calib_q['atmosphere']]
    chi2_vals = calib_q['chi2_reduced'].values
    colors_c  = [COLORS.get(a, 'gray') for a in calib_q['atmosphere']]
    bars = ax2.barh(atm_lbls, chi2_vals, color=colors_c, alpha=0.8, edgecolor='k', lw=0.7)
    ax2.set_xlabel('Chi-squared (reduced) vs JWST DREAMS', fontsize=9)
    ax2.set_title('JWST Calibration: Model-Data Agreement\n'
                  'Lower = better fit to DREAMS spectrum', fontsize=9)
    for bar_i, val in zip(bars, chi2_vals):
        ax2.text(val + chi2_vals.max()*0.02, bar_i.get_y() + bar_i.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=8)
    best = calib_q.iloc[0]
    ax2.text(0.98, 0.05, f"Best fit: Atm {best['atmosphere']}\nchi2={best['chi2_reduced']:.2f}",
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.8))
    ax2.text(0.02, 0.96,
             'Flat spectrum = no thick atm\n(Espinoza+2025 DREAMS result)',
             transform=ax2.transAxes, ha='left', va='top', fontsize=7.5,
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    # ── Panel 3: Spectral comparison (NIRSpec range) ──────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    jwst_mask = (jwst_df['wavelength_um'] >= 0.6) & (jwst_df['wavelength_um'] <= 5.3)
    ax3.errorbar(jwst_df['wavelength_um'][jwst_mask],
                 jwst_df['transit_depth_ppm'][jwst_mask],
                 yerr=jwst_df['uncertainty_ppm'][jwst_mask],
                 fmt='k.', ms=4, elinewidth=0.8, alpha=0.7, zorder=5,
                 label='Synthetic JWST DREAMS (Espinoza+2025)')
    for atm in ['A', 'B', 'C']:
        q = spectra_df[(spectra_df['atmosphere'] == atm) &
                       (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
        mask = (q['wavelength_um'] >= 0.6) & (q['wavelength_um'] <= 5.3)
        ax3.plot(q['wavelength_um'][mask], q['transit_depth_ppm'][mask],
                 color=COLORS[atm], lw=1.6, alpha=0.85, label=ATM_NAMES[atm])
    ax3.set_xlabel('Wavelength (um)', fontsize=9)
    ax3.set_ylabel('Transit Depth (ppm)', fontsize=9)
    ax3.set_title('Predicted Spectra vs Synthetic JWST DREAMS\n'
                  'NIRSpec/PRISM range (0.6-5.3 um) | R=100 | ~50 ppm per bin uncertainty',
                  fontsize=9)
    ax3.legend(fontsize=7, ncol=2)
    for lam0, name in [(1.38, 'H2O'), (1.87, 'H2O'), (2.3, 'CH4'), (3.3, 'CH4'), (4.3, 'CO2')]:
        ax3.axvline(lam0, color='purple', lw=0.7, ls=':', alpha=0.5)
        ax3.text(lam0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else
                 spectra_df['transit_depth_ppm'].max(),
                 name, ha='center', va='bottom', fontsize=6.5, color='purple', rotation=90)

    # ── Panel 4: Residuals ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    for atm in ['A', 'B', 'C']:
        q = spectra_df[(spectra_df['atmosphere'] == atm) &
                       (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
        lam_j = jwst_df['wavelength_um'].values
        mod   = np.interp(lam_j, q['wavelength_um'].values, q['transit_depth_ppm'].values)
        resid = mod - jwst_df['transit_depth_ppm'].values
        ax4.plot(lam_j, resid, color=COLORS[atm], lw=1.4, alpha=0.85,
                 label=ATM_NAMES[atm])
    ax4.axhline(0, color='k', lw=1, ls='--')
    ax4.fill_between(lam_j, -jwst_df['uncertainty_ppm'].values,
                     jwst_df['uncertainty_ppm'].values, alpha=0.15, color='gray',
                     label='1-sigma JWST noise')
    ax4.set_xlabel('Wavelength (um)', fontsize=9)
    ax4.set_ylabel('Model - DREAMS (ppm)', fontsize=9)
    ax4.set_title('Calibration Residuals\nModel minus JWST DREAMS', fontsize=9)
    ax4.legend(fontsize=7)
    ax4.set_xlim(0.6, 5.3)

    # ── Panel 5: Biosignature vulnerability (O3) ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    resp_A_sorted = resp_A.sort_values('log10_energy')
    ax5.scatter(resp_A_sorted['log10_energy'], resp_A_sorted['dO3'] * 100,
                c=resp_A_sorted['EF_NUV_peak'], cmap='hot_r',
                s=20, alpha=0.75, norm=Normalize(vmin=1, vmax=1000))
    sm = ScalarMappable(cmap='hot_r', norm=Normalize(vmin=1, vmax=1000))
    sm.set_array([])
    plt.colorbar(sm, ax=ax5, label='NUV Enhancement (x)')
    cumO3 = float(data['cumul'][data['cumul']['atmosphere']=='A']['cumul_dO3'].iloc[0])
    ax5.axhline(cumO3 * 100, color='blue', ls='--', lw=1.5,
                label=f'79-day cumul. {cumO3*100:.2f}%')
    ax5.set_xlabel('log10(E_flare / erg)', fontsize=9)
    ax5.set_ylabel('dO3 per event (%)', fontsize=9)
    ax5.set_title('O3 Vulnerability: Depletion per Flare\n'
                  'Atm A (N2-Earth) | 1D Segura+2010 calibration', fontsize=9)
    ax5.legend(fontsize=7)
    ax5.text(0.02, 0.96,
             '[!] 3D (Ridgway+2023)\nO3 increases 20x',
             transform=ax5.transAxes, fontsize=7, va='top', color='red',
             bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))

    # ── Panel 6: Feature amplitudes comparison ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    feat_q = feats[feats['scenario'] == 'quiescent']
    key_feats = ['O3_9.6', 'CH4_3.3', 'CO2_4.3', 'H2O_1.9', 'N2O_4.5']
    x_pos = np.arange(len(key_feats))
    width = 0.25
    for i, atm in enumerate(['A', 'B', 'C']):
        vals = []
        for fn in key_feats:
            sub = feat_q[(feat_q['atmosphere'] == atm) & (feat_q['feature'] == fn)]
            vals.append(float(sub['amplitude_ppm'].iloc[0]) if len(sub) > 0 else 0.0)
        ax6.bar(x_pos + i*width, vals, width, color=COLORS[atm], alpha=0.8,
                edgecolor='k', lw=0.5, label=ATM_NAMES[atm])
    ax6.set_xticks(x_pos + width)
    ax6.set_xticklabels(key_feats, rotation=30, ha='right', fontsize=8)
    ax6.set_ylabel('Feature Amplitude (ppm)', fontsize=9)
    ax6.set_title('Key Biosignature Feature Amplitudes\n'
                  'Quiescent spectra | HITRAN2020 cross-sections', fontsize=9)
    ax6.legend(fontsize=7)
    ax6.axhline(20, color='k', ls=':', lw=0.8, alpha=0.5)
    ax6.text(x_pos[-1]+width, 21, 'JWST\ndetection\nhorizon (~20 ppm)',
             fontsize=6.5, ha='right', va='bottom', color='gray')

    # ── Panel 7: Flare sequence spectrum evolution ──────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    spectra_df_A = spectra_df[(spectra_df['atmosphere'] == 'A') &
                               (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
    lam_all = spectra_df_A['wavelength_um'].values
    nirspec_m = (lam_all >= 0.6) & (lam_all <= 5.3)

    n_steps = 6
    days_steps = np.linspace(0, cat['time_days'].max(), n_steps)
    cmap_steps = plt.cm.Blues(np.linspace(0.3, 1.0, n_steps))

    q_depth = spectra_df_A['transit_depth_ppm'].values
    ax7.plot(lam_all[nirspec_m], q_depth[nirspec_m],
             'k-', lw=1.8, label='Quiescent', zorder=n_steps+2)

    for i, day in enumerate(days_steps[1:], 1):
        stepped = spectrum_at_day(spectra_df, resp_A, day, atm='A')
        stepped = stepped.sort_values('wavelength_um')
        lam_s = stepped['wavelength_um'].values
        dep_s = stepped['transit_depth_ppm'].values
        m_s   = (lam_s >= 0.6) & (lam_s <= 5.3)
        ax7.plot(lam_s[m_s], dep_s[m_s],
                 color=cmap_steps[i], lw=1.2, alpha=0.85,
                 label=f'Day {day:.0f}', zorder=n_steps-i+2)

    ax7.set_xlabel('Wavelength (um)', fontsize=9)
    ax7.set_ylabel('Transit Depth (ppm)', fontsize=9)
    ax7.set_title('Spectrum Evolution Through Flare Sequence\n'
                  'Atm A (N2-Earth) | Cumulative flare effects', fontsize=9)
    ax7.legend(fontsize=6.5, ncol=2)

    path = f"{STAGE4}/stage4_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# DIGITAL TWIN: STATE LOADER + NEW TAB RENDERERS
# ===========================================================================

def load_twin_state_cached():
    """Load twin_state.json with a short TTL so the dashboard stays fresh."""
    try:
        from twin_core import load_twin_state
        return load_twin_state()
    except Exception:
        return {}


def render_twin_state_monitor(state, data):
    """Tab: Twin State Monitor — live atmospheric and biosphere readout."""
    st.subheader("TRAPPIST-1e Digital Twin — Live Atmospheric State")
    st.caption(
        "The twin's living state vector: all atmospheric VMRs and biosphere health "
        "as they stand after the 79-day flare sequence. Updated each time Stage 2 runs."
    )

    bio = state.get('biosphere', {})
    atm = state.get('atmosphere', {}).get('A', {})
    atm_init = state.get('atmosphere_initial', {}).get('A', {})
    post_wts = state.get('posterior_weights', {'A': 1/3, 'B': 1/3, 'C': 1/3})
    meta = state.get('meta', {})

    # ── Top metric row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pf    = bio.get('population_factor', 1.0)
    pf_pct = f"{pf*100:.1f}%"
    bio_color = "normal" if pf > 0.5 else ("off" if pf < 0.2 else "inverse")
    c1.metric("Twin last updated", meta.get('last_updated_by', 'stage1').upper(),
              delta=meta.get('last_updated_utc', '')[:10])
    c2.metric("Biosphere health",  pf_pct,
              delta="STRESSED" if pf < 0.5 else "HEALTHY")
    c3.metric("O3 (Atm A)", f"{atm.get('O3', 3e-7):.2e}",
              delta=f"{(atm.get('O3',3e-7)/atm_init.get('O3',3e-7)-1)*100:+.1f}%" if atm_init.get('O3') else None)
    c4.metric("CH4 (Atm A)", f"{atm.get('CH4', 1.8e-6):.2e}",
              delta=f"{(atm.get('CH4',1.8e-6)/atm_init.get('CH4',1.8e-6)-1)*100:+.1f}%" if atm_init.get('CH4') else None)
    c5.metric("CO biotic (Atm A)", f"{bio.get('co_vmr_biotic', 3e-9):.2e}",
              delta=f"abiotic: {bio.get('co_vmr_abiotic', 1e-7):.1e}")
    c6.metric("Best-fit posterior", f"Atm {max(post_wts, key=post_wts.get)}",
              delta=f"p={max(post_wts.values()):.2f}")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**State Vector Timeline** — 79-day evolution of Atm A species")
        hist = state.get('history', [])
        if hist:
            h_df = pd.DataFrame(hist)
            fig_h, axes_h = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            species_panels = [
                (['O3', 'N2O'], ['#1565C0', '#7B1FA2'], 'Ozone & N2O (VMR)'),
                (['CH4'],       ['#2E7D32'],              'Methane (VMR)'),
                (['population_factor'], ['#E65100'],      'Biosphere Population Factor'),
            ]
            for ax, (cols, clrs, title) in zip(axes_h, species_panels):
                for col, clr in zip(cols, clrs):
                    if col in h_df.columns:
                        ax.plot(h_df['day'], h_df[col], color=clr, lw=1.8, label=col)
                ax.set_ylabel(title, fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2)

            axes_h[-1].set_xlabel('Days since start of observation', fontsize=9)
            axes_h[0].set_title('Twin State Timeline — Atmospheric Evolution', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_h, use_container_width=True)
            plt.close()
        else:
            st.info("Run Stage 2 to populate the twin state timeline.")

    with col2:
        st.markdown("**Biosphere Stress Log**")
        stress_events = bio.get('stress_events', [])
        if stress_events:
            se_df = pd.DataFrame(stress_events)
            se_df.columns = ['Day', 'EF_NUV', 'UV Dose (erg/cm2)', 'Pop. Factor']
            st.dataframe(se_df.style.format({'EF_NUV': '{:.0f}',
                                             'UV Dose (erg/cm2)': '{:.2e}',
                                             'Pop. Factor': '{:.3f}'}),
                         use_container_width=True)
        else:
            st.info("No high-stress flare events recorded yet.")

        st.markdown("**Biosphere health**")
        st.progress(float(pf))
        st.caption(f"Population factor: {pf:.3f}  |  Active: {bio.get('active', True)}")

        st.markdown("**JWST Posterior Weights**")
        fig_pw, ax_pw = plt.subplots(figsize=(4, 2))
        atm_labels = ['A: N2-Earth', 'B: CO2-Venus', 'C: Thin rock']
        atm_keys   = ['A', 'B', 'C']
        pw_vals    = [post_wts.get(k, 1/3) for k in atm_keys]
        ax_pw.barh(atm_labels, pw_vals, color=['#1565C0', '#B71C1C', '#546E7A'], alpha=0.8)
        ax_pw.set_xlim(0, 1)
        ax_pw.set_xlabel('Posterior probability', fontsize=8)
        ax_pw.axvline(1/3, color='gray', ls='--', lw=0.8, alpha=0.6)
        for i, v in enumerate(pw_vals):
            ax_pw.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_pw, use_container_width=True)
        plt.close()

        obs_log = state.get('observations', [])
        if obs_log:
            st.markdown(f"**{len(obs_log)} JWST observation(s) assimilated**")
            last = obs_log[-1]
            st.caption(f"Last: {last.get('obs_id','?')} — best fit Atm {last.get('best_fit_atm','?')}, "
                       f"chi2={last.get('chi2_best', 0):.2f}")


def render_biosphere_engine(state, data):
    """Tab: Biosphere Engine — CO/CH4 feedback and detectability."""
    st.subheader("Biosphere Engine — Eager-Nash et al. 2024")
    st.caption(
        "The biosphere modulates two key spectral features: "
        "CO (suppressed by consumption: 4CO + 2H2O → 2CO2 + CH3COOH) and "
        "CH4 (enhanced by methanogenesis: 4H2 + CO2 → 2H2O + CH4). "
        "On M-dwarf planets, abiotic CO from UV photolysis dominates the spectrum — "
        "a living biosphere is the only known mechanism to suppress it."
    )

    bio   = state.get('biosphere', {})
    col1, col2 = st.columns([3, 2])

    with col1:
        bio_on  = data.get('bio_on')
        bio_off = data.get('bio_off')
        bio_diff= data.get('bio_diff')

        if bio_on is not None and bio_off is not None:
            fig_b, axes_b = plt.subplots(2, 1, figsize=(12, 8),
                                          gridspec_kw={'height_ratios': [3, 1]})
            # Spectral comparison zoomed to CO + CH4 region
            ax = axes_b[0]
            m = (bio_on['wavelength_um'] >= 2.0) & (bio_on['wavelength_um'] <= 6.0)
            ax.plot(bio_on['wavelength_um'][m], bio_on['transit_depth_ppm'][m],
                    color='#2E7D32', lw=2.0, label='Biosphere ON (life present)')
            ax.plot(bio_off['wavelength_um'][m], bio_off['transit_depth_ppm'][m],
                    color='#B71C1C', lw=2.0, ls='--', label='Biosphere OFF (abiotic only)')
            ax.fill_between(bio_on['wavelength_um'][m],
                            bio_on['transit_depth_ppm'][m],
                            bio_off['transit_depth_ppm'][m],
                            alpha=0.18, color='green', label='Biosignature region')
            for lam0, name, clr in [(3.3, 'CH4', '#1565C0'), (4.67, 'CO', '#E65100'),
                                    (4.3, 'CO2', '#795548')]:
                ax.axvline(lam0, color=clr, lw=0.9, ls=':', alpha=0.7)
                ax.text(lam0, ax.get_ylim()[1] if ax.get_ylim()[1] != 0
                        else bio_on['transit_depth_ppm'].max(),
                        name, ha='center', fontsize=8, color=clr, va='top')
            ax.set_ylabel('Transit Depth (ppm)', fontsize=10)
            ax.set_title('Biosphere Spectral Fingerprint (Atm A, 2-6 um)\n'
                         'CO suppression + CH4 enhancement = sign of life', fontsize=10)
            ax.legend(fontsize=8)

            # Spectral difference
            ax_d = axes_b[1]
            if bio_diff is not None:
                m2 = (bio_diff['wavelength_um'] >= 2.0) & (bio_diff['wavelength_um'] <= 6.0)
                diff = bio_diff['diff_ppm'][m2]
                ax_d.bar(bio_diff['wavelength_um'][m2], diff,
                         color=np.where(diff > 0, '#2E7D32', '#B71C1C'),
                         width=0.05, alpha=0.8)
            ax_d.axhline(0, color='k', lw=1)
            ax_d.axhline(20, color='gray', ls='--', lw=0.8, label='JWST ~5-sigma (20 ppm)')
            ax_d.axhline(-20, color='gray', ls='--', lw=0.8)
            ax_d.set_xlabel('Wavelength (um)', fontsize=9)
            ax_d.set_ylabel('Biosphere-ON minus -OFF (ppm)', fontsize=8)
            ax_d.legend(fontsize=7)

            plt.tight_layout()
            st.pyplot(fig_b, use_container_width=True)
            plt.close()
        else:
            st.info("Run Stage 3 to generate biosphere spectra (spectrum_atmA_biosphere_on/off.csv).")

    with col2:
        st.markdown("**Current biosphere state**")
        pf = bio.get('population_factor', 1.0)
        st.progress(float(pf))
        st.metric("Population factor", f"{pf:.3f}", delta="ACTIVE" if bio.get('active', True) else "EXTINCT")
        st.metric("CO abiotic VMR",  f"{bio.get('co_vmr_abiotic', 1e-7):.2e}")
        st.metric("CO biotic VMR",   f"{bio.get('co_vmr_biotic', 3e-9):.2e}",
                  delta=f"x{bio.get('co_vmr_abiotic',1e-7)/max(bio.get('co_vmr_biotic',3e-9),1e-30):.0f} suppression")
        st.metric("CH4 production rate", f"{bio.get('ch4_production_rate', 1.0):.2f}x")

        with st.expander("Eager-Nash+2024 reaction pathways"):
            st.markdown(
                "**Methanogenesis** (H2/CO-consuming archaea):  \n"
                "`4H2 + CO2 → 2H2O + CH4`  \n\n"
                "**CO consumption**:  \n"
                "`4CO + 2H2O → 2CO2 + CH3COOH`  \n\n"
                "**Abiotic O2 pathway** (M-dwarf specific):  \n"
                "`CO2 + hν → CO + O`  \n"
                "`CO2 photolysis is 5-10x stronger on M-dwarfs (higher FUV/NUV ratio)`  \n\n"
                "**Key finding (Eager-Nash+2024):** CO is the dominant spectral feature "
                "on TRAPPIST-1e because M-dwarf UV drives much more CO2 photolysis than "
                "the Sun. A biosphere consuming CO suppresses this feature by ~30x, "
                "creating a detectable ~1-2 ppm spectral change at CO 4.67 um."
            )
        st.warning(
            "**Abiotic O2 false-positive (Eager-Nash+2024):**  \n"
            "Enhanced CO2 photolysis produces 5-10x more abiotic O2 on M-dwarf planets "
            "than on Earth-like orbits. O3 detection alone cannot confirm biology — "
            "CO must also be assessed to rule out abiotic pathways."
        )


def render_forward_prediction(state):
    """Tab: Forward Prediction — Monte Carlo future state projection."""
    st.subheader("Forward Prediction Engine")
    st.caption(
        "Project the twin's atmospheric and biosphere state forward in time. "
        "Uses Monte Carlo future flare sampling (Vida+2017 FFD) + photochem LUT + "
        "biosphere feedback to forecast what TRAPPIST-1e's atmosphere will look like "
        "in 30, 90, or 365 days."
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        horizon = cc1.selectbox("Forecast horizon", [30, 90, 365], index=1,
                                 help="Days to project forward from current state")
    with cc2:
        rate_label = cc2.selectbox(
            "Stellar activity", ["Nominal (0.53/day)", "High (1.06/day)", "Quiet (0.27/day)"])
        rate_mult = {"Nominal (0.53/day)": 1.0, "High (1.06/day)": 2.0, "Quiet (0.27/day)": 0.5}[rate_label]
    with cc3:
        run_pred = cc3.button("Run Forward Prediction", type="primary")

    pred = state.get('predictions', {})

    if run_pred:
        with st.spinner(f"Monte Carlo projection: {horizon} days..."):
            try:
                from twin_core import forward_predict, save_twin_state
                pred = forward_predict(state, horizon_days=horizon,
                                       flare_rate_multiplier=rate_mult)
                save_twin_state(state)
                st.success(f"Projection complete: {horizon}-day forecast saved to twin state.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if not pred or 'days' not in pred:
        st.info("Click 'Run Forward Prediction' to project the twin state forward.")
        return

    days  = np.array(pred['days'])
    atm_p = pred.get('atm_A', {})
    bio_p = pred.get('biosphere', {})

    fig_p = plt.figure(figsize=(16, 12))
    gs    = plt.GridSpec(3, 2, hspace=0.42, wspace=0.30)

    # Row 1: O3 and CH4
    ax_o3 = fig_p.add_subplot(gs[0, 0])
    if 'O3' in atm_p:
        o3 = np.array(atm_p['O3'])
        ax_o3.plot(days, o3 * 1e7, color='#1565C0', lw=2, label='O3 (VMR x1e7)')
        ax_o3.fill_between(days, o3*1e7*0.85, o3*1e7*1.15,
                           alpha=0.2, color='#1565C0', label='±15% uncertainty')
    ax_o3.axhline(3.0, color='gray', ls='--', lw=0.8, label='Initial (3.0e-7)')
    ax_o3.set_ylabel('O3 VMR (x1e-7)', fontsize=9)
    ax_o3.set_title('Ozone Forecast', fontsize=9)
    ax_o3.legend(fontsize=7); ax_o3.grid(alpha=0.2)

    ax_ch4 = fig_p.add_subplot(gs[0, 1])
    if 'CH4' in atm_p:
        ch4 = np.array(atm_p['CH4'])
        ax_ch4.plot(days, ch4 * 1e6, color='#2E7D32', lw=2, label='CH4 (VMR x1e6)')
        ax_ch4.fill_between(days, ch4*1e6*0.9, ch4*1e6*1.1,
                            alpha=0.2, color='#2E7D32', label='±10% uncertainty')
    ax_ch4.axhline(1.8, color='gray', ls='--', lw=0.8, label='Initial (1.8e-6)')
    ax_ch4.set_ylabel('CH4 VMR (x1e-6)', fontsize=9)
    ax_ch4.set_title('Methane Forecast', fontsize=9)
    ax_ch4.legend(fontsize=7); ax_ch4.grid(alpha=0.2)

    # Row 2: CO (key Eager-Nash result)
    ax_co = fig_p.add_subplot(gs[1, 0])
    if 'co_vmr' in bio_p:
        co = np.array(bio_p['co_vmr'])
        ax_co.plot(days, co * 1e9, color='#E65100', lw=2, label='CO biotic (VMR x1e9)')
    if 'CO' in atm_p:
        co_abiotic = np.array(atm_p['CO'])
        ax_co.plot(days, co_abiotic * 1e9, color='#FF8F00', lw=1.5, ls='--',
                   label='CO abiotic (no biosphere)')
    ax_co.set_ylabel('CO VMR (x1e-9)', fontsize=9)
    ax_co.set_title('CO: Biotic vs Abiotic\n(Eager-Nash+2024 — key biosignature)', fontsize=9)
    ax_co.legend(fontsize=7); ax_co.grid(alpha=0.2)

    # Row 2: Biosphere population
    ax_pf = fig_p.add_subplot(gs[1, 1])
    if 'population_factor' in bio_p:
        pf_arr = np.array(bio_p['population_factor'])
        colors = ['#2E7D32' if v > 0.5 else '#F57F17' if v > 0.2 else '#B71C1C'
                  for v in pf_arr]
        ax_pf.scatter(days, pf_arr, c=colors, s=8, alpha=0.6)
        ax_pf.plot(days, pf_arr, color='gray', lw=0.5, alpha=0.4)
        ax_pf.axhline(0.5, color='#F57F17', ls='--', lw=0.8, label='Stress threshold')
        ax_pf.axhline(0.2, color='#B71C1C', ls='--', lw=0.8, label='Extinction risk')
    ax_pf.set_ylim(0, 1.05)
    ax_pf.set_ylabel('Population factor', fontsize=9)
    ax_pf.set_title('Biosphere Health Forecast', fontsize=9)
    ax_pf.legend(fontsize=7); ax_pf.grid(alpha=0.2)

    # Row 3: Detection horizon
    ax_det = fig_p.add_subplot(gs[2, :])
    if 'O3' in atm_p and 'CH4' in atm_p:
        o3_amp_init  = 12.7   # ppm (from Stage 3)
        ch4_amp_init = 33.4
        o3_arr  = np.array(atm_p['O3'])
        ch4_arr = np.array(atm_p['CH4'])
        o3_init = 3.0e-7
        ch4_init= 1.8e-6
        o3_amps  = o3_amp_init  * (o3_arr  / o3_init)
        ch4_amps = ch4_amp_init * (ch4_arr / ch4_init)
        trans_o3  = np.clip((5.0 * 50.0 / np.maximum(o3_amps,  0.1))**2, 1, 5000)
        trans_ch4 = np.clip((5.0 * 50.0 / np.maximum(ch4_amps, 0.1))**2, 1, 5000)
        ax_det.plot(days, trans_o3,  color='#1565C0', lw=2, label='Transits for O3 5-sigma')
        ax_det.plot(days, trans_ch4, color='#2E7D32', lw=2, label='Transits for CH4 5-sigma')
        ax_det.axhline(5, color='red', ls='--', lw=1.2, label='DREAMS programme depth (5 transits)')
    ax_det.set_ylabel('Transits needed for 5-sigma detection', fontsize=9)
    ax_det.set_xlabel(f'Days from now (horizon: {horizon} days, {rate_label})', fontsize=9)
    ax_det.set_title('JWST Detection Horizon Forecast — When Will Biosignatures Become Observable?',
                     fontsize=10)
    ax_det.set_ylim(0, 200)
    ax_det.legend(fontsize=8); ax_det.grid(alpha=0.2)

    plt.suptitle(f"TRAPPIST-1e Digital Twin: {horizon}-day Forward Prediction | {rate_label}",
                 fontsize=12, fontweight='bold')
    st.pyplot(fig_p, use_container_width=True)
    plt.close()

    # Download button
    if 'days' in pred:
        export_rows = []
        for i, day in enumerate(pred['days']):
            row = {'day': day}
            for sp in ['O3', 'CH4', 'CO', 'N2O', 'NO2']:
                if sp in atm_p:
                    row[f'atm_A_{sp}'] = atm_p[sp][i]
            if 'population_factor' in bio_p:
                row['biosphere_pf'] = bio_p['population_factor'][i]
            export_rows.append(row)
        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "Download forecast CSV",
            export_df.to_csv(index=False).encode(),
            file_name=f"trappist1e_forecast_{horizon}d.csv",
            mime="text/csv",
        )


# ===========================================================================
# STREAMLIT DASHBOARD
# ===========================================================================
def run_streamlit_dashboard():
    st.set_page_config(
        page_title="TRAPPIST-1e Digital Twin",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Title ───────────────────────────────────────────────────────────────
    st.title("TRAPPIST-1e Digital Twin")
    st.markdown(
        "**A dynamic virtual replica of TRAPPIST-1e's atmosphere and biosphere** — "
        "continuously updated as stellar flares accumulate, JWST observations arrive, "
        "and the twin's biosphere responds.  \n"
        "Pipeline: Stage 1 (Ingest) → Stage 2 (Atmosphere + Biosphere) → "
        "Stage 3 (Spectra) → Stage 4 (Calibration + Prediction)  \n"
        "*Espinoza+2025 DREAMS | Eager-Nash+2024 | Ducrot+2022 | Segura+2010 | "
        "Chen+2021 | Ridgway+2023*"
    )

    # ── Load data (cached) ──────────────────────────────────────────────────
    @st.cache_data
    def _load():
        return load_all_data()

    data = _load()

    # ── Load digital twin state (short TTL so dashboard stays fresh) ─────────
    @st.cache_data(ttl=30)
    def _twin():
        return load_twin_state_cached()

    twin_state = _twin()
    cat  = data['cat']
    uv   = data['uv']
    spectra_df = data['spectra']
    feats = data['feats']
    resp_A = data['resp_A']
    cumul = data['cumul']

    # Build JWST spectrum
    @st.cache_data
    def _jwst(n_transits):
        lam_all = spectra_df['wavelength_um'].unique()
        lam_all.sort()
        return build_dreams_spectrum(lam_all, n_transits=n_transits)

    # Calibration
    @st.cache_data
    def _calib(n_transits):
        jwst = _jwst(n_transits)
        return compute_calibration(spectra_df, jwst), jwst

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        atm_choice = st.radio(
            "Atmospheric composition",
            options=['A', 'B', 'C'],
            format_func=lambda x: {
                'A': 'A: N2-Earth (biosignatures)',
                'B': 'B: CO2-Venus (false-positive O3)',
                'C': 'C: Thin remnant (no chemistry)',
            }[x],
        )
        n_transits = st.slider("JWST transits (noise level)", 1, 20, 5, 1)

        flare_classes = st.multiselect(
            "Show flare classes",
            ['micro', 'moderate', 'large', 'superflare'],
            default=['micro', 'moderate', 'large', 'superflare'],
        )
        e_min = st.slider("Min flare log10(E/erg)", 30.0, 33.0, 30.0, 0.1)
        timeline_day = st.slider(
            "Flare sequence: day (animation)",
            0.0, float(cat['time_days'].max()), 0.0, 0.5,
        )

        st.markdown("---")
        st.markdown(
            "**Data sources**  \n"
            "- TESS TIC 267519185 (TRAPPIST-1)  \n"
            "- Vida+2017 FFD statistics  \n"
            "- Ducrot+2022 flare temperatures  \n"
            "- DREAMS NIRSpec/PRISM (Espinoza+2025)  \n"
            "- HITRAN2020 cross-sections  \n"
        )
        st.markdown(
            "**Limitations (1D model)**  \n"
            "- Ridgway+2023: 3D GCM predicts O3 *increases* 20x  \n"
            "- SEPs not modelled (Segura+2010: dominate depletion)  \n"
            "- No magnetic field shielding  \n"
        )

        st.markdown("---")
        st.markdown("**JWST Data Assimilation**")
        st.caption("Upload a real or synthetic JWST spectrum to update the twin's posterior weights.")
        uploaded = st.file_uploader("Upload JWST spectrum (CSV)", type=["csv"],
                                     help="Columns: wavelength_um, transit_depth_ppm, [uncertainty_ppm]")
        if uploaded is not None:
            try:
                jwst_up = pd.read_csv(uploaded)
                if 'wavelength_um' in jwst_up.columns and 'transit_depth_ppm' in jwst_up.columns:
                    if 'uncertainty_ppm' not in jwst_up.columns:
                        jwst_up['uncertainty_ppm'] = 40.0
                    lam_j = jwst_up['wavelength_um'].values
                    obs_p = jwst_up['transit_depth_ppm'].values
                    sig_p = jwst_up['uncertainty_ppm'].values
                    # Build spectra dict for assimilation
                    spectra_dict = {}
                    for (atm_l, scen), grp in data['spectra'].groupby(['atmosphere', 'scenario']):
                        g = grp.sort_values('wavelength_um')
                        spectra_dict[(atm_l, scen)] = np.interp(
                            lam_j, g['wavelength_um'].values, g['transit_depth_ppm'].values)
                    from twin_core import assimilate_observation, save_twin_state as _save
                    assimilate_observation(twin_state, lam_j, obs_p, sig_p,
                                           spectra_dict, obs_id=uploaded.name)
                    _save(twin_state)
                    st.success(f"Assimilated! Best fit: Atm {twin_state['posterior_weights'] and max(twin_state['posterior_weights'], key=twin_state['posterior_weights'].get)}")
                    st.rerun()
                else:
                    st.error("CSV must have columns: wavelength_um, transit_depth_ppm")
            except Exception as e:
                st.error(f"Assimilation error: {e}")

    calib_df, jwst_df = _calib(n_transits)

    # ── Metric cards ────────────────────────────────────────────────────────
    cumul_A = cumul[cumul['atmosphere'] == 'A'].iloc[0]
    best    = calib_df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flares (79-day window)", len(cat),
              delta=f"{(cat['class']=='superflare').sum()} superflares")
    m2.metric("Cumul. O3 change (Atm A)",
              f"{float(cumul_A['cumul_dO3'])*100:.2f}%",
              delta="1D model (Ridgway+2023: +2000%)")
    m3.metric("Best-fit atmosphere (chi2)",
              f"Atm {best['atmosphere']}",
              delta=f"chi2_red = {best['chi2_reduced']:.2f}")
    m4.metric("O3 feature (9.6 um, Atm A)",
              "12.7 ppm",
              delta="JWST detection horizon ~20 ppm")

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Twin State Monitor",
        "Flare Timeline",
        "Spectral Comparison",
        "Biosignature Vulnerability",
        "Calibration Residuals",
        "Chi-squared Grid",
        "Flare Animation",
        "Biosphere Engine",
        "Forward Prediction",
    ])

    # ========================================================================
    # TAB 1: FLARE TIMELINE
    # ========================================================================
    with tab1:
        st.subheader("TRAPPIST-1 Flare History — TESS Sector 10 (79 days)")
        st.caption(
            "Flare catalog from Vida+2017 FFD statistics applied to 1800s-cadence TESS data. "
            "Energies follow power-law dN/dE ~ E^-1.72. Temperatures 5300-8600 K (Ducrot+2022)."
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Flare timeline scatter
            cat_f = cat[cat['class'].isin(flare_classes) &
                        (cat['log10_energy'] >= e_min)]

            fig1, ax = plt.subplots(figsize=(12, 4))
            CLSS = {'micro': '#90CAF9', 'moderate': '#FB8C00',
                    'large': '#E53935', 'superflare': '#6A1A9A'}
            for cls, clr in CLSS.items():
                if cls not in flare_classes:
                    continue
                sub = cat_f[cat_f['class'] == cls]
                if len(sub):
                    ax.scatter(sub['time_days'], sub['log10_energy'],
                               c=clr, s=np.clip(sub['log10_energy']*5, 5, 120),
                               alpha=0.85, label=f"{cls} (N={len(sub)})", zorder=3)
            ax.axhline(32, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.axhline(33, color='purple', ls='--', lw=0.8, alpha=0.5)
            if timeline_day > 0:
                ax.axvline(timeline_day, color='red', lw=2, ls='-', alpha=0.7,
                           label=f'Current day: {timeline_day:.0f}')
            ax.set_xlabel('Days', fontsize=10)
            ax.set_ylabel('log10(E / erg)', fontsize=10)
            ax.set_title(f'Flare Timeline | {len(cat_f)} flares shown', fontsize=10)
            ax.legend(fontsize=8, ncol=5)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close()

        with col2:
            # UV enhancement histogram
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bins = np.linspace(2, 8, 25)
            ax2.hist(uv['log10_F_NUV'], bins=bins, color='mediumpurple',
                     ec='indigo', alpha=0.8)
            ax2.axvline(5, color='green', ls='--', lw=1.5,
                        label='Segura+2010 threshold')
            ax2.axvline(6.5, color='red', ls='--', lw=1.5,
                        label='AD Leo 1985 event')
            ax2.set_xlabel('log10(F_NUV / erg/cm^2)', fontsize=9)
            ax2.set_ylabel('Flares', fontsize=9)
            ax2.set_title('NUV Fluence at TRAPPIST-1e', fontsize=9)
            ax2.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Flare summary table
        st.markdown("**Top 10 strongest flares**")
        top10 = cat.nlargest(10, 'energy_erg')[
            ['time_days', 'energy_erg', 'log10_energy', 'duration_min',
             'temperature_K', 'class']
        ].copy()
        top10['energy_erg'] = top10['energy_erg'].map('{:.2e}'.format)
        top10['duration_min'] = top10['duration_min'].round(1)
        top10['temperature_K'] = top10['temperature_K'].round(0).astype(int)
        st.dataframe(top10, use_container_width=True)

    # ========================================================================
    # TAB 2: SPECTRAL COMPARISON
    # ========================================================================
    with tab2:
        st.subheader("Predicted vs JWST DREAMS Transmission Spectrum")
        st.caption(
            "Synthetic DREAMS spectrum (Espinoza+2025): featureless, consistent with thin/no atmosphere. "
            "NIRSpec/PRISM 0.6-5.3 um, R=100. "
            f"Noise model: {n_transits} transit(s), ~20-80 ppm/bin."
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

            # Main spectral comparison
            ax = axes3[0]
            jwst_m = (jwst_df['wavelength_um'] >= 0.6) & (jwst_df['wavelength_um'] <= 5.3)
            ax.errorbar(jwst_df['wavelength_um'][jwst_m],
                        jwst_df['transit_depth_ppm'][jwst_m],
                        yerr=jwst_df['uncertainty_ppm'][jwst_m],
                        fmt='k.', ms=4, elinewidth=0.8, alpha=0.7, zorder=6,
                        label='Synthetic JWST DREAMS (Espinoza+2025)')

            COLORS = {'A': '#1565C0', 'B': '#B71C1C', 'C': '#546E7A'}
            for atm_l in ['A', 'B', 'C']:
                q = spectra_df[(spectra_df['atmosphere'] == atm_l) &
                               (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
                m = (q['wavelength_um'] >= 0.6) & (q['wavelength_um'] <= 5.3)
                lw = 2.0 if atm_l == atm_choice else 1.2
                alpha = 0.95 if atm_l == atm_choice else 0.55
                label_str = {
                    'A': 'Atm A: N2-Earth',
                    'B': 'Atm B: CO2-Venus',
                    'C': 'Atm C: Thin rock',
                }[atm_l]
                ax.plot(q['wavelength_um'][m], q['transit_depth_ppm'][m],
                        color=COLORS[atm_l], lw=lw, alpha=alpha, label=label_str)

                # Post-flare (selected atmosphere only)
                if atm_l == atm_choice:
                    pf = spectra_df[(spectra_df['atmosphere'] == atm_l) &
                                    (spectra_df['scenario'] == 'postflare_cumul')].sort_values('wavelength_um')
                    m2 = (pf['wavelength_um'] >= 0.6) & (pf['wavelength_um'] <= 5.3)
                    ax.plot(pf['wavelength_um'][m2], pf['transit_depth_ppm'][m2],
                            color=COLORS[atm_l], lw=1.2, ls='--', alpha=0.7,
                            label=f'Atm {atm_l} post-79d flares')

            # Feature annotations
            for lam0, name in [(1.38, 'H2O'), (1.87, 'H2O'),
                               (2.3, 'CH4'), (3.3, 'CH4'), (4.3, 'CO2'), (4.5, 'N2O')]:
                ax.axvline(lam0, color='purple', lw=0.7, ls=':', alpha=0.5)
                ax.text(lam0, ax.get_ylim()[1] if ax.get_ylim()[1] != 0
                        else spectra_df['transit_depth_ppm'].max(),
                        name, ha='center', fontsize=7, color='purple', rotation=90, va='top')

            ax.set_ylabel('Transit Depth (ppm)', fontsize=10)
            ax.set_title('NIRSpec/PRISM Spectral Comparison', fontsize=10)
            ax.legend(fontsize=8, ncol=3)
            ax.axvspan(0.6, 5.3, alpha=0.04, color='gold')

            # Residuals subplot
            ax_r = axes3[1]
            for atm_l in ['A', 'B', 'C']:
                q = spectra_df[(spectra_df['atmosphere'] == atm_l) &
                               (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
                lam_j = jwst_df['wavelength_um'].values
                mod   = np.interp(lam_j, q['wavelength_um'].values,
                                  q['transit_depth_ppm'].values)
                resid = mod - jwst_df['transit_depth_ppm'].values
                ax_r.plot(lam_j, resid, color=COLORS[atm_l], lw=1.3, alpha=0.8)

            ax_r.axhline(0, color='k', lw=1, ls='--')
            ax_r.fill_between(jwst_df['wavelength_um'].values,
                              -jwst_df['uncertainty_ppm'].values,
                              jwst_df['uncertainty_ppm'].values,
                              alpha=0.15, color='gray', label='1-sigma noise')
            ax_r.set_xlabel('Wavelength (um)', fontsize=10)
            ax_r.set_ylabel('Residual (ppm)', fontsize=10)
            ax_r.set_xlim(0.6, 5.3)
            ax_r.legend(fontsize=7)

            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("**Selected atmosphere**")
            atm_info = data['atm'][data['atm']['atmosphere'] == atm_choice].iloc[0]
            st.metric("Name", atm_info['name'][:20])
            st.metric("P (bar)", f"{atm_info['P_bar']:.1f}")
            st.metric("T (K)", f"{atm_info['T_K']:.0f}")
            st.metric("JWST status", atm_info['jwst_status'][:18])

            st.markdown("---")
            st.markdown("**Chi2 summary**")
            calib_q = calib_df[calib_df['scenario'] == 'quiescent']
            for _, row in calib_q.iterrows():
                best_flag = " (best)" if row['atmosphere'] == calib_df.iloc[0]['atmosphere'] else ""
                st.metric(
                    f"Atm {row['atmosphere']}{best_flag}",
                    f"chi2 = {row['chi2_reduced']:.2f}",
                    delta=f"RMS = {row['rms_ppm']:.1f} ppm"
                )

            st.markdown("---")
            st.caption(
                "Chi2 < 1.5: acceptable fit  \n"
                "Chi2 > 3.0: tension with DREAMS  \n"
                "Flat spectrum preferred = featureless/thin atmosphere  \n"
                "consistent with Espinoza+2025 result."
            )

    # ========================================================================
    # TAB 3: BIOSIGNATURE VULNERABILITY
    # ========================================================================
    with tab3:
        st.subheader("Biosignature Vulnerability Assessment")
        st.caption(
            "How flare activity modifies key biosignature mixing ratios. "
            "1D model using Segura+2010 / Tilley+2019 / Chen+2021 lookup tables. "
            "Ridgway+2023 3D GCM finds the OPPOSITE for O3 — unresolved tension."
        )

        col1, col2 = st.columns([3, 2])

        with col1:
            # Per-flare dO3 and dCH4 vs energy
            fig4, axes4 = plt.subplots(2, 2, figsize=(11, 7))
            resp_a_f = resp_A[resp_A['log10_energy'] >= e_min]

            # O3
            ax = axes4[0, 0]
            sc = ax.scatter(resp_a_f['log10_energy'], resp_a_f['dO3'] * 100,
                            c=resp_a_f['EF_NUV_peak'].clip(1, 5000),
                            cmap='Reds', s=25, alpha=0.75,
                            norm=plt.Normalize(1, 5000))
            plt.colorbar(sc, ax=ax, label='NUV Enhancement (x)')
            ax.set_xlabel('log10(E / erg)', fontsize=9)
            ax.set_ylabel('dO3 per event (%)', fontsize=9)
            ax.set_title('O3 Depletion per Flare\n(Segura+2010; 1D)', fontsize=9)
            ax.text(0.02, 0.96, '[!] Ridgway+2023 3D: O3 increases 20x',
                    transform=ax.transAxes, fontsize=7, va='top', color='red',
                    bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.9))

            # CH4
            ax = axes4[0, 1]
            ax.scatter(resp_a_f['log10_energy'], resp_a_f['dCH4'] * 100,
                       c=resp_a_f['EF_NUV_peak'].clip(1, 5000),
                       cmap='Greens', s=25, alpha=0.75,
                       norm=plt.Normalize(1, 5000))
            ax.set_xlabel('log10(E / erg)', fontsize=9)
            ax.set_ylabel('dCH4 per event (%)', fontsize=9)
            ax.set_title('CH4 Depletion per Flare\n(OH oxidation; Chen+2021)', fontsize=9)

            # NO2 (flare-enhanced; potential new biosignature)
            ax = axes4[1, 0]
            ax.scatter(resp_a_f['log10_energy'], resp_a_f['dNO2'] * 100,
                       c=resp_a_f['EF_NUV_peak'].clip(1, 5000),
                       cmap='Purples', s=25, alpha=0.75,
                       norm=plt.Normalize(1, 5000))
            ax.set_xlabel('log10(E / erg)', fontsize=9)
            ax.set_ylabel('dNO2 per event (%)', fontsize=9)
            ax.set_title('NO2 Enhancement per Flare\n(Chen+2021: flares can ENHANCE NO2)', fontsize=9)

            # Cumulative 79-day state (bar chart)
            ax = axes4[1, 1]
            cumul_A_row = data['cumul'][data['cumul']['atmosphere'] == 'A'].iloc[0]
            species = ['O3', 'CH4', 'N2O', 'NO2', 'HNO3']
            cols_c  = [f'cumul_d{sp}' for sp in species]
            vals    = [cumul_A_row.get(c, 0.0) * 100 for c in cols_c]
            clrs_c  = ['#1565C0' if v < 0 else '#E53935' for v in vals]
            ax.barh(species, vals, color=clrs_c, alpha=0.8, edgecolor='k', lw=0.5)
            ax.axvline(0, color='k', lw=1)
            ax.set_xlabel('Cumulative change (%)', fontsize=9)
            ax.set_title('79-Day Cumulative State\n(Atm A: N2-Earth)', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("**Key biosignature feature amplitudes**")
            key_feats = ['O3_9.6', 'CH4_3.3', 'CO2_4.3', 'H2O_1.9', 'N2O_4.5']
            feat_q = feats[(feats['atmosphere'] == atm_choice) &
                           (feats['scenario'] == 'quiescent')]
            feat_pf= feats[(feats['atmosphere'] == atm_choice) &
                           (feats['scenario'] == 'postflare_cumul')]

            for fn in key_feats:
                q_row = feat_q[feat_q['feature'] == fn]
                p_row = feat_pf[feat_pf['feature'] == fn]
                if len(q_row) == 0:
                    continue
                amp_q = float(q_row['amplitude_ppm'].iloc[0])
                amp_p = float(p_row['amplitude_ppm'].iloc[0]) if len(p_row) else amp_q
                delta_pct = (amp_p - amp_q) / amp_q * 100 if amp_q > 0 else 0
                st.metric(
                    label=fn.replace('_', ' '),
                    value=f"{amp_q:.1f} ppm (quiescent)",
                    delta=f"{delta_pct:+.2f}% after flares"
                )

            st.markdown("---")
            st.markdown("**JWST detection horizon**")
            st.progress(min(12.7 / 20, 1.0))
            st.caption(
                "O3 at 9.6 um: 12.7 ppm (quiescent)  \n"
                "JWST sensitivity: ~20 ppm for 10 transits  \n"
                "O3 currently **below detection threshold**  \n\n"
                "After 79 days of flares (1D model):  \n"
                f"O3 change: {float(data['cumul'][data['cumul']['atmosphere']=='A']['cumul_dO3'].iloc[0])*100:.2f}%  \n"
                "3D models (Ridgway+2023): O3 would INCREASE 20x"
            )

    # ========================================================================
    # TAB 4: CALIBRATION RESIDUALS
    # ========================================================================
    with tab4:
        st.subheader("Calibration Residuals: Model vs JWST DREAMS")
        st.caption(
            "Residuals = predicted transit depth minus synthetic DREAMS spectrum. "
            "Within the grey band = within 1-sigma JWST noise. "
            "Chi-squared quantifies global agreement."
        )

        fig5, axes5 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        scenarios  = ['quiescent', 'postflare_cumul']
        scen_labels = {'quiescent': 'Quiescent', 'postflare_cumul': 'Post 79-day flares'}
        COLORS = {'A': '#1565C0', 'B': '#B71C1C', 'C': '#546E7A'}

        for ax_i, (atm_l, ax) in enumerate(zip(['A', 'B', 'C'], axes5)):
            for scen, ls in zip(scenarios, ['-', '--']):
                q = spectra_df[(spectra_df['atmosphere'] == atm_l) &
                               (spectra_df['scenario'] == scen)].sort_values('wavelength_um')
                if len(q) == 0:
                    continue
                lam_j = jwst_df['wavelength_um'].values
                mod   = np.interp(lam_j, q['wavelength_um'].values,
                                  q['transit_depth_ppm'].values)
                resid = mod - jwst_df['transit_depth_ppm'].values
                chi2  = float(calib_df[(calib_df['atmosphere']==atm_l) &
                                       (calib_df['scenario']==scen)]['chi2_reduced'].iloc[0]) \
                        if len(calib_df[(calib_df['atmosphere']==atm_l) &
                                       (calib_df['scenario']==scen)]) > 0 else 0.0
                ax.plot(lam_j, resid, color=COLORS[atm_l], lw=1.5, ls=ls, alpha=0.85,
                        label=f"{scen_labels[scen]} (chi2={chi2:.2f})")

            ax.axhline(0, color='k', lw=1, ls='--')
            ax.fill_between(jwst_df['wavelength_um'].values,
                            -jwst_df['uncertainty_ppm'].values,
                            jwst_df['uncertainty_ppm'].values,
                            alpha=0.15, color='gray', label='1-sigma JWST')
            atm_names = {'A': 'N2-Earth', 'B': 'CO2-Venus', 'C': 'Thin rock'}
            ax.set_ylabel(f"Atm {atm_l} ({atm_names[atm_l]})\nResidual (ppm)", fontsize=9)
            ax.set_xlim(0.6, 5.3)
            ax.legend(fontsize=7, loc='upper right')
            ax.axhline(0, color='k', lw=0.5)

        axes5[-1].set_xlabel('Wavelength (um)', fontsize=10)
        axes5[0].set_title(
            'Calibration Residuals: Predicted - JWST DREAMS\n'
            'Grey band = 1-sigma noise | Dashed = post-flare | Solid = quiescent',
            fontsize=10)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        # Chi-squared table
        st.markdown("**Chi-squared summary table**")
        disp = calib_df.copy()
        disp['scenario'] = disp['scenario'].map(
            {'quiescent': 'Quiescent', 'postflare_cumul': 'Post-flare',
             'vulcan_single': 'VULCAN single flare'})
        disp['atmosphere'] = disp['atmosphere'].map(
            {'A': 'A: N2-Earth', 'B': 'B: CO2-Venus', 'C': 'C: Thin rock'})
        disp = disp.rename(columns={
            'chi2_reduced': 'Chi2 (reduced)',
            'rms_ppm': 'RMS (ppm)',
            'mean_offset_ppm': 'Mean offset (ppm)',
        })
        st.dataframe(disp[['atmosphere','scenario','Chi2 (reduced)','RMS (ppm)','Mean offset (ppm)']],
                     use_container_width=True)

        st.info(
            "**Interpretation:** The flat DREAMS spectrum (no thick atmosphere detected) "
            "is best matched by Atm C (thin rock / featureless). "
            "Atm A (N2-Earth with biosignatures) has larger residuals due to CH4/H2O features. "
            "This is consistent with Espinoza+2025 who disfavor thick secondary atmospheres. "
            "However, an N2-dominated atmosphere with low mixing ratios remains **permitted** — "
            "the biosignatures may simply be below JWST sensitivity at current observation depth."
        )

    # ========================================================================
    # TAB 5: CHI-SQUARED GRID
    # ========================================================================
    with tab5:
        st.subheader("Chi-squared Grid: All Atmospheres x Scenarios")

        fig6, ax6 = plt.subplots(figsize=(8, 5))
        pivot = calib_df.pivot_table(
            values='chi2_reduced', index='atmosphere', columns='scenario')
        pivot.columns = [c.replace('_', '\n') for c in pivot.columns]
        pivot.index   = [f"Atm {a}" for a in pivot.index]
        im = ax6.imshow(pivot.values, cmap='RdYlGn_r',
                        vmin=0.8, vmax=pivot.values.max() * 1.1)
        plt.colorbar(im, ax=ax6, label='Chi-squared (reduced)')
        ax6.set_xticks(range(len(pivot.columns)))
        ax6.set_xticklabels(pivot.columns, fontsize=9)
        ax6.set_yticks(range(len(pivot.index)))
        ax6.set_yticklabels(pivot.index, fontsize=10)
        ax6.set_title('Model-JWST Agreement Matrix\nGreen = best fit | Red = poor fit',
                      fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax6.text(j, i, f"{val:.2f}", ha='center', va='center',
                             fontsize=10, fontweight='bold',
                             color='white' if val > pivot.values.max() * 0.7 else 'black')
        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

        st.markdown(
            "**Reading the grid:**  \n"
            "- Chi2 ~ 1.0 = perfect statistical agreement  \n"
            "- Chi2 ~ 1-2 = acceptable  \n"
            "- Chi2 > 3 = model tension with DREAMS data  \n\n"
            "**Digital twin feedback loop:**  \n"
            "As new JWST TRAPPIST-1e data arrives (GO 6456/9256 programme), "
            "this chi-squared grid updates automatically, narrowing which atmospheric "
            "compositions are viable and refining the photochemical model parameters."
        )

    # ========================================================================
    # TAB 6: FLARE ANIMATION
    # ========================================================================
    with tab6:
        st.subheader("Flare Sequence Animation: Spectral Evolution")
        st.caption(
            f"Stepping through the 79-day flare sequence for Atm A (N2-Earth). "
            f"Current position: day {timeline_day:.1f}. "
            "Use the sidebar slider to animate. "
            "Spectrum evolves as cumulative O3/CH4/N2O changes accumulate with recovery."
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            fig7, axes7 = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

            ax_s = axes7[0]
            q_spec = spectra_df[(spectra_df['atmosphere'] == 'A') &
                                (spectra_df['scenario'] == 'quiescent')].sort_values('wavelength_um')
            lam_q   = q_spec['wavelength_um'].values
            dep_q   = q_spec['transit_depth_ppm'].values
            nirspec = (lam_q >= 0.6) & (lam_q <= 5.3)

            # Quiescent reference
            ax_s.plot(lam_q[nirspec], dep_q[nirspec], 'k-', lw=2,
                      alpha=0.5, label='Quiescent (day 0)', zorder=5)

            # JWST DREAMS
            jwst_m = (jwst_df['wavelength_um'] >= 0.6) & (jwst_df['wavelength_um'] <= 5.3)
            ax_s.errorbar(jwst_df['wavelength_um'][jwst_m],
                          jwst_df['transit_depth_ppm'][jwst_m],
                          yerr=jwst_df['uncertainty_ppm'][jwst_m],
                          fmt='.', color='gray', ms=3, elinewidth=0.6, alpha=0.5, zorder=4,
                          label='JWST DREAMS (synthetic)')

            # Current day spectrum
            if timeline_day > 0:
                stepped = spectrum_at_day(spectra_df, resp_A, timeline_day, atm='A')
                stepped = stepped.sort_values('wavelength_um')
                lam_st  = stepped['wavelength_um'].values
                dep_st  = stepped['transit_depth_ppm'].values
                m_st    = (lam_st >= 0.6) & (lam_st <= 5.3)
                ax_s.plot(lam_st[m_st], dep_st[m_st], color='#1565C0', lw=2.5,
                          label=f'Day {timeline_day:.0f} (post-flares)', zorder=6)
                ax_s.fill_between(lam_q[nirspec], dep_q[nirspec],
                                  np.interp(lam_q[nirspec], lam_st[m_st], dep_st[m_st]),
                                  alpha=0.25, color='red',
                                  label='Spectral change from flares')

            # Feature lines
            for lam0, name in [(1.38,'H2O'),(2.3,'CH4'),(3.3,'CH4'),(4.3,'CO2')]:
                ax_s.axvline(lam0, color='purple', lw=0.7, ls=':', alpha=0.5)
                ax_s.text(lam0, dep_q[nirspec].max(), name,
                          ha='center', fontsize=7, color='purple', rotation=90, va='top')

            ax_s.set_ylabel('Transit Depth (ppm)', fontsize=10)
            ax_s.set_title(f'Atm A (N2-Earth) at Day {timeline_day:.0f} / 79', fontsize=10)
            ax_s.legend(fontsize=8)

            # Timeline progress subplot
            ax_t = axes7[1]
            all_flares_up = resp_A[resp_A['time_days'] <= timeline_day]
            all_flares_future = resp_A[resp_A['time_days'] > timeline_day]
            ax_t.scatter(all_flares_future['time_days'], all_flares_future['log10_energy'],
                         c='lightgray', s=15, alpha=0.5, label='Future')
            ax_t.scatter(all_flares_up['time_days'], all_flares_up['log10_energy'],
                         c=all_flares_up['log10_energy'], cmap='hot_r',
                         vmin=30, vmax=33, s=30, alpha=0.85, label='Occurred', zorder=3)
            if timeline_day > 0:
                ax_t.axvline(timeline_day, color='red', lw=2, label=f'Day {timeline_day:.0f}')
            ax_t.set_xlabel('Days', fontsize=9)
            ax_t.set_ylabel('log10(E)', fontsize=9)
            ax_t.set_xlim(0, cat['time_days'].max())
            ax_t.legend(fontsize=7)

            plt.tight_layout()
            st.pyplot(fig7, use_container_width=True)
            plt.close()

        with col2:
            # Current atmospheric state
            st.markdown("**Atmospheric state at day {:.0f}**".format(timeline_day))
            state = compute_cumulative_at_day(resp_A, timeline_day)
            n_occurred = (resp_A['time_days'] <= timeline_day).sum()
            st.metric("Flares occurred", n_occurred,
                      delta=f"{len(resp_A) - n_occurred} remaining")
            for sp, val in state.items():
                if abs(val) > 1e-5:
                    st.metric(
                        f"Cumul. d{sp}",
                        f"{val*100:.3f}%",
                        delta="depleted" if val < 0 else "enhanced",
                    )

            st.markdown("---")
            st.caption(
                "**How to use:**  \n"
                "Drag the 'Flare sequence: day' slider in the sidebar to step through "
                "the 79-day observation window. Each flare event modifies the "
                "atmospheric chemistry (with partial recovery), shifting the "
                "predicted transmission spectrum."
            )
            st.caption(
                "**Scientific context:**  \n"
                "This stepped animation shows what JWST would observe at each "
                "epoch during a multi-transit programme. The spectral change per "
                "step is sub-ppm — below current JWST sensitivity — demonstrating "
                "that **individual flares do not produce detectable spectral changes** "
                "with the current observing programme."
            )

    # ========================================================================
    # TAB 0: TWIN STATE MONITOR
    # ========================================================================
    with tab0:
        render_twin_state_monitor(twin_state, data)

    # ========================================================================
    # TAB 7: BIOSPHERE ENGINE
    # ========================================================================
    with tab7:
        render_biosphere_engine(twin_state, data)

    # ========================================================================
    # TAB 8: FORWARD PREDICTION
    # ========================================================================
    with tab8:
        render_forward_prediction(twin_state)

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**NAKA Biosignature Detection | TRAPPIST-1e Digital Twin**  \n"
        "Pipeline: Stage 1 (TESS Ingest) → Stage 2 (Photochemistry) → "
        "Stage 3 (Spectral RT) → Stage 4 (JWST Calibration)  \n"
        "Key references: Espinoza+2025 (DREAMS), Ducrot+2022, Segura+2010, "
        "Tilley+2019, Chen+2021, Ridgway+2023, Miranda-Rosete+2024, "
        "Lecavelier des Etangs+2008, HITRAN2020"
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if _STREAMLIT:
    run_streamlit_dashboard()

else:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n" + "#"*65)
    print("  TRAPPIST-1e DIGITAL TWIN -- STAGE 4: JWST CALIBRATION")
    print("#"*65 + "\n")
    print("  Run with:  streamlit run stage4_dashboard.py")
    print("  (Generating static summary figure...)\n")

    print("="*65)
    print("Loading all pipeline data")
    print("="*65)
    data = load_all_data()
    print(f"  Flare catalog:   {len(data['cat'])} flares")
    print(f"  Spectra:         {len(data['spectra'])} wavelength-atmosphere entries")
    print(f"  Feature amps:    {len(data['feats'])} entries")

    print("\n" + "="*65)
    print("Building synthetic JWST DREAMS spectrum")
    print("="*65)
    lam_all = np.sort(data['spectra']['wavelength_um'].unique())
    jwst_df = build_dreams_spectrum(lam_all, n_transits=5)
    print(f"  {len(jwst_df)} wavelength bins, 0.6-5.3 um (NIRSpec/PRISM)")
    print(f"  Noise range: {jwst_df['uncertainty_ppm'].min():.1f} - "
          f"{jwst_df['uncertainty_ppm'].max():.1f} ppm per bin")
    print(f"  Baseline depth: {jwst_df['transit_depth_ppm'].mean():.1f} +/- "
          f"{jwst_df['transit_depth_ppm'].std():.1f} ppm")
    jwst_df.to_csv(f"{STAGE4}/jwst_dreams_synthetic.csv", index=False)

    print("\n" + "="*65)
    print("Chi-squared calibration")
    print("="*65)
    calib_df = compute_calibration(data['spectra'], jwst_df)
    print("\n  Results (sorted by chi2_reduced):")
    for _, r in calib_df.iterrows():
        print(f"    Atm {r['atmosphere']} / {r['scenario']:20s}: "
              f"chi2 = {r['chi2_reduced']:.3f}  RMS = {r['rms_ppm']:.1f} ppm")

    best = calib_df.iloc[0]
    print(f"\n  Best fit: Atm {best['atmosphere']} ({best['scenario']}), "
          f"chi2_red = {best['chi2_reduced']:.3f}")
    print("  Interpretation: flat DREAMS spectrum (no thick atmosphere)")
    print("  -> consistent with thin/absent secondary atmosphere")
    print("  -> N2-Earth (Atm A) has larger residuals from CH4/H2O features")

    print("\n" + "="*65)
    print("Generating static summary figure")
    print("="*65)
    path = generate_static_summary(data, jwst_df, calib_df)

    print("\n" + "="*65)
    print("STAGE 4 COMPLETE")
    print("="*65)
    print(f"""
  TRAPPIST-1e (TIC 267519185) -- Stage 4 Results
  ───────────────────────────────────────────────
  JWST synthetic spectrum: {len(jwst_df)} bins (NIRSpec/PRISM 0.6-5.3 um)
  Best-fit atmosphere:     Atm {best['atmosphere']} (chi2_red = {best['chi2_reduced']:.2f})

  Calibration interpretation:
    The featureless DREAMS spectrum best matches Atm C (thin remnant) or
    Atm A at low feature contrast. An N2-dominated atmosphere with sub-ppb
    biosignature mixing ratios remains consistent with observations.

    Key finding: O3 feature (12.7 ppm at 9.6 um) is below the JWST
    detection horizon (~20 ppm for 10 transits of TRAPPIST-1e).
    Even with the 79-day flare sequence, spectral changes are sub-ppm --
    within the systematic noise floor.

  To launch interactive dashboard:
    streamlit run stage4_dashboard.py

  Output files:  {STAGE4}/
    jwst_dreams_synthetic.csv
    calibration_results.csv
    stage4_summary.png

  Full pipeline:
    Stage 1: trappist1_flare_catalog.csv, stellar_spectrum.csv
    Stage 2: atmospheric_compositions.csv, photochem response
    Stage 3: transmission spectra, feature_amplitudes.csv
    Stage 4: JWST calibration, interactive dashboard
""")
