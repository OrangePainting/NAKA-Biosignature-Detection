"""
twin_core.py — TRAPPIST-1e Digital Twin: Shared State Engine
=============================================================

The TRAPPIST-1e digital twin maintains a persistent, evolving model of the
planet's atmosphere and biosphere. Unlike a one-pass analysis pipeline, the
twin accumulates evidence over time, responds to new observations, and can be
projected forward into the future.

State is serialized to twin_state.json in the project root and read/written
by each pipeline stage. The dashboard reads it with a short cache TTL so
the UI reflects the latest run within ~30 seconds.

Key concepts
------------
1. Living atmosphere: VMRs for all 3 compositions evolve as flares accumulate.
2. Biosphere feedback (Eager-Nash et al. 2024):
     - Methanogenesis:  4H2 + CO2 -> 2H2O + CH4  (biogenic CH4 source)
     - CO consumption:  4CO + 2H2O -> 2CO2 + CH3COOH  (CO sink)
     - UV stress: cumulative NUV dose degrades biosphere population.
3. Forward prediction: Monte Carlo future flare sampling + photochem LUT.
4. Data assimilation: Bayesian posterior update when JWST data is uploaded.

References
----------
Eager-Nash et al. 2024  MNRAS 531, 468  (biosphere model, CO feedback)
Segura+2010; Tilley+2019; Chen+2021    (photochem LUT — same as Stage 2)
"""

import json
import os
import copy
import numpy as np

STATE_FILE = "twin_state.json"

# ---------------------------------------------------------------------------
# Photochem LUT nodes (duplicated here from stage2 so twin_core is standalone)
# x = log10(F_NUV / erg cm^-2) at planet; values = fractional change per flare
# ---------------------------------------------------------------------------
_LUT_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_LUT_A = {
    'O3':  np.array([-5e-5, -2e-4, -8e-4, -3e-3, -8e-3, -2.0e-2, -5e-2, -0.18]),
    'CH4': np.array([-3e-6, -1e-5, -5e-5, -2e-4, -5e-4, -2.0e-3, -7e-3, -2.5e-2]),
    'N2O': np.array([+2e-6, +8e-6, +3e-5, +1e-4, +3e-4, +8e-4,  +1e-3, -5e-3]),
    'NO2': np.array([+5e-5, +2e-4, +8e-4, +3e-3, +1e-2, +4e-2,  +1.5e-1, +5e-1]),
    'CO':  np.array([+1e-7, +5e-7, +2e-6, +8e-6, +3e-5, +1.5e-4, +8e-4, +4e-3]),
}
_RECOVERY_TAU = {
    'O3': 30.0, 'CH4': 1825.0, 'N2O': 14.0,
    'NO2': 0.5,  'HNO3': 3.0,  'OH': 0.0001,
    'CO':  5.0,  'SO2': 7.0,
}

# UV energy fractions of bolometric flare energy (Kowalski+2013)
_F_UV_NUV    = 0.025
_FOUR_PI_A2  = 4.0 * np.pi * (0.02928 * 1.496e13)**2   # cm^2

# Biosphere constants (Eager-Nash+2024, Table 2 + PALEO model parameters)
_BIO_CO_CONSUMPTION   = 0.97    # fraction of abiotic CO consumed by healthy biosphere
_BIO_CH4_RATE_VMR_DAY = 2.0e-11 # CH4 VMR added per day by healthy biosphere
_BIO_UV_STRESS_SCALE  = 5.0e9   # erg/cm^2 — dose that halves population (sigma in Gaussian)


# ===========================================================================
# STATE INITIALIZATION & PERSISTENCE
# ===========================================================================

def initialize_twin_state(flare_catalog_rows: list) -> dict:
    """
    Create a fresh TwinState dict from the Stage 1 flare catalog.
    Writes twin_state.json. Called at the end of Stage 1.

    flare_catalog_rows: list of dicts from flare_catalog.csv
    """
    from datetime import datetime, timezone

    # Initial atmospheric VMRs for all 3 compositions (mirrors stage2 definitions)
    atm_init = {
        'A': {
            'N2': 0.7900, 'O2': 0.2095, 'CO2': 4.20e-4,
            'CH4': 1.80e-6, 'O3': 3.00e-7, 'N2O': 3.20e-7,
            'H2O': 1.00e-2, 'NO2': 5.00e-10, 'HNO3': 3.00e-10,
            'CO': 1.00e-7,   # abiotic baseline (M-dwarf UV enhanced; Eager-Nash+2024)
        },
        'B': {
            'CO2': 0.9600, 'N2': 0.0350, 'SO2': 1.50e-4,
            'CO':  2.00e-5, 'O2': 5.00e-7, 'O3': 1.00e-10,
            'H2O': 1.00e-5, 'HCl': 5.00e-7,
        },
        'C': {
            'N2': 0.9800, 'CO2': 0.0200, 'Ar': 1.00e-4,
        },
    }

    state = {
        'meta': {
            'created_utc':       datetime.now(timezone.utc).isoformat(),
            'last_updated_utc':  datetime.now(timezone.utc).isoformat(),
            'last_updated_by':   'stage1',
            'pipeline_version':  '2.0-digital-twin',
            'obs_days_completed': 79.0,
            'n_flares_total':    len(flare_catalog_rows),
        },

        # Current best-estimate atmospheric VMRs (evolve as flares accumulate)
        'atmosphere': copy.deepcopy(atm_init),

        # Initial (background) VMRs — never mutated; used as recovery targets
        'atmosphere_initial': copy.deepcopy(atm_init),

        # Cumulative fractional VMR changes (fractional from initial)
        'cumulative_changes': {'A': {}, 'B': {}, 'C': {}},

        # Biosphere state (applies to Atm A; Eager-Nash+2024 model)
        'biosphere': {
            'active':                  True,
            'population_factor':       1.0,          # 1.0 = fully healthy
            'cumulative_uv_dose':      0.0,           # erg cm^-2
            'uv_lethal_threshold':     _BIO_UV_STRESS_SCALE * 5.0,
            'co_vmr_abiotic':          1.00e-7,       # what CO would be without life
            'co_vmr_biotic':           3.00e-9,       # actual CO with CO consumption
            'ch4_production_rate':     1.0,           # normalised; 1.0 = full flux
            'methanogenesis_active':   True,
            'co_consumption_active':   True,
            'stress_events':           [],
            'h2_outgassing_tmol_yr':   5.0,           # geological H2 supply
        },

        # Flare event log (last 100 events to keep JSON small)
        'flare_log': [
            {
                'day':          r.get('time_days', 0),
                'energy_erg':   r.get('energy_erg', 0),
                'log10_energy': r.get('log10_energy', 0),
                'duration_sec': r.get('duration_sec', 0),
                'flare_class':  str(r.get('class', 'unknown')),
            }
            for r in flare_catalog_rows[-100:]
        ],

        # JWST observation log and posterior weights
        'observations': [],
        'posterior_weights': {'A': 1/3, 'B': 1/3, 'C': 1/3},

        # Forward prediction cache (filled by forward_predict())
        'predictions': {},

        # State history for time-series display (filled by stage2)
        'history': [],
    }

    save_twin_state(state)
    return state


def load_twin_state(path: str = STATE_FILE) -> dict:
    """Load state from JSON. Returns a fresh empty state if file not found."""
    if not os.path.exists(path):
        return initialize_twin_state([])
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_twin_state(state: dict, path: str = STATE_FILE) -> None:
    """Atomic write: write to .tmp then rename to avoid corruption."""
    def _make_serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _make_serial(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serial(i) for i in obj]
        return obj

    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_make_serial(state), f, indent=2)
    os.replace(tmp, path)


# ===========================================================================
# ATMOSPHERE STATE HELPERS
# ===========================================================================

def update_atmosphere(state: dict, atm_label: str, species_changes: dict) -> dict:
    """
    Apply fractional VMR changes to state['atmosphere'][atm_label].
    Clips all VMRs to [0, 1]. Mutates and returns state.
    """
    vmr = state['atmosphere'][atm_label]
    for sp, frac in species_changes.items():
        if sp in vmr and np.isfinite(frac):
            vmr[sp] = float(np.clip(vmr[sp] * (1.0 + frac), 0.0, 1.0))
    return state


# ===========================================================================
# BIOSPHERE MODEL  (Eager-Nash et al. 2024)
# ===========================================================================

def biosphere_co_consumption(state: dict, co_vmr_abiotic: float) -> float:
    """
    CO consumption by biosphere via 4CO + 2H2O -> 2CO2 + CH3COOH.

    A healthy biosphere (population_factor = 1.0) consumes _BIO_CO_CONSUMPTION
    fraction of the abiotic CO. Consumption scales linearly with population.

    Returns the biotic CO VMR for insertion into Atm A.

    Eager-Nash+2024: CO is a dominant spectral feature on M-dwarf planets due
    to enhanced CO2 photolysis. A living biosphere suppresses it by ~2 orders
    of magnitude, making CO the clearest spectral signature of life.
    """
    bio = state['biosphere']
    if not bio['active'] or not bio['co_consumption_active']:
        return co_vmr_abiotic
    pf = bio['population_factor']
    biotic_co = co_vmr_abiotic * (1.0 - _BIO_CO_CONSUMPTION * pf)
    biotic_co = max(biotic_co, co_vmr_abiotic * 0.001)  # floor at 0.1% of abiotic
    bio['co_vmr_abiotic'] = float(co_vmr_abiotic)
    bio['co_vmr_biotic']  = float(biotic_co)
    return float(biotic_co)


def biosphere_ch4_production(state: dict, dt_days: float) -> float:
    """
    Biogenic CH4 from methanogenesis: 4H2 + CO2 -> 2H2O + CH4.

    Production rate scales with population_factor and geological H2 supply.
    Returns VMR increment to add to Atm A CH4 over dt_days.

    Base rate (_BIO_CH4_RATE_VMR_DAY) is calibrated so a healthy biosphere
    maintains the Earth-like CH4 mixing ratio (~1.8 ppm) against OH destruction.
    """
    bio = state['biosphere']
    if not bio['active'] or not bio['methanogenesis_active']:
        return 0.0
    pf   = bio['population_factor']
    rate = _BIO_CH4_RATE_VMR_DAY * pf * bio['ch4_production_rate']
    return float(rate * dt_days)


def update_biosphere_stress(state: dict, ef_nuv: float, duration_sec: float,
                             f_nuv_quiescent: float = 0.070) -> dict:
    """
    Accumulate UV stress from a flare event.

    Dose increment = (EF_NUV - 1) * F_NUV_quiescent * duration_sec
    Population factor decays as: pf *= exp(-dose / _BIO_UV_STRESS_SCALE)

    This implements the Eager-Nash+2024 UV stress model where cumulative
    UV exposure progressively reduces biosphere productivity.
    """
    bio = state['biosphere']
    if not bio['active']:
        return state

    dose = max(0.0, ef_nuv - 1.0) * f_nuv_quiescent * duration_sec
    bio['cumulative_uv_dose'] = bio.get('cumulative_uv_dose', 0.0) + dose

    if dose > 0:
        decay = float(np.exp(-dose / _BIO_UV_STRESS_SCALE))
        bio['population_factor'] = float(np.clip(bio['population_factor'] * decay, 0.0, 1.0))

    # Log stress events above a threshold
    if ef_nuv > 100:
        bio.setdefault('stress_events', []).append({
            'day':              state.get('meta', {}).get('current_day', 0.0),
            'ef_nuv':           float(ef_nuv),
            'dose_erg_cm2':     float(dose),
            'population_after': float(bio['population_factor']),
        })

    # Biosphere collapse threshold
    if bio['population_factor'] < 0.01:
        bio['active'] = False
        bio['population_factor'] = 0.0

    return state


def biosphere_recovery(state: dict, dt_days: float,
                        recovery_timescale_days: float = 730.0) -> dict:
    """
    Exponential recovery of population_factor between stress events.
    pf(t + dt) = 1.0 - (1.0 - pf(t)) * exp(-dt / tau)
    A fully killed biosphere does not recover (Eager-Nash+2024 caveat).
    """
    bio = state['biosphere']
    if not bio['active'] or bio['population_factor'] >= 1.0:
        return state
    pf = bio['population_factor']
    new_pf = 1.0 - (1.0 - pf) * float(np.exp(-dt_days / recovery_timescale_days))
    bio['population_factor'] = float(np.clip(new_pf, 0.0, 1.0))
    return state


# ===========================================================================
# FORWARD PREDICTION ENGINE
# ===========================================================================

def forward_predict(state: dict, horizon_days: int = 90,
                    flare_rate_multiplier: float = 1.0,
                    seed: int = 1234) -> dict:
    """
    Project atmospheric and biosphere state forward in time using Monte Carlo
    future flare sampling and the photochem LUT.

    Parameters
    ----------
    state                  : current TwinState dict
    horizon_days           : days to forecast (30, 90, or 365)
    flare_rate_multiplier  : 1.0 = nominal, 2.0 = high activity, 0.5 = quiet
    seed                   : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
      'days'       : list of time points (days)
      'atm_A'      : dict {species: list of VMR values}
      'biosphere'  : dict {population_factor: list, co_vmr: list, ch4_vmr: list}
      'uncertainty': dict {species: list of 1-sigma spread from MC ensemble}
      'horizon_days': int
    """
    rng = np.random.default_rng(seed)

    # Current state snapshot
    vmr0    = copy.deepcopy(state['atmosphere']['A'])
    bio0    = copy.deepcopy(state['biosphere'])
    pf0     = bio0['population_factor']

    # Ensure CO is present in Atm A
    if 'CO' not in vmr0:
        vmr0['CO'] = bio0.get('co_vmr_biotic', 3e-9)

    # Future flare sampling
    rate     = 0.53 * flare_rate_multiplier
    n_flares = int(rng.poisson(rate * horizon_days))
    alpha, E_min, E_max = 1.72, 1e30, 5e33
    b = 1.0 - alpha
    u = rng.uniform(0, 1, max(n_flares, 1))
    energies = (u * (E_max**b - E_min**b) + E_min**b) ** (1.0 / b)
    times    = np.sort(rng.uniform(0, horizon_days, max(n_flares, 1)))

    # Time grid for output
    n_steps = min(horizon_days * 4, 400)
    days_out = np.linspace(0, horizon_days, n_steps)

    # Tracking arrays
    species_tracked = ['O3', 'CH4', 'CO', 'N2O', 'NO2']
    traj = {sp: np.zeros(n_steps) for sp in species_tracked}
    bio_pf   = np.zeros(n_steps)
    bio_co   = np.zeros(n_steps)
    bio_ch4  = np.zeros(n_steps)

    # Run forward integration
    vmr  = copy.deepcopy(vmr0)
    pf   = pf0
    prev_t = 0.0
    flare_idx = 0

    for step_i, day in enumerate(days_out):
        # Apply flares that occurred since last step
        while flare_idx < len(times) and times[flare_idx] <= day:
            E     = energies[flare_idx]
            E_NUV = E * _F_UV_NUV
            F_NUV = E_NUV / _FOUR_PI_A2
            log_F = float(np.clip(np.log10(max(F_NUV, 1e-30)), 1.0, 8.0))
            dur   = 600.0 * (E / 1e32)**0.39  # seconds, Davenport 2014

            # Photochemistry from LUT
            for sp, lut_arr in _LUT_A.items():
                dX = float(np.interp(log_F, _LUT_X, lut_arr))
                if sp in vmr:
                    vmr[sp] = float(np.clip(vmr[sp] * (1.0 + dX), 0.0, 1.0))

            # Biosphere UV stress
            ef_nuv = 1.0 + (F_NUV / (0.070 * dur)) if dur > 0 else 1.0
            ef_nuv = float(np.clip(ef_nuv, 1.0, 1e7))
            dose   = max(0.0, ef_nuv - 1.0) * 0.070 * dur
            pf    *= float(np.exp(-dose / _BIO_UV_STRESS_SCALE))
            pf     = float(np.clip(pf, 0.0, 1.0))

            flare_idx += 1

        # Recovery between steps
        dt = day - prev_t
        for sp in species_tracked:
            tau = _RECOVERY_TAU.get(sp, 30.0)
            if sp in vmr and tau > 0:
                bg = vmr0.get(sp, vmr[sp])
                vmr[sp] = float(bg + (vmr[sp] - bg) * np.exp(-dt / tau))

        # Biosphere recovery
        pf = float(1.0 - (1.0 - pf) * np.exp(-dt / 730.0))

        # Biosphere CO consumption and CH4 production
        co_abiotic = vmr.get('CO', vmr0.get('CO', 1e-7))
        co_biotic  = co_abiotic * (1.0 - _BIO_CO_CONSUMPTION * pf)
        vmr['CO']  = float(max(co_biotic, co_abiotic * 0.001))
        vmr['CH4'] = float(np.clip(
            vmr.get('CH4', vmr0.get('CH4', 1.8e-6)) + _BIO_CH4_RATE_VMR_DAY * pf * dt,
            0.0, 1.0))

        # Record
        for sp in species_tracked:
            traj[sp][step_i] = vmr.get(sp, 0.0)
        bio_pf[step_i]  = pf
        bio_co[step_i]  = vmr.get('CO', 0.0)
        bio_ch4[step_i] = vmr.get('CH4', 0.0)
        prev_t = day

    # Build output
    pred = {
        'horizon_days': horizon_days,
        'flare_rate_multiplier': flare_rate_multiplier,
        'days': days_out.tolist(),
        'atm_A': {sp: traj[sp].tolist() for sp in species_tracked},
        'biosphere': {
            'population_factor': bio_pf.tolist(),
            'co_vmr':            bio_co.tolist(),
            'ch4_vmr':           bio_ch4.tolist(),
        },
    }
    state['predictions'] = pred
    return pred


# ===========================================================================
# DATA ASSIMILATION
# ===========================================================================

def assimilate_observation(state: dict, lam_jwst: np.ndarray,
                            obs_ppm: np.ndarray, sigma_ppm: np.ndarray,
                            spectra_dict: dict, obs_id: str = 'uploaded') -> dict:
    """
    Bayesian posterior update given a new JWST observation.

    spectra_dict: {(atm, scenario): transit_depth_ppm_array at lam_jwst}
                  e.g. {('A','quiescent'): array, ('B','quiescent'): array, ...}

    Method:
      1. chi2 per (atm, scenario) pair
      2. Likelihood L_i = exp(-chi2_i / 2)
      3. Posterior = prior * L  (normalised)
      4. Best-fit atm identified
      5. Appended to state['observations']
    """
    from datetime import datetime, timezone

    prior   = state.get('posterior_weights', {'A': 1/3, 'B': 1/3, 'C': 1/3})
    chi2s   = {}
    likes   = {}

    for (atm, scen), model_ppm in spectra_dict.items():
        model_at_jwst = np.interp(lam_jwst, lam_jwst, model_ppm)
        chi2 = float(np.mean(((model_at_jwst - obs_ppm) / np.maximum(sigma_ppm, 1.0))**2))
        chi2s[(atm, scen)] = chi2
        likes[atm] = likes.get(atm, 0.0) + float(np.exp(-chi2 / 2.0))

    # Posterior for each atmosphere (averaged over scenarios)
    n_scen = max(sum(1 for (a, _) in spectra_dict if a == atm_l) for atm_l in 'ABC')
    posterior_raw = {
        atm: prior.get(atm, 1/3) * likes.get(atm, 1e-30)
        for atm in ['A', 'B', 'C']
    }
    total = sum(posterior_raw.values()) or 1.0
    posterior = {atm: v / total for atm, v in posterior_raw.items()}

    best_atm = max(posterior, key=posterior.__getitem__)
    best_chi2 = min(chi2s.values())

    obs_record = {
        'obs_id':          obs_id,
        'uploaded_utc':    datetime.now(timezone.utc).isoformat(),
        'n_bins':          len(lam_jwst),
        'best_fit_atm':    best_atm,
        'chi2_best':       best_chi2,
        'chi2_by_atm':     {atm: min(v for (a, _), v in chi2s.items() if a == atm)
                            for atm in ['A', 'B', 'C']
                            if any(a == atm for a, _ in chi2s)},
        'posterior_weights': posterior,
    }
    state.setdefault('observations', []).append(obs_record)
    state['posterior_weights'] = posterior
    return state


# ===========================================================================
# DETECTION HORIZON
# ===========================================================================

def compute_detection_horizon(state: dict, species: str,
                               feature_amp_ppm: float,
                               noise_per_transit_ppm: float = 50.0) -> dict:
    """
    Given current VMR and spectral feature amplitude, compute:
      - Current amplitude (ppm)
      - Transits needed for 5-sigma detection
      - Biosphere contribution to amplitude
      - Whether detectable in the current DREAMS programme (5 transits)

    noise_per_transit_ppm: sigma per spectral bin per transit (default ~50 ppm
    for TRAPPIST-1e at NIRSpec/PRISM resolution, Espinoza+2025).
    """
    vmr_now  = state['atmosphere']['A'].get(species, 0.0)
    vmr_init = state['atmosphere_initial']['A'].get(species, vmr_now)
    vmr_ratio = vmr_now / vmr_init if vmr_init > 0 else 1.0

    amp_now = feature_amp_ppm * vmr_ratio

    # Biosphere enhancement: CH4 is enhanced by methanogenesis
    bio_pf = state['biosphere']['population_factor']
    if species == 'CH4':
        bio_boost_ppm = feature_amp_ppm * _BIO_CH4_RATE_VMR_DAY * bio_pf * 79.0 / vmr_init if vmr_init > 0 else 0.0
    elif species == 'CO':
        # Biosphere suppresses CO: biotic < abiotic
        bio_boost_ppm = -(feature_amp_ppm * (1.0 - vmr_ratio))  # negative = suppressed
    else:
        bio_boost_ppm = 0.0

    sigma_5transit = noise_per_transit_ppm / np.sqrt(5)
    transits_needed_5sig = max(1.0, (5.0 * noise_per_transit_ppm / max(amp_now, 0.1))**2)
    detectable_now = amp_now > 5.0 * sigma_5transit

    return {
        'species':              species,
        'amplitude_ppm':        float(amp_now),
        'vmr_ratio':            float(vmr_ratio),
        'transits_needed_5sig': float(transits_needed_5sig),
        'detectable_5transits': bool(detectable_now),
        'biosphere_effect_ppm': float(bio_boost_ppm),
        'noise_per_transit':    float(noise_per_transit_ppm),
    }
