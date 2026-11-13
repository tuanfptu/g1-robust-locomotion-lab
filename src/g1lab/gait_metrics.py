"""Sprint gait metrics from MuJoCo telemetry (acceptance side; independent of the Isaac reward proxies).

Inputs are per-physics-substep samples (500 Hz) of foot/floor contact, split into toe and heel contact
spheres, plus forward ground force, and per-control-step kinematics. All functions are pure (numpy only).
"""

import numpy as np


def _runs(flags):
    """(start, stop) index pairs of consecutive True runs."""
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def debounce(contact, release_samples):
    """Fill no-contact gaps shorter than `release_samples` inside a stance (per foot, edges untouched)."""
    contact = np.array(contact, dtype=bool, copy=True)
    for foot in range(contact.shape[1]):
        for start, stop in _runs(~contact[:, foot]):
            if start != 0 and stop != len(contact) and stop - start < release_samples:
                contact[start:stop, foot] = True
    return contact


def _median(values):
    return float(np.median(values)) if len(values) else None


def contact_metrics(contact, toe, heel, forward_force, dt, body_weight, release_s=0.04):
    """contact/toe/heel: (T, 2) bool per foot; forward_force: (T,) N in heading frame; dt: s per sample.

    Contact is debounced (gaps < release_s inside a stance are filled, matching the Isaac reward tracker).
    Strike type comes from which spheres carry load in the first loaded sample of each stance.
    Only complete phases (not touching the window edges) are counted.
    """
    contact = debounce(contact, max(1, round(release_s / dt)))
    toe, heel = np.asarray(toe, dtype=bool), np.asarray(heel, dtype=bool)
    forward_force = np.asarray(forward_force, dtype=float)
    t = len(contact)
    stance, swing, peak_force, push = [], [], [], []
    strikes = {"forefoot": 0, "heel": 0, "flat": 0}
    for foot in range(2):
        for start, stop in _runs(contact[:, foot]):
            if start == 0 or stop == t:
                continue
            stance.append((stop - start) * dt)
            loaded = np.flatnonzero(toe[start:stop, foot] | heel[start:stop, foot])
            first = start + (loaded[0] if len(loaded) else 0)
            toe_first, heel_first = toe[first, foot], heel[first, foot]
            strikes[
                "forefoot"
                if toe_first and not heel_first
                else "heel"
                if heel_first and not toe_first
                else "flat"
            ] += 1
            segment = forward_force[start:stop] / body_weight
            peak_force.append(float(segment.max()))
            push.append(float(np.clip(segment, 0, None).mean()))
        for start, stop in _runs(~contact[:, foot]):
            if start == 0 or stop == t:
                continue
            swing.append((stop - start) * dt)
    flight = ~contact.any(axis=1)
    flights = [(b - a) * dt for a, b in _runs(flight) if a != 0 and b != t]
    touchdowns = sum(strikes.values())
    duration = t * dt
    return {
        "flight_fraction": float(flight.mean()) if t else None,
        "double_support_fraction": float(contact.all(axis=1).mean()) if t else None,
        "median_flight_time_s": _median(flights),
        "flights_per_s": len(flights) / duration if duration else None,
        "median_stance_time_s": _median(stance),
        "median_swing_time_s": _median(swing),
        "cadence_steps_per_s": touchdowns / duration if duration else None,
        "forefoot_strike_fraction": strikes["forefoot"] / touchdowns if touchdowns else None,
        "heel_strike_fraction": strikes["heel"] / touchdowns if touchdowns else None,
        "median_peak_forward_grf_bw": _median(peak_force),
        "median_mean_propulsive_grf_bw": _median(push),
    }


def swing_kinematics(contact_steps, knee_drop, ankle_pitch_rel, shoulder_pitch, hip_pitch, torso_pitch_deg):
    """Per-control-step arrays; contact_steps: (S, 2) bool (foot on ground at the end of the step).

    knee_drop: pelvis minus knee height (m), lower = higher knee. ankle_pitch_rel: ankle pitch minus default
    (rad), negative = dorsiflexion. shoulder/hip pitch: (S, 2) joint angles.
    """
    contact_steps = np.asarray(contact_steps, dtype=bool)
    knee_drop, ankle = np.asarray(knee_drop), np.asarray(ankle_pitch_rel)
    knee_best, dorsi_best = [], []
    s = len(contact_steps)
    for foot in range(2):
        for start, stop in _runs(~contact_steps[:, foot]):
            if start == 0 or stop == s:
                continue
            knee_best.append(float(knee_drop[start:stop, foot].min()))
            dorsi_best.append(float(ankle[start:stop, foot].min()))
    shoulder, hip = np.asarray(shoulder_pitch), np.asarray(hip_pitch)
    amplitude = [
        float(np.percentile(shoulder[:, i], 95) - np.percentile(shoulder[:, i], 5)) for i in range(2)
    ]
    coupling = [
        float(np.corrcoef(shoulder[:, i], hip[:, i])[0, 1])
        if np.std(shoulder[:, i]) > 1e-6 and np.std(hip[:, i]) > 1e-6
        else None
        for i in range(2)
    ]
    return {
        "median_swing_min_knee_drop_m": _median(knee_best),
        "median_swing_peak_dorsiflexion_rad": _median(dorsi_best),
        "shoulder_pitch_amplitude_rad": amplitude,
        "shoulder_hip_same_side_correlation": coupling,
        "mean_torso_pitch_deg": float(np.mean(torso_pitch_deg)) if len(torso_pitch_deg) else None,
    }
