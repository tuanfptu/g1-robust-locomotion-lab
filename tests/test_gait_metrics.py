import numpy as np

from g1lab.gait_metrics import contact_metrics, swing_kinematics


def _running_pattern(cycles=6, stance=10, flight=5):
    """Alternating stance with flight between steps: L stance, flight, R stance, flight, ..."""
    rows = []
    for _ in range(cycles):
        rows += [(1, 0)] * stance + [(0, 0)] * flight + [(0, 1)] * stance + [(0, 0)] * flight
    return np.array([(0, 0)] * 3 + rows + [(0, 0)] * 3, dtype=bool)


def test_running_pattern_has_flight_and_forefoot_strikes():
    contact = _running_pattern()
    toe = contact.copy()
    heel = np.zeros_like(contact)
    # heels come down after the first sample of each stance -> still a forefoot strike
    for foot in range(2):
        starts = np.flatnonzero(np.diff(np.concatenate([[0], contact[:, foot].astype(int)])) == 1)
        heel[:, foot] = contact[:, foot]
        heel[starts, foot] = False
    force = np.where(contact.any(axis=1), 100.0, 0.0)
    m = contact_metrics(contact, toe, heel, force, dt=0.01, body_weight=350.0)
    assert m["double_support_fraction"] == 0
    assert abs(m["median_flight_time_s"] - 0.05) < 1e-9
    assert abs(m["median_stance_time_s"] - 0.10) < 1e-9
    assert abs(m["median_swing_time_s"] - 0.20) < 1e-9
    assert m["forefoot_strike_fraction"] == 1.0 and m["heel_strike_fraction"] == 0.0
    assert abs(m["median_peak_forward_grf_bw"] - 100 / 350) < 1e-9
    assert 0.3 < m["flight_fraction"] < 0.4


def test_walking_pattern_has_no_flight_and_heel_strikes():
    rows = []
    for _ in range(6):
        rows += [(1, 1)] * 3 + [(1, 0)] * 10 + [(1, 1)] * 3 + [(0, 1)] * 10
    contact = np.array(rows, dtype=bool)
    heel = contact.copy()
    toe = np.zeros_like(contact)
    m = contact_metrics(contact, toe, heel, np.zeros(len(contact)), dt=0.01, body_weight=350.0)
    assert m["flight_fraction"] == 0 and m["median_flight_time_s"] is None
    assert m["double_support_fraction"] > 0.2
    assert m["heel_strike_fraction"] == 1.0


def test_swing_kinematics_counter_swing_correlation():
    t = np.linspace(0, 4 * np.pi, 400)
    hip = np.stack([np.sin(t), -np.sin(t)], axis=1)
    shoulder = -0.8 * hip
    contact = np.stack([np.sin(t) > 0, np.sin(t) <= 0], axis=1)
    knee = np.full((400, 2), 0.44) - 0.2 * np.abs(hip)
    ankle = -0.3 * np.abs(hip)
    k = swing_kinematics(contact, knee, ankle, shoulder, hip, np.full(400, 8.0))
    assert all(c < -0.99 for c in k["shoulder_hip_same_side_correlation"])
    assert k["median_swing_min_knee_drop_m"] < 0.25
    assert k["median_swing_peak_dorsiflexion_rad"] < -0.25
    assert k["mean_torso_pitch_deg"] == 8.0
