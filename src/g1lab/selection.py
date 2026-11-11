"""Conservative multi-task checkpoint selection for local pilots."""


def score(metrics):
    # Equal task weighting; failed episodes also incur their measured tracking error.
    return sum(m["survival_rate"] - m["velocity_rmse_mps"]["mean"] for m in metrics.values()) / len(metrics)


def acceptable(candidate, reference, yaw_tolerance=None):
    return (
        all(candidate[t]["survival_rate"] >= reference[t]["survival_rate"] for t in reference)
        and all(
            candidate[t]["velocity_rmse_mps"]["mean"] <= reference[t]["velocity_rmse_mps"]["mean"] + 0.03
            for t in reference
        )
        and score(candidate) > score(reference)
        and (
            yaw_tolerance is None
            or all(
                candidate[t]["yaw_rmse_radps"]["mean"]
                <= reference[t]["yaw_rmse_radps"]["mean"] + yaw_tolerance
                for t in reference
            )
        )
    )
