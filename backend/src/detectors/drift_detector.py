import numpy as np
from river.drift import ADWIN, PageHinkley, KSWIN


# Available detector types and their River class + default parameters
DETECTOR_REGISTRY = {
    "adwin": {
        "class": ADWIN,
        "default_params": {
            "delta": 0.002,     # significance level (lower = less sensitive, fewer false alarms)
            "clock": 32,        # how often ADWIN checks for change
            "max_buckets": 5,
            "min_window_length": 5,
            "grace_period": 10,
        },
    },
    "page_hinkley": {
        "class": PageHinkley,
        "default_params": {
            "min_instances": 30,     # minimum samples before detection starts
            "delta": 0.005,          # magnitude of allowed changes (tolerance)
            "threshold": 50,         # detection threshold (higher = less sensitive)
            "alpha": 1 - 0.0001,     # forgetting factor for moving average
        },
    },
    "kswin": {
        "class": KSWIN,
        "default_params": {
            "alpha": 0.005,         # significance level for KS test
            "window_size": 100,     # size of the sliding window
            "stat_size": 30,        # size of the recent window to compare
            "seed": 42,
        },
    },
}


class DriftDetector:
    """
    Wrapper for River drift detectors.

    Monitors a stream of values (typically prediction errors) and signals
    when concept drift is detected. Tracks all drift points for evaluation.
    """

    def __init__(self, method: str = "adwin", **kwargs):
        """
        :param method: detector type ("adwin", "page_hinkley", "kswin")
        :param **kwargs: override default parameters for the chosen detector
        """
        if method not in DETECTOR_REGISTRY:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(DETECTOR_REGISTRY.keys())}"
            )

        self.method = method
        registry_entry = DETECTOR_REGISTRY[method]

        # Merge default params with any user overrides
        params = {**registry_entry["default_params"], **kwargs}
        self.detector = registry_entry["class"](**params)

        # Tracking state
        self.step: int = 0
        self.drift_points: list[int] = []
        self.warning_points: list[int] = []
        self.error_history: list[float] = []

    def update(self, error: float) -> bool:
        """
        Feed a single error value to the detector.

        :param error: prediction error for the current time step (e.g. abs(actual - predicted))
        :return: True if drift is detected at this step
        """
        self.error_history.append(error)
        self.detector.update(error)

        drift_detected = False

        if self.detector.drift_detected:
            self.drift_points.append(self.step)
            drift_detected = True

        # Some detectors also support warning zones
        if hasattr(self.detector, "warning_detected") and self.detector.warning_detected:
            self.warning_points.append(self.step)

        self.step += 1
        return drift_detected

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        registry_entry = DETECTOR_REGISTRY[self.method]
        # Recreate with same params
        params = {**registry_entry["default_params"]}
        self.detector = registry_entry["class"](**params)
        self.step = 0
        self.drift_points = []
        self.warning_points = []
        self.error_history = []

    def get_drift_points(self) -> list[int]:
        """Return list of step indices where drift was detected."""
        return self.drift_points.copy()

    def get_summary(self) -> dict:
        """Return a summary of detection results."""
        return {
            "method": self.method,
            "total_steps": self.step,
            "num_drifts_detected": len(self.drift_points),
            "drift_points": self.drift_points.copy(),
            "num_warnings": len(self.warning_points),
        }


def evaluate_detector(
    detected_points: list[int],
    true_drift_points: list[int],
    tolerance: int = 100,
    total_steps: int = 0,
) -> dict:
    """
    Evaluate drift detection performance against known drift points.
    Useful for synthetic datasets where you know exactly when drift occurs.

    A detection is a True Positive if it falls within ±tolerance steps
    of a true drift point. Each true drift point can only be matched once.

    :param detected_points: step indices where detector flagged drift
    :param true_drift_points: actual drift point step indices
    :param tolerance: max distance between detected and true point to count as a match
    :param total_steps: total number of steps in the evaluation (for false alarm rate)
    :return: dict with TP, FP, FN, precision, recall, detection_delay metrics
    """
    matched_true = set()
    true_positives = 0
    detection_delays = []

    for det in sorted(detected_points):
        matched = False
        for j, true_pt in enumerate(true_drift_points):
            if j in matched_true:
                continue
            if abs(det - true_pt) <= tolerance:
                true_positives += 1
                matched_true.add(j)
                detection_delays.append(det - true_pt)
                matched = True
                break

    false_positives = len(detected_points) - true_positives
    false_negatives = len(true_drift_points) - true_positives

    precision = true_positives / max(len(detected_points), 1)
    recall = true_positives / max(len(true_drift_points), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (precision + recall) > 0 else 0.0

    avg_delay = float(np.mean(detection_delays)) if detection_delays else float("nan")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_detection_delay": avg_delay,
        "detection_delays": detection_delays,
    }