import numpy as np
from river.drift import ADWIN, PageHinkley, KSWIN


# Nested dictionary to store detector and thier default parametes values 
DETECTOR_REGISTRY = {
    "adwin": {
        "class": ADWIN,
        "default_params": {
            "delta": 4.0,     # significance level (lower = less sensitive, fewer false alarms)
        },
    },
    "page_hinkley": {
        "class": PageHinkley,
        "default_params": {
            "min_instances": 30,     # The minimum number of instances before detecting change
            "delta": 0.005,          # The delta factor for the Page-Hinkley test
            "threshold": 1,         # dThe change detection threshold (lambda).
            "alpha": 1 - 0.0001,     # The forgetting factor, used to weight the observed value and the mean
            "mode": "both" # Whether to consider increases ("up"), decreases ("down") or both ("both") when monitoring the fading mean.
        },
    },
    "kswin": {
        "class": KSWIN,
        "default_params": {
            "alpha": 0.0015,         # significance level for KS test
            "window_size": 200,     # size of the sliding window
            "stat_size": 60,        # size of the recent window to compare
            "seed": 42,
            "window" : None
        },
    },
}


class DriftDetector:
    """
    Wrapper for River drift detectors.

    Monitors a stream of values and signals when concept drift is detected.
    """

    def __init__(self, method:str = "adwin", **kwargs):
        """
        :param method: detector type ("adwin", "page_hinkley", "kswin")
        :param kwargs: overfide default parameters if needed for choosen detector 
        """
        if method not in DETECTOR_REGISTRY: 
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(DETECTOR_REGISTRY.keys())}"
            )
            
        self.method = method 
        registry_entry = DETECTOR_REGISTRY[method]
        
        # merge default params with any override from user 
        # Take everything from the efault param and override with any user provided params 
        params = {**registry_entry["default_params"], **kwargs}
        self.detector = registry_entry["class"](**params)
        self.custom_params = kwargs
        
        # Tracking state
        self.step: int = 0
        self.drift_points: list[int] = []
        self.warning_points: list[int] = []
        self.error_history: list[float] = []
    
    def reset(self) -> None:
        """Reset the detector to its initial state."""
        registry_entry = DETECTOR_REGISTRY[self.method]
        # Recreate with same params
        params = {**registry_entry["default_params"], **self.custom_params}
        self.detector = registry_entry["class"](**params)


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

        # ADWIN support early warning detection. 
        if hasattr(self.detector, "warning_detected") and self.detector.warning_detected:
            self.warning_points.append(self.step)

        self.step += 1
        return drift_detected


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
    tolerance: int = 300,
    total_steps: int = 0,
) -> dict:
    """
    Evaluate drift detection performance against known drift points.
    
    :param detected_points: step indices where detector flagged drift
    :param true_drift_points: actual drift point step indices
    :param tolerance: max distance between detected and true point to count as a match
    :param total_steps: total number of steps in the evaluation (for false alarm rate)
    :return: dict with TP, FP, FN, precision, recall, detection_delay metrics
    """
    matched_det = set()
    true_positives = 0
    detection_delays = []

    for j, true_pt in enumerate(true_drift_points):
        best_match = None
        best_dist = float("inf")
        for i, det in enumerate(detected_points):
            if i in matched_det:
                continue
            dist = abs(det - true_pt)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_match = i
        if best_match is not None:
            true_positives += 1
            matched_det.add(best_match)
            detection_delays.append(detected_points[best_match] - true_pt)

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