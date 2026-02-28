import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.data_utils.data_loader import load_synthetic_series
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_PATH = SCRIPT_DIR.parent.parent / "data" / "graph"
GRAPH_PATH.mkdir(parents=True, exist_ok=True)

save_path = GRAPH_PATH / "synthetic_drift_series.png"

CONCEPT_LENGTH = 2000
TOTAL_CONCEPTS = 10
DRIFT_POINTS = [CONCEPT_LENGTH * i for i in range(1, TOTAL_CONCEPTS)]

DRIFT_TYPES = [
    "linear_gradual_drift",
    "linear_abrupt_drift",
    "nonlinear_gradual_drift",
    "nonlinear_abrupt_drift",
]


def format_title(name: str) -> str:
    """Convert 'linear_gradual_drift' to 'Linear Gradual Drift'"""
    return name.replace("_", " ").title()


def plot_all_drift_types(
    series_number: int = 1,
    tick_interval: int = 2000,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # plot each drift type in a subplot. ravel flatten into 1d array 
    for ax, drift_name in zip(axes.ravel(), DRIFT_TYPES):

        series = load_synthetic_series(drift_name, series_number)

        # Plot series
        ax.plot(series, linewidth=0.5)

        # Plot drift points
        for i, dp in enumerate(DRIFT_POINTS):
            ax.axvline(
                dp,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Drift Point" if i == 0 else None
            )

        # X ticks every 2000
        ax.set_xticks(
            np.arange(
                0,
                CONCEPT_LENGTH * TOTAL_CONCEPTS + 1,
                tick_interval
            )
        )

        ax.set_title(format_title(drift_name), fontsize=15, y = -0.15)


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

        


if __name__ == "__main__":
    # 2x2 grid of all drift types
    plot_all_drift_types(series_number=1, save_path=save_path)
