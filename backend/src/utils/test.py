

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_model_history(
    history_str: str,
    title: str = "Active Model History",
    save_path: str = None,
) -> None:
    """
    Plot a Gantt-style chart where each model version is a row,
    and bars show the time steps it was active.

    Parameters
    ----------
    history_str : comma-separated string e.g. "v0,v0,v1,v1,v3,v3"
    title       : plot title
    save_path   : optional path to save the figure
    """
    history = [v.strip() for v in history_str.split(",")]
    n_steps = len(history)

    # Unique labels in order of first appearance
    unique_labels = list(dict.fromkeys(history))

    label_to_y = {lbl: i for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(unique_labels) * 0.8)))

    # Find contiguous runs and draw one bar per run
    t = 0
    while t < n_steps:
        label = history[t]
        start = t
        while t < n_steps and history[t] == label:
            t += 1
        duration = t - start
        ax.barh(
            label_to_y[label],
            duration,
            left=start,
            height=0.5,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )

    # Axes
    ax.set_xlim(0, n_steps)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(i + 1) for i in range(n_steps)], fontsize=12)
    ax.set_xlabel("Retrain Step", fontsize=12)

    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels, fontsize=12)
    ax.set_ylabel("Model Version", fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()  # v0 at top, latest at bottom
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    plt.close()





# Example
if __name__ == "__main__":
    history = "v0,v1,v0,v3,v1,v5,v6,v7,v5,v9,v11,v9,v13,v17"
    plot_model_history(history, title="ELM_kswin - linear_gradual_drift_5")