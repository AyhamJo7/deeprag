import matplotlib.pyplot as plt
import numpy as np


def plot_retrieval_decisions(
    retrieval_steps: list[int], sequence_length: int, save_path: str
):
    """Plots when retrieval actions were taken during generation.

    Args:
        retrieval_steps: A list of timesteps where retrieval occurred.
        sequence_length: The total length of the generated sequence.
        save_path: Path to save the visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    # Create a timeline
    timeline = np.zeros(sequence_length)
    if retrieval_steps:
        timeline[retrieval_steps] = 1

    ax.eventplot(np.where(timeline)[0], orientation="horizontal", colors="b")
    ax.set_xlim(0, sequence_length)
    ax.set_yticks([])
    ax.set_ylabel("Generation Steps")
    ax.set_title("Retrieval Decisions Over Time")
    plt.savefig(save_path)
    plt.close(fig)
