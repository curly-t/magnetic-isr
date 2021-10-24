import matplotlib.pyplot as plt

from .plot_system_responses import plot_sr


def plot_comparison(responses, colors=None, markers=None):
    """Plots different groups of system responses with different colors, and markers.
    The function excpects the responses = [[sys_responses_1], [sys_responses_2], ...]
    And each [sys_responses_X] typically contains multiple system responses (SingleResults Class)."""

    if colors is None:
        colors = ['k']*len(responses)

    if markers is None:
        markers = ['o']*len(responses)

    fig = None
    ax_AR = None
    ax_phase = None
    for i, sys_responses in enumerate(responses):
        fig, ax_AR, ax_phase = plot_sr(sys_responses, color=colors[i], marker=markers[i], fig=fig, ax_AR=ax_AR, ax_phase=ax_phase)
    plt.show()
