import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from train_simple_mlp import PlotGeneratorCallback


def regenerate_plots_for_run(run_name):
    """Regenerates plots for a given training run."""
    log_dir = "./logs"
    run_log_dir = os.path.join(log_dir, run_name)

    if not os.path.isdir(run_log_dir):
        print(f"Error: Log directory not found at {run_log_dir}")
        return

    print(f"Attempting to regenerate plots for: {run_name}")

    # Instantiate the callback and call the plotting method directly.
    # The plotting method only needs the log path and model name.
    plot_callback = PlotGeneratorCallback(
        log_path=run_log_dir, model_name=run_name, verbose=1
    )

    # Manually trigger the plot generation.
    plot_callback._generate_plots()

    print("Plot regeneration process finished.")


if __name__ == "__main__":
    # The user specified which run's plot is broken.
    run_to_fix = "TD3_tf_agents_style_heavy_STRUCTURED_CREDIT_20250722_080146"

    # Allow overriding via command line argument for future use.
    if len(sys.argv) > 1:
        run_to_fix = sys.argv[1]

    regenerate_plots_for_run(run_to_fix) 