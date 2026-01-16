import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def save_results_csv(measurements, known_length_mm, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "measurement_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "measured_mm", "error_mm"])

        for i, m in enumerate(measurements):
            writer.writerow([i + 1, m, m - known_length_mm])

    print("Saved CSV:", csv_path)


def plot_error_histogram(measurements, known_length_mm, out_dir):
    errors = measurements - known_length_mm

    plt.figure()
    plt.hist(errors, bins=10)
    plt.xlabel("Measurement Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Measurement Error Distribution")

    path = os.path.join(out_dir, "error_histogram.png")
    plt.savefig(path)
    plt.close()

    print("Saved plot:", path)


def plot_repeatability(measurements, known_length_mm, out_dir):
    mean = np.mean(measurements)
    std = np.std(measurements)

    upper_3s = mean + 3 * std
    lower_3s = mean - 3 * std

    plt.figure()
    plt.plot(measurements, marker="o", label="Measured")
    plt.axhline(known_length_mm, linestyle="--", label="True Value")
    plt.axhline(upper_3s, linestyle=":", label="+3σ")
    plt.axhline(lower_3s, linestyle=":", label="-3σ")

    plt.xlabel("Sample Index")
    plt.ylabel("Measurement (mm)")
    plt.title("Repeatability Analysis")
    plt.legend()

    path = os.path.join(out_dir, "repeatability_plot.png")
    plt.savefig(path)
    plt.close()

    print("Saved plot:", path)
