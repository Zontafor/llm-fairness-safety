import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure reproducibility
np.random.seed(42)

# Simulate fairness metrics data (as referenced in insurance_model.ipynb)
def plot_fairness_metrics():
    metrics = ['Elder TPR', 'Young/Elder TPR Ratio', 'AUC', 'Recall Disparity']
    pre = [0.27, 2.57, 0.823, 0.22]
    post = [0.44, 1.22, 0.811, 0.09]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, pre, width, label='Pre-Mitigation')
    ax.bar(x + width/2, post, width, label='Post-Mitigation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=10)
    ax.set_ylabel("Metric Value")
    ax.set_title("Fairness Metrics Before vs After Mitigation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figure1_fairness_metrics.png")
    plt.close()

# Simulate regression interval width data (from residual calibration)
def plot_interval_widths():
    age_bins = ['Young', 'Mid-Age', 'Senior', 'Elder']
    pre = [1200, 1800, 2400, 3100]
    post = [1100, 1500, 1700, 2000]

    x = np.arange(len(age_bins))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, pre, width, label='Pre-Calibrated')
    ax.bar(x + width/2, post, width, label='Post-Calibrated')
    ax.set_xticks(x)
    ax.set_xticklabels(age_bins)
    ax.set_ylabel("95% Prediction Interval Width")
    ax.set_title("Interval Widths by Age Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figure2_interval_widths.png")
    plt.close()

# Simulate DiCE recourse accessibility (used in xai_counterfactual_dice.py)
def plot_recourses_by_income():
    income_groups = ['<30k', '30k-60k', '60k-90k', '90k+']
    recourses = [1.2, 1.8, 2.3, 3.0]

    fig, ax = plt.subplots()
    sns.barplot(x=income_groups, y=recourses, palette='Blues_d')
    ax.set_ylabel("Average Feasible Recourse Count")
    ax.set_title("DiCE Recourse Availability by Income Group")
    plt.tight_layout()
    plt.savefig("figure3_dice_recourse_by_income.png")
    plt.close()

if __name__ == "__main__":
    plot_fairness_metrics()
    plot_interval_widths()
    plot_recourses_by_income()