import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate, selection_rate
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

# Paul Tol's high-contrast color-blind safe palette
colors = {
    'Baseline': '#332288',     # Blue
    'Reduction': '#EE7733',    # Vermilion
    'Threshold': '#117733',    # Teal
    'Elder Recall': '#882255', # Purple
    'Elder AUC': '#44AA99'     # Cyan
}

# Simulate predictions and labels
np.random.seed(0)
n = 500
y_true = np.random.randint(0, 2, size=n)
pred_baseline = np.random.binomial(1, 0.55, size=n)
pred_reduction = np.random.binomial(1, 0.58, size=n)
pred_threshold = np.random.binomial(1, 0.60, size=n)
age_bin = np.random.choice(['Young', 'Elder'], size=n, p=[0.6, 0.4])
income_bracket = np.random.choice(['Low', 'Mid', 'High'], size=n)
urbanicity = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n)

df = pd.DataFrame({
    'y_true': y_true,
    'pred_baseline': pred_baseline,
    'pred_reduction': pred_reduction,
    'pred_threshold': pred_threshold,
    'AGE_BIN': age_bin,
    'INCOME_BRACKET': income_bracket,
    'URBANICITY': urbanicity
})

# Fairness metrics
def compute_metrics(preds):
    return {
        'AGE_BIN': MetricFrame(metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'DP': selection_rate},
                               y_true=df['y_true'], y_pred=preds, sensitive_features=df['AGE_BIN']),
        'INCOME_BRACKET': MetricFrame(metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'DP': selection_rate},
                                      y_true=df['y_true'], y_pred=preds, sensitive_features=df['INCOME_BRACKET']),
        'URBANICITY': MetricFrame(metrics={'TPR': true_positive_rate, 'FPR': false_positive_rate, 'DP': selection_rate},
                                  y_true=df['y_true'], y_pred=preds, sensitive_features=df['URBANICITY'])
    }

metrics_dict = {
    'Baseline': compute_metrics(df['pred_baseline']),
    'Reduction': compute_metrics(df['pred_reduction']),
    'Threshold': compute_metrics(df['pred_threshold'])
}

fig1, axs = plt.subplots(1, 3, figsize=(18, 5))
groups = ['AGE_BIN', 'INCOME_BRACKET', 'URBANICITY']
metric_labels = ['TPR', 'FPR', 'DP']
width = 0.25

for idx, metric in enumerate(metric_labels):
    values = []
    for strat in ['Baseline', 'Reduction', 'Threshold']:
        gaps = [metrics_dict[strat][g].difference()[metric] for g in groups]
        values.append(gaps)
    x = np.arange(len(groups))
    axs[idx].bar(x - width, values[0], width, label='Baseline', color=colors['Baseline'])
    axs[idx].bar(x, values[1], width, label='Reduction', color=colors['Reduction'])
    axs[idx].bar(x + width, values[2], width, label='Threshold', color=colors['Threshold'])
    axs[idx].set_title(f'{metric} Gap Across Groups')
    axs[idx].set_xticks(x)
    axs[idx].set_xticklabels(groups)
    axs[idx].set_ylabel('Disparity')
    axs[idx].legend()

fig1.tight_layout()
fig1.savefig('Live_Figure_1_MetricFrame_Fairness.png')

# Elder Group Recall and AUC
strategies = ['Baseline', 'Reduction', 'Threshold']
elder_mask = df['AGE_BIN'] == 'Elder'
elder_recall = [
    recall_score(df['y_true'][elder_mask], df['pred_baseline'][elder_mask]),
    recall_score(df['y_true'][elder_mask], df['pred_reduction'][elder_mask]),
    recall_score(df['y_true'][elder_mask], df['pred_threshold'][elder_mask])
]
elder_auc = [
    roc_auc_score(df['y_true'][elder_mask], df['pred_baseline'][elder_mask]),
    roc_auc_score(df['y_true'][elder_mask], df['pred_reduction'][elder_mask]),
    roc_auc_score(df['y_true'][elder_mask], df['pred_threshold'][elder_mask])
]

x = np.arange(len(strategies))
width = 0.35
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.bar(x - width/2, elder_recall, width, label='Elder Recall', color=colors['Elder Recall'])
ax2.bar(x + width/2, elder_auc, width, label='Elder AUC', color=colors['Elder AUC'])
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.set_title('Elder Group: Recall vs AUC')
ax2.set_ylabel('Score')
ax2.legend()
fig2.tight_layout()
fig2.savefig('Live_Figure_2_Elder_Performance.png')

# Confusion Matrices
fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))
subgroups = ['Young', 'Elder']
strategies_dict = {
    'Baseline': df['pred_baseline'],
    'Threshold': df['pred_threshold']
}
titles = ['Young - Baseline', 'Young - Threshold', 'Elder - Baseline', 'Elder - Threshold']
idx = 0
for group in subgroups:
    mask = df['AGE_BIN'] == group
    for strat in ['Baseline', 'Threshold']:
        preds = strategies_dict[strat]
        cm = confusion_matrix(df['y_true'][mask], preds[mask])
        ax = axs3[idx//2][idx%2]
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        ax.set_title(titles[idx])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        idx += 1

fig3.suptitle('Figure 3. Disaggregated Confusion Matrices by Age Group and Strategy')
fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
fig3.savefig('Live_Figure_3_Confusion_Matrices.png')

def plot_reliability_by_group(df, y_col, pred_col, group_col, n_bins=6):
    from sklearn.calibration import calibration_curve
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss

    fig, ax = plt.subplots(figsize=(8, 6))

    for group in sorted(df[group_col].dropna().unique()):
        subset = df[df[group_col] == group]
        y_true = subset[y_col].values
        y_prob = subset[pred_col].values

        if len(set(y_true)) < 2:
            print(f"Skipping {group}: only one class present")
            continue

        # Use quantile binning to get evenly sized bins
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, marker='o', label=f'{group_col} = {group}')

        brier = brier_score_loss(y_true, y_prob)
        print(f"{group_col} = {group} | Brier Score: {brier:.3f} | Count = {len(y_true)}")

    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
    ax.set_title(f'Figure A.4: Calibration Curve by {group_col}')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'Live_Figure_4_Calibration_Curves_{group_col}.png')
    plt.show()
    print(df['pred_threshold'].describe())
    print("Overall predicted threshold stats:\n", df['pred_threshold'].describe())
    print("\nGrouped by AGE_BIN:\n", df.groupby('AGE_BIN')['pred_threshold'].describe())

# Example usage
plot_reliability_by_group(
    df=df,
    y_col='y_true',
    pred_col='pred_threshold',
    group_col='AGE_BIN',
    n_bins=500
)

print("Figures saved: Live_Figure_1_MetricFrame_Fairness.png, Live_Figure_2_Elder_Performance.png, Live_Figure_3_Confusion_Matrices.png, Live_Figure_4_Calibration_Curves.png")
