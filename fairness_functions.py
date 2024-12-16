import pandas as pd
import numpy as np



def calculate_tpr_fpr(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr


def reweigh_data(X, y, sensitive_feature, target):
    # Create a dataframe combining features, sensitive feature, and target
    combined_df = pd.concat([X, y], axis=1)
    combined_df['sensitive_feature'] = sensitive_feature

    # Calculate weights
    group_target_counts = combined_df.groupby(['sensitive_feature', y.name]).size()
    total_counts = group_target_counts.sum()
    group_weights = total_counts / group_target_counts
    weights = combined_df.apply(
        lambda row: group_weights[(row['sensitive_feature'], row[y.name])], axis=1
    )
    return weights


def evaluate_fairness(y_true, y_pred, sensitive_feature):
    cond_satisfied = True

    # Combine true labels, predictions, and sensitive feature into a single DataFrame
    fairness_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_feature': sensitive_feature
    })

    print("\nFairness DataFrame:")
    print(fairness_df)

    # Calculate metrics for privileged and unprivileged groups
    privileged_group = fairness_df[fairness_df['sensitive_feature'] == 1]
    unprivileged_group = fairness_df[fairness_df['sensitive_feature'] == 0]

    # Statistical Parity: Difference in positive prediction rates
    p_privileged = privileged_group['y_pred'].mean()
    p_unprivileged = unprivileged_group['y_pred'].mean()
    statistical_parity_diff = p_unprivileged - p_privileged

    # Equal Opportunity: Difference in TPRs
    tpr_privileged = privileged_group[privileged_group['y_true'] == 1]['y_pred'].mean()
    tpr_unprivileged = unprivileged_group[unprivileged_group['y_true'] == 1]['y_pred'].mean()
    equal_opportunity_diff = tpr_unprivileged - tpr_privileged

    # Average Odds Difference: Average of TPR and FPR differences
    fpr_privileged = privileged_group[privileged_group['y_true'] == 0]['y_pred'].mean()
    fpr_unprivileged = unprivileged_group[unprivileged_group['y_true'] == 0]['y_pred'].mean()
    average_odds_diff = 0.5 * ((tpr_unprivileged - tpr_privileged) + (fpr_unprivileged - fpr_privileged))

    # Disparate Impact: Ratio of positive prediction rates
    disparate_impact = p_unprivileged / p_privileged if p_privileged > 0 else 0

    # Print results
    print("\nFairness Metrics:")
    print(f"Statistical Parity Difference: {statistical_parity_diff}")
    print(f"Equal Opportunity Difference: {equal_opportunity_diff}")
    print(f"Average Odds Difference: {average_odds_diff}")
    print(f"Disparate Impact: {disparate_impact}")

    with open("fairness_metrics.txt", "a") as file:
        file.write("\nFairness Metrics:\n")
        file.write(f"Statistical Parity Difference: {statistical_parity_diff}\n")
        file.write(f"Equal Opportunity Difference: {equal_opportunity_diff}\n")
        file.write(f"Average Absolute Odds Difference: {average_odds_diff}\n")
        file.write(f"Disparate Impact: {disparate_impact}\n")
        # file.write(f"Theil Index: {theil_index}")

    # Evaluate fairness level
    fairness_conditions = {
        "Statistical Parity": -0.10 <= statistical_parity_diff <= 0.10,
        "Equal Opportunity": -0.10 <= equal_opportunity_diff <= 0.10,
        "Average Absolute Odds": -0.10 <= average_odds_diff <= 0.15,
        "Disparate Impact": 0.80 <= disparate_impact <= 1.30,
        # "Theil Index": theil_index <= 0.25
    }

    for metric_name, condition in fairness_conditions.items():
        print(f"{metric_name} Fairness Condition Satisfied: {condition}\n")

        if (not condition):
            cond_satisfied = False

        with open("fairness_metrics.txt", "a") as file:
            file.write(f"{metric_name} Fairness Condition Satisfied: {condition}\n")

    if cond_satisfied:
        with open("fairness_metrics.txt", "a") as file:
            file.write(f"All conditions satisfied\n")
    else:
        with open("fairness_metrics.txt", "a") as file:
            file.write(f"Not all conditions satisfied\n")

    return {
        "Statistical Parity": statistical_parity_diff,
        "Equal Opportunity": equal_opportunity_diff,
        "Average Odds": average_odds_diff,
        "Disparate Impact": disparate_impact
    }, cond_satisfied
