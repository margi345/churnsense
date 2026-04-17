import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.data.loader import load_config


# Main evaluate function 
def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = None,
    monthly_revenue_per_customer: float = 65.0,
    retention_cost_per_customer: float = 10.0,
) -> dict:

    config = load_config()
    if threshold is None:
        threshold = config["model"]["threshold"]

    y_pred = (y_prob >= threshold).astype(int)

    # ── ML metrics 
    auc_roc   = roc_auc_score(y_true, y_prob)
    auc_pr    = average_precision_score(y_true, y_prob)
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    cm        = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # ── Business metrics 
    # Customers we correctly flagged as churners
    correctly_flagged   = int(tp)

    # Revenue saved if we retain correctly flagged churners
    # Assume 30% retention success rate from outreach campaign
    retention_rate      = 0.30
    customers_retained  = int(correctly_flagged * retention_rate)
    revenue_saved       = customers_retained * monthly_revenue_per_customer * 12

    # Cost of running retention campaign on all flagged customers
    total_flagged       = int(tp + fp)
    campaign_cost       = total_flagged * retention_cost_per_customer

    # Net business value
    net_value           = revenue_saved - campaign_cost

    # ── Optimal threshold search 
    best_threshold      = find_optimal_threshold(y_true, y_prob)

    metrics = {
        # ML metrics
        "auc_roc":            round(auc_roc, 4),
        "auc_pr":             round(auc_pr, 4),
        "f1_score":           round(f1, 4),
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "threshold_used":     threshold,

        # Confusion matrix
        "true_positives":     int(tp),
        "true_negatives":     int(tn),
        "false_positives":    int(fp),
        "false_negatives":    int(fn),

        # Business metrics
        "correctly_flagged_churners": correctly_flagged,
        "estimated_customers_retained": customers_retained,
        "estimated_revenue_saved_usd":  round(revenue_saved, 2),
        "campaign_cost_usd":            round(campaign_cost, 2),
        "net_business_value_usd":       round(net_value, 2),
        "optimal_threshold":            round(best_threshold, 4),
    }

    return metrics


# Find threshold that maximizes net business value 
def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    monthly_revenue: float = 65.0,
    retention_cost: float = 10.0,
    retention_rate: float = 0.30,
) -> float:

    best_value    = -np.inf
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        retained  = tp * retention_rate
        revenue   = retained * monthly_revenue * 12
        cost      = (tp + fp) * retention_cost
        net       = revenue - cost

        if net > best_value:
            best_value     = net
            best_threshold = t

    return best_threshold


# Pretty print metrics 
def print_metrics(metrics: dict) -> None:
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:     {metrics['auc_pr']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Threshold:  {metrics['threshold_used']}")
    print("\n--- Confusion Matrix ---")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("\n--- Business Impact ---")
    print(f"  Churners correctly flagged:   {metrics['correctly_flagged_churners']}")
    print(f"  Estimated customers retained: {metrics['estimated_customers_retained']}")
    print(f"  Estimated revenue saved:      ${metrics['estimated_revenue_saved_usd']:,.2f}")
    print(f"  Campaign cost:                ${metrics['campaign_cost_usd']:,.2f}")
    print(f"  Net business value:           ${metrics['net_business_value_usd']:,.2f}")
    print(f"  Optimal threshold:            {metrics['optimal_threshold']}")
    print("="*50 + "\n")