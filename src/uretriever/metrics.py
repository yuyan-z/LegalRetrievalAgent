from collections.abc import Sequence


def citation_f1(
    pred: Sequence[str],
    gold: Sequence[str],
) -> dict[str, float]:
    """Compute F1 score for citations of a single query.

    Args:
        pred: List of predicted canonical citation IDs
        gold: List of ground truth canonical citation IDs

    Returns:
        Dictionary with precision, recall, and F1
    """
    pred_set = set(pred)
    gold_set = set(gold)

    # both empty
    if len(pred_set) == 0 and len(gold_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # prediction empty but gold not
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # gold empty but prediction not
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    TP = len(pred_set & gold_set)      # true_positives: number of correctly predicted citations
    precision = TP / len(pred_set)
    recall = TP / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(
    preds: Sequence[Sequence[str]],
    golds: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Macro F1: average F1 across all queries.

    Args:
        preds: List of predicted citation lists (one per query)
        golds: List of gold citation lists (one per query)

    Returns:
        Dictionary with macro precision, recall, and F1
    """
    if len(preds) != len(golds):
        raise ValueError(f"Length mismatch: {len(preds)} predictions vs {len(golds)} gold")

    if len(preds) == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, gold in zip(preds, golds):
        scores = citation_f1(pred, gold)
        precision_scores.append(scores["precision"])
        recall_scores.append(scores["recall"])
        f1_scores.append(scores["f1"])

    n = len(f1_scores)
    return {
        "macro_precision": sum(precision_scores) / n,
        "macro_recall": sum(recall_scores) / n,
        "macro_f1": sum(f1_scores) / n,
    }


def average_precision(
    ranked_pred: Sequence[str],
    gold: Sequence[str],
) -> float:
    """Compute Average Precision for a single query.

    Args:
        ranked_pred: Ranked list of predicted citations (most relevant first)
        gold: Set of gold citations (order doesn't matter)

    Returns:
        Average Precision score
    """
    gold_set = set(gold)

    if not gold_set:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, pred in enumerate(ranked_pred, 1):
        if pred in gold_set:
            hits += 1
            precision_sum += hits / i

    return precision_sum / len(gold_set)


def mean_average_precision(
    ranked_preds: Sequence[Sequence[str]],
    golds: Sequence[Sequence[str]],
) -> float:
    """Compute Mean Average Precision (MAP) across all queries.
    Use this metric when citation ranking matters (not just set overlap).

    Args:
        predictions_ranked: List of ranked predictions per query
        gold: List of gold citations per query

    Returns:
        Mean Average Precision score
    """
    if len(ranked_preds) != len(golds):
        raise ValueError(f"Length mismatch: {len(ranked_preds)} predictions vs {len(golds)} gold")

    if len(ranked_preds) == 0:
        return 0.0

    ap_scores = []
    for ranked_pred, gold in zip(ranked_preds, golds):
        ap = average_precision(ranked_pred, gold)
        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores)
