import re
from typing import Optional, List, Dict, Any


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from solution text."""
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'final answer[:\s]*\$?([^\n\$]+)\$?',
        r'answer is[:\s]*\$?([^\n\$]+)\$?',
        r'= ([0-9]+)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[\\${}]', '', answer)
            return answer.strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""

    answer = answer.lower().strip()
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'[\\${}]', '', answer)

    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return f"{num:.6f}".rstrip('0').rstrip('.')
    except ValueError:
        pass

    return answer


def evaluate_answer(predicted: str, expected: str) -> Dict[str, Any]:
    """Compare predicted answer against expected answer."""
    pred_answer = extract_answer(predicted)
    exp_answer = normalize_answer(expected)
    pred_normalized = normalize_answer(pred_answer) if pred_answer else ""

    is_correct = pred_normalized == exp_answer

    if not is_correct and pred_normalized and exp_answer:
        try:
            pred_num = float(pred_normalized)
            exp_num = float(exp_answer)
            is_correct = abs(pred_num - exp_num) < 1e-6
        except ValueError:
            pass

    return {
        "correct": is_correct,
        "predicted_raw": pred_answer,
        "predicted_normalized": pred_normalized,
        "expected_normalized": exp_answer,
    }


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from list of results."""
    if not results:
        return {}

    n_correct = sum(1 for r in results if r.get("correct", False))
    n_total = len(results)

    total_tokens = sum(r.get("metrics", {}).get("total_tokens", 0) for r in results)
    total_time = sum(r.get("metrics", {}).get("solve_time", 0) or r.get("metrics", {}).get("search_time", 0) for r in results)

    pruned = sum(r.get("metrics", {}).get("pruned_nodes", 0) for r in results)
    total_nodes = sum(r.get("metrics", {}).get("total_nodes", 0) for r in results)

    return {
        "accuracy": round(n_correct / n_total, 4) if n_total > 0 else 0,
        "n_correct": n_correct,
        "n_total": n_total,
        "avg_tokens": round(total_tokens / n_total, 2) if n_total > 0 else 0,
        "avg_time": round(total_time / n_total, 2) if n_total > 0 else 0,
        "total_tokens": total_tokens,
        "total_time": round(total_time, 2),
        "prune_rate": round(pruned / total_nodes, 4) if total_nodes > 0 else 0,
    }
