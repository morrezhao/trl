# gsm8k_eval.py
import re
from typing import Optional


def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Extract the answer from model response.
    Supports multiple formats:
    - <answer>...</answer> tags
    - #### followed by number
    - The answer is: XXX
    - boxed{XXX}
    """
    # Try <answer>...</answer> format first
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Try #### format (GSM8K standard)
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", response)
    if hash_match:
        return hash_match.group(1).strip()

    # Try "The answer is" format
    answer_is_match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", response)
    if answer_is_match:
        return answer_is_match.group(1).strip()

    # Try boxed format (LaTeX)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed_match:
        return boxed_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> Optional[float]:
    """
    Normalize answer string to a numeric value.
    Handles:
    - Plain numbers: 42, -3.14
    - Numbers with commas: 1,234
    - Dollar amounts: $50
    - Percentages: 50%
    - Fractions in text: 1/2
    """
    if answer is None:
        return None

    # Remove common prefixes/suffixes
    answer = answer.strip()
    answer = re.sub(r"^[\$]", "", answer)  # Remove dollar sign
    answer = re.sub(r"[%]$", "", answer)   # Remove percent sign
    answer = answer.replace(",", "")        # Remove commas

    # Handle fractions like "1/2"
    frac_match = re.match(r"^(-?\d+)/(\d+)$", answer)
    if frac_match:
        num, denom = int(frac_match.group(1)), int(frac_match.group(2))
        if denom != 0:
            return num / denom

    # Try to extract number from string
    num_match = re.search(r"(-?\d+\.?\d*)", answer)
    if num_match:
        try:
            return float(num_match.group(1))
        except ValueError:
            return None

    return None


def extract_ground_truth(answer_text: str) -> Optional[float]:
    """
    Extract ground truth from GSM8K answer field.
    GSM8K format: "reasoning steps\n#### final_answer"
    """
    # GSM8K uses #### to mark the final answer
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", answer_text)
    if hash_match:
        return normalize_answer(hash_match.group(1).strip())

    # Fallback: try to get the last number in the answer
    numbers = re.findall(r"(-?\d+\.?\d*)", answer_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None

    return None


def verify_gsm8k_response(response: str, ground_truth: str) -> bool:
    """
    Verify if the model's response matches the ground truth answer.

    Args:
        response: Model's generated response
        ground_truth: The 'answer' field from GSM8K dataset

    Returns:
        True if the extracted answer matches the ground truth
    """
    # Extract predicted answer from response
    pred_answer = extract_answer_from_response(response)
    if pred_answer is None:
        return False

    pred_value = normalize_answer(pred_answer)
    if pred_value is None:
        return False

    # Extract ground truth value
    gt_value = extract_ground_truth(ground_truth)
    if gt_value is None:
        return False

    # Compare with tolerance for floating point
    # For integers, require exact match
    if gt_value == int(gt_value) and pred_value == int(pred_value):
        return int(pred_value) == int(gt_value)

    # For floats, use relative tolerance
    return abs(pred_value - gt_value) < 1e-6 * max(abs(gt_value), 1)
