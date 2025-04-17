import re

def extract_numeric_answer(text: str) -> float:
    """
    Extract the first numeric value after 'The answer is'.
    """
    match = re.search(r"The answer is\s+([\-+]?\d*\.?\d+)", text.strip(), re.IGNORECASE)
    if match:
        return float(match.group(1))
    raise ValueError(f"No numeric value found in text: {text}")

def evaluate_accuracy(response: str, correct_answer: str) -> bool:
    """
    Returns True if the numeric values match exactly.
    """
    try:
        pred = extract_numeric_answer(response)
        truth = float(correct_answer)
        print(pred, truth)
        return int(round(pred)) == int(round(truth))
    except Exception:
        return False
