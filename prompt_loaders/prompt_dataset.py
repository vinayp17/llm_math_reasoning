from datasets import load_dataset
import re

DEFAULT_PREFIX = "Please answer the following question in a single line ony similar to: The answer is "

def load_prompts_by_dataset(name: str, n=None, prefix: str = DEFAULT_PREFIX):
    if name == "aqua_rat":
        return load_aqua_rat_prompts(n, prefix)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def extract_numeric_from_option(option_text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", option_text)
    if match:
        return float(match.group(0))
    raise ValueError(f"No numeric value found in option: {option_text}")

def load_aqua_rat_prompts(n=None, prefix: str = DEFAULT_PREFIX):
    ds = load_dataset("aqua_rat", split="validation")
    prompts = []

    for item in ds.select(range(n)) if n else ds:
        q = item["question"]
        correct_option_letter = item["correct"]  # e.g., "C"
        options = item["options"]  # list of strings like ["A) 20", "B) 30", ...]

        # Find the correct option text
        correct_option_text = None
        for opt in options:
            if opt.strip().startswith(correct_option_letter):
                correct_option_text = opt
                break

        if not correct_option_text:
            raise ValueError(f"Correct option '{correct_option_letter}' not found in options: {options}")
        try:
            numeric_answer = extract_numeric_from_option(correct_option_text)

            full_prompt = f"{prefix}\n{q.strip()}"
            prompts.append({
                #"id": item["id"],
                "prompt": full_prompt,
                "answer": str(numeric_answer)
            })
        except Exception:
            continue
    return prompts
