import pandas as pd
import os
import argparse
from tqdm import tqdm
from llm_interface.openai_model import OpenAIModel

def process_csv_with_llm(csv_path: str, output_path: str, llm: OpenAIModel):
    df = pd.read_csv(csv_path)

    judgments = []
    for idx, row in tqdm(df.iterrows(), desc="Querying LLM"):
        response = str(row["response"]).strip()
        correct = str(row["correct"]).strip()

        if not response or not correct:
            judgments.append("Skipped - Empty field")
            continue

        prompt = (
            f"Are these two answers the same?\n\n"
            f"Response: {response}\n"
            f"Correct: {correct}\n"
            f"Answer with True or False only."
        )

        verdict = llm.query(prompt)
        judgments.append(verdict)

    df["llm_judgment"] = judgments
    df.to_csv(output_path, index=False)
    print(f"LLM judgments saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare LLM responses to correct answers in CSV.")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    llm = OpenAIModel(model_name="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    process_csv_with_llm(args.input_csv, args.output_csv, llm)


if __name__ == "__main__":
    main()
