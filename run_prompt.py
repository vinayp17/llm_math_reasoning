import os
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import csv
from utils import Model
from prompt_loaders.prompt_dataset import load_prompts_by_dataset
from evaluators.evaluate_accuracy import evaluate_accuracy
from evaluators.summary import plot_accuracy_summary
from llm_interface.openai_model import OpenAIModel
from llm_interface.anthropic_model import AnthropicModel
from llm_interface.deepseek_model import DeepSeekModel
from llm_interface.open_weight_model import OpenWeightModel

def get_llm_instance(model: Model, api_keys: dict):
    if model.model_name.lower() == "gpt":
        return OpenAIModel(model.version, api_keys["openai"])
    elif model.model_name.lower() == "claude":
        return AnthropicModel(model.version, api_keys["anthropic"])
    elif model.model_name.lower() == "deepseek":
        return DeepSeekModel(
            model_name=model.version,
            api_key=api_keys["deepseek"],
            api_url=api_keys["deepseek_url"]
        )
    elif model.model_name.lower() == "qwen" or model.model_name.lower() == "llama":
        return OpenWeightModel(
            model_name=model.version,
        )
    else:
        raise ValueError(f"Unsupported model type: {model.model_name}")

# === Prompt Loader ===
def load_prompt_dataset(dataset_name: str, num_prompts: int):
    return load_prompts_by_dataset(dataset_name, n=num_prompts)


def evaluate_models_on_prompts(models, prompts, api_keys):
    results = []

    for model in models:
        llm = get_llm_instance(model, api_keys)
        print(f"\n🔍 Evaluating model: {model.model_name} ({model.version})")

        for p in tqdm(prompts, desc=f"{model.model_name}-{model.version}"):
            try:
                response = llm.query(p["prompt"])
                is_correct = evaluate_accuracy(response, p["answer"])
                status = "success"
            except Exception as e:
                response = str(e)
                is_correct = False
                status = "failure"

            results.append({
                "model_name": model.model_name,
                "version": model.version,
                "prompt" : p["prompt"],
                #"prompt_id": p["id"],
                "response": response,
                "correct": p["answer"],
                "is_correct": is_correct,
                "status": status
            })

    return results



def main(
    models_csv="LLM_Math_Evaluation_List.csv",
    dataset_name="aqua_rat",
    num_prompts=5,
    api_keys=None
):
    # Load models
    with open(models_csv, newline='') as f:
        reader = csv.DictReader(f)
        models = [Model(row["Category"], row["Model Name"], row["Version"], row["Notes"]) for row in reader]

    # Load prompts
    prompts = load_prompt_dataset(dataset_name, num_prompts)

    # Run evaluation
    results = evaluate_models_on_prompts(models, prompts, api_keys)

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"results_math_eval_{dataset_name}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Results saved to {output_path}")

    #Plot Summary
    #plot_accuracy_summary(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run math dataset evaluation")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., gsm8k, math_qa)"
    )

    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts to evaluate from the dataset"
    )

    args = parser.parse_args()

    print(f"Dataset selected: {args.dataset}")
    print(f"Number of prompts to evaluate: {args.num_prompts}")

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "deepseek_url": "https://api.deepseek.com/v1/chat/completions",
        #"qwen": os.getenv("QWEN_API_KEY"),
        #"qwen_url": "https://api.qwen.ai/v1/chat/completions"
        #"anthropic": "<your-anthropic-key>"
    }

    #If num_prompts is zero, let's read everything :)
    main(models_csv="small_eval_list.csv", api_keys=api_keys, num_prompts=None if not args.num_prompts else args.num_prompts, dataset_name=args.dataset)
