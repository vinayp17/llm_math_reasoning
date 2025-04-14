import os
from utils import Model
from llm_interface.openai_model import OpenAIModel
from llm_interface.anthropic_model import AnthropicModel

def get_llm_instance(model: Model, api_keys: dict):
    if model.model_name.lower() == "gpt":
        return OpenAIModel(model.version, api_keys["openai"])
    elif model.model_name.lower() == "claude":
        return AnthropicModel(model.version, api_keys["anthropic"])
    else:
        raise ValueError(f"Unsupported model type: {model.model_name}")

if __name__ == "__main__":
    from csv import DictReader

    # Load models from CSV
    with open("LLM_Math_Evaluation_List.csv", newline='') as f:
        reader = DictReader(f)
        models = [Model(row["Category"], row["Model Name"], row["Version"], row["Notes"]) for row in reader]

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        #"anthropic": "<your-anthropic-key>"
    }

    prompt = "If x + 3 = 10, what is the value of x?"

    for m in models:
        try:
            llm = get_llm_instance(m, api_keys)
            answer = llm.query(prompt)
            print(f"\n🔍 {m.model_name} ({m.version}) replied:\n{answer}")
        except Exception as e:
            print(f"⚠️ Failed for {m.model_name} {m.version}: {e}")
