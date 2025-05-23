import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_summary(results_csv):
    df = pd.read_csv(results_csv)
    summary = df.groupby(["model_name", "version"])["is_correct"].mean().reset_index()
    summary["accuracy (%)"] = summary["is_correct"] * 100

    plt.figure(figsize=(10, 6))
    plt.barh(
        summary["model_name"] + " " + summary["version"],
        summary["accuracy (%)"]
    )
    plt.xlabel("Accuracy (%)")
    plt.title("LLM Math Evaluation Accuracy on AQuA-RAT")
    plt.tight_layout()
    plt.show()

def plot_accuracy_summary_llm(results_csv):
    df = pd.read_csv(results_csv)
    df['llm_judgement'] = df['llm_judgement'].map({'True': True, 'False': False})
    summary = df.groupby(["model_name", "version"])["llm_judgement"].mean().reset_index()
    summary["accuracy (%)"] = summary["llm_judgement"] * 100

    plt.figure(figsize=(10, 6))
    plt.barh(
        summary["model_name"] + " " + summary["version"],
        summary["accuracy (%)"]
    )
    plt.xlabel("Accuracy (%)")
    plt.title("LLM Math Evaluation Accuracy on AQuA-RAT")
    plt.tight_layout()
    plt.show()

