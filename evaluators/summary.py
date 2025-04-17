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
