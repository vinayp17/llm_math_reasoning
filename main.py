#!/usr/bin/env python

from utils import load_models_from_csv

if __name__ == "__main__":
    models = load_models_from_csv("LLM_Math_Evaluation_List.csv")
    for m in models:
        print(m)
