# LoL Pro Match Predictor

A machine learning project that predicts the outcome of professional League of Legends matches based on champion picks using Node2Vec graph embeddings.

## Overview

This project analyzes 2087 major region professional matches (LCK, LPL, LEC, LTA N, LTA S) from the 2025 season to predict match winners from team compositions alone. We compare one-hot encoding baselines against a proposed Node2Vec champion embedding approach.

## Dataset

Data sourced from [Oracle's Elixir](https://oracleselixir.com/tools/downloads). Download the 2025 match data CSV and place it in the root directory as `2025_LoL_esports_match_data_from_OraclesElixir.csv` before running the notebook.

## Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression (one-hot) | 49.7% |
| Naive Bayes (one-hot) | 49.4% |
| Random Forest (one-hot) | 48.7% |
| **GB + Node2Vec Embeddings** | **51.4%** |


## How to Run

1. Install dependencies:
pip install pandas numpy scikit-learn networkx node2vec streamlit

2. Run the notebook `lol_match_predictor.ipynb` end to end

3. To run the demo locally:

streamlit run app.py

