# Benchmarking ML Models for Quarterly Groundwater Level Forecasting in India

This repository contains the codebase and data pipelines for evaluating multiple machine learning models (including Random Forest, XGBoost, Chronos, LSTM, and GRU) on quarterly groundwater level forecasting across India.

## Structure
- `src/`: Core Python modules for data loading, preprocessing, sequence building, and feature engineering.
- `data/`: Raw and processed groundwater datasets.
- `outputs/`: Generated tables, figures, models, and publication-ready metrics.
- `run_cycle*.py`: Step-by-step experiment runners, from initial exploratory data analysis to final spatial cross-validation and feature ablation.

## Setup
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
