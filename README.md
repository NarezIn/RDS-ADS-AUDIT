# Technical Audit of a random Loan Repayment Prediction ADS on Kaggle

**DS-UA 202, Responsible Data Science, Spring 2026**  
**Authors:** Brian Chen, Simon Ni

## Overview

This repository contains the technical audit of an Algorithmic Decision-Support System (ADS) originally submitted to the Kaggle competition [Predicting Loan Payback](https://www.kaggle.com/c/playground-series-s5e11). The ADS predicts whether a borrower will fully repay a loan using an ensemble of LightGBM and CatBoost models. Our audit evaluates the system not only on predictive performance, but also on fairness, interpretability, stability, and robustness.

## Key Findings

- The model achieves accuracy 0.906 and ROC-AUC 0.922, but exhibits a "loose approver" profile — correctly identifying 98% of repayers but missing 39% of true defaulters (FPR = 0.388).
- Formal fairness metrics (demographic parity, equal opportunity) pass across gender, marital status, and education level.
- SHAP and LIME analyses reveal that predictions are dominated by employment-status indicators, with conventional credit features contributing an order of magnitude less.
- The training data is synthetically generated, leaving external validity unverified.

## Audit Scope

- Accuracy analysis across demographic subpopulations
- Fairness analysis: demographic parity, equal opportunity, error rate analysis, intersectional analysis
- Global interpretability via SHAP
- Local interpretability via LIME
- Stability and robustness via Gaussian noise perturbation and calibration

## Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm catboost shap lime matplotlib seaborn
```

## Dataset

The dataset is sourced from the Kaggle competition [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11/data). Download and place `train.csv` and `test.csv` in the project root before running the notebooks.
