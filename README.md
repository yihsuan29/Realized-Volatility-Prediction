# Realized Volatility Prediction
NYCU Data Mining Final Project - Realized Volatility Prediction <br>
This task is based on the Kaggle competition [**Optiver – Realized Volatility Prediction**](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview).
**Note:** As required by the Kaggle competition, all source code is implemented and executed within a **single Jupyter notebook**.


## Problem Definition
Given historical **limit order book (top 2 levels)** and **trade data**, predict the **realized volatility over the next 10 minutes** for each stock.


## Data
- **112 stocks**, 3,830 time buckets
- Data sources:
  - `train`: realized volatility (ground truth)
  - `book`: best bid/ask prices & sizes
  - `trade`: transaction price, volume, order count


## Method
1. **Feature Engineering**
   - Financial technical indicators from price, volume, and time
   - Multiple look-back windows
2. **Stock Clustering**
   - K-Means clustering to capture cross-stock common patterns
3. **Models**
   - LightGBM
   - Feedforward Neural Network (with stock ID embedding)
   - TabNet
4. **Ensemble**
   - Weighted averaging of model predictions


## Evaluation
- **Metric:** RMSPE (official Kaggle metric)
- **Validation:** 5-fold cross validation


## Results
- Best ensemble RMSPE: **0.2199**
- Achieved **Top 4%** on Kaggle private leaderboard
- Ensemble outperforms all single models


