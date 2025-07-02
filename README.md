# Stock Price Prediction using K-Nearest Neighbors (KNN)

This project explores the application of the K-Nearest Neighbors (KNN) algorithm in two scenarios:

1. **Classification**: Predicting whether to **buy** (+1) or **sell** (–1) a stock, based on simple daily price‐derived features.  
2. **Regression**: Predicting the **next‑day closing price** itself.

By leveraging KNN, a straightforward yet powerful supervised learning method, we demonstrate how even basic features can yield actionable insights in a financial context.

---

## 📖 Project Overview

### Objective

- **Classification:** Generate a buy/sell signal by comparing today’s close price to tomorrow’s.  
- **Regression:** Forecast tomorrow’s closing price directly.

### Why KNN?

- **Intuitive**: decisions based on “nearest” historical days.  
- **Versatile**: same algorithm can classify (discrete labels) or regress (continuous values).  
- **Minimal assumptions**: no underlying distributional hypotheses.

---

## 📦 Dataset

- **Source**: National Stock Exchange (NSE) of India — downloaded as CSV from the **Historical Data** section.  
- **Ticker**: TATA Consumer Products Ltd. (formerly TATAGLOBAL)  
- **Time span**: as far back as available (e.g. 2010–2025)  
- **Columns** include:
  - `Date`  
  - `Open`, `High`, `Low`, `Close`  
  - (others like `Volume`, `Turnover`, etc., which aren’t used here)

---

## 🛠 Approach

### 1. Data Preparation

- **Load & clean** the CSV (`<CSV_FILENAME>.csv`).  
- **Ensure numeric** types for `Open`, `High`, `Low`, `Close`.  
- **Sort** by date and drop rows with missing prices.

### 2. Feature Engineering

- **Open–Close** = `Open – Close`  
- **High–Low**   = `High – Low`  
- These two simple “range” features capture daily volatility.

### 3. Classification: Buy/Sell Signal

1. **Target**: label = **+1** if `Close(t+1) > Close(t)`, else **–1**.  
2. **Train/Test Split**: last 25% of days for test (no shuffle).  
3. **KNN Classifier** (`KNeighborsClassifier`):  
   - Hyperparameter: `n_neighbors` via `GridSearchCV` (2–15).  
4. **Evaluation**: report train & test accuracy (%) and show sample predictions.

### 4. Regression: Price Forecast

1. **Target**: `Close(t)` itself.  
2. Same **features**, **split**, and **grid search** over `n_neighbors`.  
3. **Evaluation**: compute **RMSE** (root mean square error) and show sample actual vs predicted prices.  
4. **Visualization**: line‑plot of actual vs predicted closing prices on the test set.

---

## 📂 Repository Structure

