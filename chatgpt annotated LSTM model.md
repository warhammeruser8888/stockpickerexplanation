# README: Understanding the Cleaned LSTM Stock Model

This guide explains **how the `CleanedLSTMStockModel.py` script works**, step by step, in plain English — no background in finance, Python, or machine learning required.

---

## 1. Purpose of the Script

This program predicts future stock prices using a type of **machine learning model** called a **Long Short-Term Memory (LSTM)** network. It takes historical stock data (like prices over time) and tries to learn patterns that can help forecast the next day’s price.

The model also evaluates its predictions against actual market performance and simulates a simple “buy/sell” trading strategy.

---

## 2. Importing Tools (Libraries)

The script begins by importing several Python **libraries**, which are like toolkits for specific jobs:

| Library | Purpose |
|----------|----------|
| `yfinance` | Downloads stock market data from Yahoo Finance |
| `pandas` | Organizes and manipulates large data tables easily |
| `numpy` | Handles math and number calculations efficiently |
| `matplotlib` | Draws charts and graphs |
| `scikit-learn` | Helps prepare data and measure prediction accuracy |
| `torch` (PyTorch) | Builds and trains deep learning models like LSTM |
| `scipy` | Performs advanced math (used here for portfolio optimization) |

---

## 3. Setting the Random Seed

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
This section ensures that results are **reproducible** — running the program twice with the same seed (42) will give the same results.

---

## 4. Downloading Stock Data

```python
def get_stock_data(ticker, start_date='2010-01-01'):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date)
    return stock_data
```

- A **ticker** is a stock symbol like “AAPL” for Apple.
- The function downloads historical stock prices (Open, Close, High, Low, Volume, etc.) starting from the given date.

---

## 5. Calculating Financial Ratios

```python
def calculate_ratios(stock_data):
    stock_data['P/E'] = stock_data['Close'] / stock_data['Close'].rolling(window=12).mean()
    stock_data['ROE'] = stock_data['Close'].pct_change() * 100
    stock_data['Debt-to-Equity'] = stock_data['Close'] / stock_data['Open']
    return stock_data
```

| Metric | Meaning | Purpose |
|---------|----------|----------|
| **P/E (Price-to-Earnings)** | Stock price relative to average past prices (a rough “value” indicator) | Detects whether a stock is expensive or cheap compared to its history |
| **ROE (Return on Equity)** | Percent change in price from one day to the next | Measures profitability |
| **Debt-to-Equity** | Ratio of opening and closing prices (placeholder metric) | Used as a rough stand-in for leverage |

---

## 6. Adding Technical Indicators

```python
def add_technical_indicators(stock_data):
    stock_data['12-day EMA'] = stock_data['Close'].ewm(span=12).mean()
    stock_data['26-day EMA'] = stock_data['Close'].ewm(span=26).mean()
    stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']
    ...
```
These are **technical analysis tools** used by traders:

| Indicator | Meaning | Purpose |
|------------|----------|----------|
| **EMA (Exponential Moving Average)** | Smooths recent prices | Highlights short-term trends |
| **MACD (Moving Average Convergence Divergence)** | Difference between two EMAs | Detects momentum (buy/sell signals) |
| **RSI (Relative Strength Index)** | Measures if a stock is “overbought” or “oversold” | Helps identify reversal points |

---

## 7. Creating the Prediction Target

```python
stock_data['Target'] = stock_data['Close'].shift(-1)
```
This line creates the **label** — tomorrow’s closing price. The model learns to predict this based on today’s features.

---

## 8. Creating Lag and Rolling Features

Lag features show past prices; rolling features show recent averages:

```python
for lag in range(1, 4):
    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)
stock_data['Rolling_mean_7'] = stock_data['Close'].rolling(window=7).mean()
stock_data['Rolling_std_7'] = stock_data['Close'].rolling(window=7).std()
```

This lets the model “see” price patterns over the past few days.

---

## 9. Cleaning and Scaling the Data

All missing values (`NaN`) and infinite values are filled or removed.  
Then, everything is **scaled** (normalized) using `RobustScaler`, which keeps outliers (extreme values) from breaking the model.

---

## 10. Splitting Data into Train, Validation, and Test Sets

The data is split as follows:

- **70%** → Training (used for learning)  
- **10%** → Validation (used to tune performance)  
- **20%** → Test (used for final evaluation)

---

## 11. Preparing the Dataset for the LSTM

A custom **Dataset class** is defined:

```python
class StockDataset(Dataset):
    ...
```
It converts the data into sequences of `sequence_length` (e.g., 60 days).  
Each sequence is used to predict the next day’s price.

---

## 12. Building the LSTM Model

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        ...
```

An **LSTM (Long Short-Term Memory)** model is a type of **Recurrent Neural Network (RNN)** that can “remember” information across time.  
It’s ideal for **time series** like stock prices because it understands trends over many days.

Key parts:
- **hidden_size:** how many “memory units” it uses
- **num_layers:** how deep the network is
- **bidirectional:** looks at patterns both forward and backward in time
- **dropout:** prevents overfitting (learning noise instead of real patterns)

---

## 13. Training the Model

```python
for epoch in range(num_epochs):
    model.train()
    ...
```
- The model predicts prices, compares them with actual prices, and adjusts its internal parameters.
- The **loss** (error) is reduced over time using the **Adam optimizer**.
- **Early stopping** ends training when no improvement is seen for a long time.

---

## 14. Testing and Evaluation

After training, the best model is reloaded and tested on unseen data.  
It calculates the **Mean Absolute Error (MAE)** — how far off the predictions are, on average.

Then, it plots:
- Actual prices vs. predicted prices  
- A backtest of a simple trading strategy: buy if prediction > current price, else sell.

---

## 15. Risk and Portfolio Analysis

### Value at Risk (VaR)
Measures how much loss could happen in a worst-case scenario (e.g., 5% worst outcomes).

### Portfolio Optimization
Uses math (`scipy.optimize`) to find the ideal mix of investments that minimizes risk for a given return.

---

## 16. Saving the Model

```python
torch.save(model.state_dict(), f"{ticker}_lstm_model.pth")
```
Saves the trained model so it can be reused later without retraining.

---

## 17. Key Takeaways

- This script demonstrates how **machine learning** can analyze and forecast stock price movements.  
- It uses real financial and technical indicators, but performance depends on market volatility and randomness.  
- The LSTM learns from past data patterns, but **it cannot guarantee future profits** — the stock market is inherently uncertain.

---

**End of README**  
This explanation is designed for non-programmers — you can now follow the logic of each major part of the code without needing prior technical knowledge.
