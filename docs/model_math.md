# Model Mathematics

This document details the mathematical formulations used in the AI Equity Analyst model.

## Technical Indicators

**Key Terms:**
*   **SMA_20 (Simple Moving Average)**: The unweighted mean of the previous 20 data points. It smoothes out price data to identify the trend direction.
*   **EMA (Exponential Moving Average)**: A type of moving average that places a greater weight and significance on the most recent data points. `EMA_12` refers to the 12-period EMA.

### Relative Strength Index (RSI)
The RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.

`RSI = 100 - [100 / (1 + RS)]`

Where:
`RS = Average Gain / Average Loss`

The average gain and loss are typically calculated over a 14-period window.

### Moving Average Convergence Divergence (MACD)
MACD is a trend-following momentum indicator.

`MACD = EMA_12(Close) - EMA_26(Close)`

The Signal Line is a 9-day EMA of the MACD line:
`Signal = EMA_9(MACD)`

### Bollinger Bands
Bollinger Bands consist of a middle band (N-period Simple Moving Average) and two outer bands.

`Middle Band = SMA_20(Close)`
`Upper Band = Middle Band + (2 * σ)`
`Lower Band = Middle Band - (2 * σ)`

Where `σ` (sigma) is the standard deviation of the Close price over the same period.

### Average True Range (ATR)
ATR measures market volatility. It is the moving average of the True Range (TR).

`TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)`
`ATR = SMA_14(TR)`

---

## Machine Learning Models

### Random Forest Classifier
A bagging ensemble method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees.

### XGBoost Classifier
An implementation of gradient boosted decision trees designed for speed and performance. It minimizes a regularized objective function:

`L(φ) = Σ l(y_hat, y) + Σ Ω(f)`

Where `l` is a differentiable convex loss function and `Ω` penalizes the complexity of the model (variance).

### Logistic Regression
A linear model for classification. It models the probability of the default class using the sigmoid function:

`P(Y=1|X) = 1 / (1 + e^-(β0 + β1*X1 + ... + βn*Xn))`

### Stacked Ensemble
Combines multiple classification models via a meta-classifier.
1.  **Base Learners**: Random Forest, XGBoost.
2.  **Meta Learner**: Logistic Regression.
The base learners' predictions are used as features for the meta learner.

---

## Deep Learning Models

### Long Short-Term Memory (LSTM)
A type of Recurrent Neural Network (RNN) capable of learning order dependence in sequence prediction problems.

The key components are the cell state (`C_t`) and the hidden state (`h_t`).

**Gates**:
- **Forget Gate**: `f_t = σ(W_f · [h_t-1, x_t] + b_f)`
- **Input Gate**: `i_t = σ(W_i · [h_t-1, x_t] + b_i)`
- **Candidate Layer**: `C̃_t = tanh(W_C · [h_t-1, x_t] + b_C)`
- **Output Gate**: `o_t = σ(W_o · [h_t-1, x_t] + b_o)`

**Update**:
`C_t = f_t * C_t-1 + i_t * C̃_t`
`h_t = o_t * tanh(C_t)`

### Gated Recurrent Unit (GRU)
A simplified version of LSTM with fewer parameters.

**Gates**:
- **Update Gate**: `z_t = σ(W_z · [h_t-1, x_t])`
- **Reset Gate**: `r_t = σ(W_r · [h_t-1, x_t])`
- **New Memory Content**: `h̃_t = tanh(W · [r_t * h_t-1, x_t])`

**Update**:
`h_t = (1 - z_t) * h_t-1 + z_t * h̃_t`

### Voting Ensemble
Combining predictions from multiple models to improve stability.

`P_final = (P_RF + P_XGB + P_LSTM) / 3`

The final prediction is based on a threshold (e.g., 0.5):
`y_hat = 1 if P_final > 0.5, else 0`
