

Apple Stock Price Prediction using Linear Regression
====================================================

**Project by Karan Zaveri**  
_Email: [karanzaveri92@gmail.com](mailto:karanzaveri92@gmail.com)_  
_License: MIT License_

Project Description
-------------------

This project aims to predict Apple Inc.'s stock prices using historical stock market data. By leveraging the **Linear Regression** machine learning model, we predict the future stock price based on historical prices and key technical indicators like moving averages. This project demonstrates the fundamental steps involved in building a predictive machine learning model, from data preprocessing and feature engineering to model training, evaluation, and visualization of results.

Stock price prediction can be a complex task due to the highly volatile and non-linear nature of financial markets. However, using basic statistical methods like **Linear Regression**, we attempt to build a baseline model that can provide insights into future price trends.

### Purpose

The purpose of this project is to:

1.  Predict Apple stock prices using historical data and technical indicators.
2.  Demonstrate how to prepare stock market data for machine learning models.
3.  Provide a framework for evaluating the performance of predictive models using standard metrics such as **R-squared** and **Mean Absolute Error (MAE)**.
4.  Explore the residuals and understand how well the model performs.

### Key Steps in the Project:

*   **Data Collection**: Fetch historical stock data using the `yfinance` library.
*   **Feature Engineering**: Compute moving averages to capture trends in stock prices.
*   **Modeling**: Train a **Linear Regression** model to predict future stock prices.
*   **Evaluation**: Assess model performance using metrics like **R-squared** and **MAE**.
*   **Visualization**: Visualize predicted vs actual stock prices and residuals (errors).
*   **Prediction**: Make predictions for the next day's stock price.

* * *

Project Structure
-----------------

```
Apple Stock Price Prediction/
│
├── Apple Stock Price Prediction.ipynb  # Jupyter notebook containing the code and analysis
├── README.md                           # Project documentation
└── LICENSE                             # MIT license file
```

*   **`Apple Stock Price Prediction.ipynb`** : The main notebook where the stock price prediction is performed. This notebook includes data collection, preprocessing, modeling, evaluation, and visualization of results.
*   **`README.md`** : This file provides the project description, purpose, structure, and insights into the model’s performance.
*   **`LICENSE`** : The MIT license that governs the use of this project.

* * *

Project Workflow
----------------

1.  **Data Preprocessing**:  
    Historical stock price data for Apple Inc. was collected using the `yfinance` API. We retrieved closing prices and computed three moving averages:
    
    *   **10-day moving average (MA10)**
    *   **50-day moving average (MA50)**
    *   **100-day moving average (MA100)**
    
    These features are key indicators used in technical analysis of stock prices.
    
2.  **Model Training (Linear Regression)**:  
    We split the data into training and testing sets (80% train, 20% test) and scaled the features to improve the performance of the **Linear Regression** model. The model was trained on historical prices and moving averages to predict the next day’s stock price.
    
3.  **Model Evaluation**:  
    The model was evaluated using:
    
    *   **R-squared (R²)**: Measures how well the model explains the variance in the stock price data. A score close to 1 indicates a better fit.
    *   **Mean Absolute Error (MAE)**: Shows the average dollar difference between predicted and actual stock prices.
4.  **Prediction**:  
    We predicted the stock price for the next day using the trained model and calculated the difference between predicted and actual stock prices for the test set.
    
5.  **Visualization**:  
    The notebook includes plots of:
    
    *   Actual vs predicted stock prices over time.
    *   Residuals (errors) to inspect how well the model performed.

### Flowchart Summary:

```
Data Collection --> Data Preprocessing --> Train-Test Split --> Feature Scaling -->
Model Training --> Model Evaluation --> Prediction --> Visualization
```

This flow ensures the project follows a systematic approach to predicting stock prices and assessing the model's performance.

* * *

Code Explanation
----------------

### Data Collection

```python

import yfinance as yf
import pandas as pd

# Load Apple stock data
apple_stock = yf.Ticker("AAPL")
stock_data = apple_stock.history(period="max")
```

In this step, we fetch Apple's historical stock data using the `yfinance` API.

* * *

### Feature Engineering

```python

# Feature Engineering - Add Moving Averages as features
stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA100'] = stock_data['Close'].rolling(window=100).mean()
stock_data = stock_data.dropna()

# Create the target variable (next day's price)
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data = stock_data.dropna()
```

We create technical indicators by calculating the moving averages for different time periods. The target variable is the next day’s stock price.

* * *

### Model Training

```python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Splitting the data into train and test sets
X = stock_data[['Close', 'MA10', 'MA50', 'MA100']]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
```

Here, we split the data into training and testing sets, scale the features, and train a **Linear Regression** model on the training data.

* * *

### Model Evaluation and Metrics

```python

from sklearn.metrics import r2_score, mean_absolute_error

# Model predictions
lr_predictions = lr.predict(X_test_scaled)

# Evaluate model performance
r2 = r2_score(y_test, lr_predictions)
mae = mean_absolute_error(y_test, lr_predictions)
```

We calculate the model’s **R-squared** value and **Mean Absolute Error (MAE)** to assess its performance.

* * *

### Visualizing Predictions

```python
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'][-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(stock_data['Date'][-len(y_test):], lr_predictions, label='Predicted Prices', color='red')
plt.title('Apple Stock Price Prediction - Linear Regression')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
```

This section visualizes the actual vs predicted prices using Matplotlib. The plot helps in understanding how closely the predicted prices follow the actual stock prices.

* * *

Outputs
-------

### Key Model Outputs:

1.  **R-squared**:  
    Indicates how well the model explains the variance in the stock prices. A score closer to 1 means better accuracy.
    
2.  **Mean Absolute Error (MAE)**:  
    The average difference between the predicted and actual stock prices in dollar terms.
    
3.  **Next Day’s Predicted Stock Price**:  
    We predict the stock price for the next day based on the trained model and calculate how far off it is from the actual stock price.
    

* * *

Conclusion
----------

In this project, we built and evaluated a **Linear Regression** model to predict Apple Inc.'s stock prices. The model provided reasonably accurate predictions, with a mean absolute error in the range of a few dollars. The residual analysis and visual inspection indicate that the model generally captures the stock price trends but may struggle with periods of high volatility.

Further improvements could involve the use of more sophisticated models like **Random Forest** or **XGBoost**, as well as the incorporation of additional features such as trading volume, volatility indicators, or external sentiment analysis.

For any further questions or contributions, feel free to contact **Karan Zaveri** at _[karanzaveri92@gmail.com](mailto:karanzaveri92@gmail.com)_ .

* * *

License
-------

This project is licensed under the **MIT License**.
