---
# file: _projects/Time Series.md
layout:      project
title:       Time Series Forecasting: A Comparative Study of ARIMA, ETS, and Ensemble Methods
date:        15 May 2023
image:
  path:       /assets/projects/Time Series.jpeg
  srcset:
    1920w:   /assets/projects/Time Series.jpeg
    960w:    /assets/projects/Time Series.jpeg
    480w:    /assets/projects/Time Series.jpeg
# caption:     Hyde is a brazen two-column Jekyll theme.
description: >
              Time series forecasting is an essential component of decision-making processes in various domains, ranging from finance and economics to healthcare and energy. In this article, we'll delve into 
              three popular forecasting techniques – ARIMA, ETS, and Ensemble methods – and evaluate their performance on a real-world dataset.
# links:
#  - title:   Demo
 #   url:     https://github.com/Zakaria-Alsahfi/zakariaalsahfi.github.io/blob/0961f50cde2b2a1350269f4194206c795398dffc/_posts/2019-05-15-womenclothing.md
#  - title:   Source
 #   url:     https://github.com/poole/hyde
featured:    false
tags: [Data Science, Machine Learning, Time Series, Python]
---


1. Introduction to Time Series Forecasting

Time series data is essentially a sequence of observations recorded at regular time intervals. Forecasting is the practice of predicting future values based on past observations. Time series forecasting has been foundational in several sectors, including stock market predictions, sales forecasting, and weather prediction.

2. Overview of Forecasting Methods

a. ARIMA (AutoRegressive Integrated Moving Average)

ARIMA is a combination of autoregressive (AR) and moving average (MA) models, alongside an integration component to make the data stationary. The "I" in ARIMA stands for "integrated" and represents the number of differences required to make the time series stationary.

**from statsmodels.tsa.statespace.sarimax import SARIMAX**

**# Fit ARIMA model**

**model = SARIMAX(data['Count'], order=(1, 1, 1), seasonal\_order=(1, 1, 1, 2))**

**results = model.fit(disp=-1)**

Key Components:

AR(p): Autoregression - a regression model that utilizes the dependent relationship between a current observation and a number of lagged observations.

I(d): Integrated - differencing of observations to make the time series stationary.

MA(q): Moving Average - a model that uses the dependency between a current observation and a residual error from a moving average model applied to lagged observations.

b. ETS (Error, Trend, Seasonality)

ETS models encapsulate the error, trend, and seasonality in a time series dataset.

**from statsmodels.tsa.holtwinters import ExponentialSmoothing**

**# Fit ETS model**

**model = ExponentialSmoothing(data['Count'], trend='add', seasonal='add', seasonal\_periods=12)**

**results = model.fit()**

Components:

Error: Can be additive or multiplicative.

Trend: Can be none, additive, or multiplicative.

Seasonality: Can be none, additive, or multiplicative.

c. Ensemble Methods

Ensemble methods combine multiple forecasting models to produce an overall forecast, often yielding better results than any individual model. A simple ensemble technique averages the forecasts from multiple models. More advanced techniques can weight models differently based on their performance.

**# Ensemble forecasting (average of ARIMA and ETS)**

**ensemble\_forecast = (arima\_forecast + ets\_forecast) / 2**

3. Comparative Analysis

Using a real-world dataset, we conducted a study to evaluate the performance of these models:

Results:

**ARIMA: RMSE=645.39, MAPE=16.80%**

**ETS: RMSE=783.58, MAPE=20.61%**

**Ensemble: RMSE=713.71, MAPE=18.70%**

Graph depicting forecasts of ARIMA, ETS, and Ensemble against actual data.

*Note: The graph showcases the historical data, and forecasts from ARIMA, ETS, and the Ensemble method.*

Interpretation:

ARIMA performed the best among the three models, showcasing the lowest RMSE and MAPE.

ETS, though a robust method, did not outperform ARIMA in this case.

The Ensemble method, which averaged the forecasts of ARIMA and ETS, provided a balanced performance.

4. Conclusion

While ARIMA outperformed the other methods in this study, the best forecasting method often depends on the specific dataset and problem at hand. It is crucial to test various models and sometimes even combine them to achieve the most accurate and reliable forecasts.

Remember, the real world is full of uncertainties, and while forecasting can provide a roadmap, it's essential to continuously update models with new data and ensure they remain relevant and accurate.
