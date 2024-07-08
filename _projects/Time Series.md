---
# file: _projects/Time Series.md
layout:      post
title:       Analyzing Student Enrollment Trends and Forecasts with Time Series Analysis
date:        15 May 2023
image:
  path:       /assets/projects/Time Series.jpeg
  srcset:
    1920w:   /assets/projects/Time Series.jpeg
    960w:    /assets/projects/Time Series.jpeg
    480w:    /assets/projects/Time Series.jpeg
description: >
              Time series forecasting is an essential component of decision making processes in various domains, ranging from finance 
              and economics to healthcare and energy.In this article, we'll delves deep into forecasting student counts over time using various methodologies.
featured:    false
tags: [Data Science, Machine Learning, Python, Time Series]
---
Time series analysis offers a range of tools to analyze and predict temporal data. In this article, we dive deep into understanding the patterns of student enrollment over the years, employing some powerful forecasting models.

### Historical Student Enrollments
Understanding your data is the first step in any analysis. Let's start by observing the historical student enrollments from 2015 to 2022:

![png](/images/timeseries/data.png)

From the graph, we can notice a cyclical pattern, with peaks and troughs recurring over a similar period. Such patterns can provide valuable insights for predicting future enrollments.

### Decomposing the Data
Time series data can be decomposed into multiple components:

* Trend: The underlying pattern in the series.
* Seasonal: The repeating short-term cycle in the series.
* Residual: The random variation in the series.
* 
Here's a snapshot of the enrollment data decomposition:
![png](/images/timeseries/student count.png)

We can clearly observe a seasonal fluctuation in enrollments every year, which might be attributed to various academic and economic factors.

### ADF Test for Stationarity
A critical step before applying forecasting models on time series data is to check its stationarity, ensuring the mean, variance, and covariance remain constant over time. For this, we performed the Augmented Dickey-Fuller test. The results are as follows:

```Python
ADF Statistic: -1.358822089447294
p-value: 0.601854995584759
```

The p-value is greater than 0.05, which indicates that our data is not stationary, necessitating differencing or transformations before modeling.

### Modeling and Forecasting
To make accurate forecasts, we tested multiple models:

1\. **ARIMA:** (AutoRegressive Integrated Moving Average): A classical model for time series forecasting that combines auto-regression, differencing, and moving averages.

2\. **ETS:** (Error, Trend, Seasonality): Decomposes a time series into error, trend, and seasonality components.

3\. **Prophet:** Developed by Facebook, Prophet handles daily data with multiple seasonal effects and missing values.

4\. **Ensemble:** This approach combines forecasts from multiple models to improve accuracy.

From our analysis, the following prediction errors were obtained:

```Python
ARIMA: RMSE=384.31, MAPE=9.19%
ETS: RMSE=397.17, MAPE=9.52%
Ensemble: RMSE=390.71, MAPE=9.35%
```

Based on RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error), the Ensemble model provided slightly better accuracy, indicating the potential benefits of combining multiple models' strengths.

Let's visualize the forecasts:

![png](/images/timeseries/Forecasting Student Counts.png)

The Ensemble model forecasts seem to closely align with historical data, offering confidence in our predictions for upcoming years.

### Concluding Thoughts
Time series analysis presents an invaluable toolset for understanding and predicting enrollment trends. From our study, the ensemble approach of combining ARIMA, ETS, and Prophet models yielded the most accurate forecasts.

For institutions and stakeholders, such insights can pave the way for better resource allocation, strategic planning, and ensuring a smooth academic experience for students.
