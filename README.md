# Simple Linear Regression prediction for trading markets
This was my first prediction model in Python for my Phoenix Cartel expert advisors, before I:
- moved to Metatrader historical data
- used more accurate Arima model (auto-regressive integrated moving average)
- started neural networks for self learning robots

Overall, you will reach an accuracy of 85-90% on next day (high, low, close) price, depending on your risk strategy.


!!! TRADE AT YOUR OWN RISKS !!!
I no longer use this code as I achieved better accuracy.


# How to use it

    from LinearRegressionForecast import LinearRegressionForecast

    forecast = LinearRegressionForecast("EURUSD", "2009-01-01", "2021-01-30")
    High = forecast.run_prediction("High")
    Low = forecast.run_prediction("Low")
    Close = forecast.run_prediction("Close")

    print(High[0], Low[0], Close[0])

