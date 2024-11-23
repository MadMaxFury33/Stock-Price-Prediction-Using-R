#Load Libraries
library(forecast)
library(tseries)
library(zoo)
library(ggplot2)

# Data preparation
colnames(AMZN) <- c("Date", "Close")
AMZN$Date <- as.Date(AMZN$Date)

# Handle missing values
AMZN$Close <- na.locf(AMZN$Close, na.rm = FALSE)

# Convert to time series 
stock_ts <- ts(AMZN$Close, frequency = 252)

# Check for stationarity and differencing loop
adf_test <- adf.test(stock_ts, alternative = "stationary")
differencing_iter <- 0
max_iterations <- 3
while (adf_test$p.value > 0.05 && differencing_iter < max_iterations) {
  stock_ts <- diff(stock_ts)
  adf_test <- adf.test(stock_ts, alternative = "stationary")
  differencing_iter <- differencing_iter + 1
}

if (differencing_iter == max_iterations) {
  cat("Maximum differencing iterations reached.\n")
}

# Train-test split
train_size <- floor(0.8 * length(stock_ts))
train_ts <- head(stock_ts, train_size)
test_ts <- tail(stock_ts, length(stock_ts) - train_size)

# Save last value 
last_train_value <- AMZN$Close[train_size]

# Build ARIMA model
arima_model <- auto.arima(train_ts, stepwise = TRUE, approximation = TRUE)
arima_forecast <- forecast(arima_model, h = length(test_ts))
arima_test <- as.numeric(arima_forecast$mean)

# Build ETS model
ets_model <- ets(train_ts)
ets_forecast <- forecast(ets_model, h = length(test_ts))
ets_test <- as.numeric(ets_forecast$mean)

# Transform predictions back to the original scale
arima_test_original <- cumsum(c(last_train_value, arima_test))[-1]  
ets_test_original <- cumsum(c(last_train_value, ets_test))[-1]      

# Accuracy Calculation
test_actual <- as.numeric(AMZN$Close[(length(AMZN$Close) - length(test_ts) + 1):length(AMZN$Close)])
arima_accuracy <- accuracy(arima_test_original, test_actual)
ets_accuracy <- accuracy(ets_test_original, test_actual)

cat("\nARIMA Model Accuracy:\n")
print(arima_accuracy)

cat("\nETS Model Accuracy:\n")
print(ets_accuracy)

# Forecast future stock prices (30 days)
future_days <- 30
future_arima_forecast <- forecast(arima_model, h = future_days)
future_ets_forecast <- forecast(ets_model, h = future_days)

arima_future_original <- cumsum(c(tail(AMZN$Close, 1), as.numeric(future_arima_forecast$mean)))[-1]
ets_future_original <- cumsum(c(tail(AMZN$Close, 1), as.numeric(future_ets_forecast$mean)))[-1]

# Combine all data
forecast_dates <- AMZN$Date[(length(AMZN$Date) - length(test_ts) + 1):length(AMZN$Date)]
future_dates <- seq(max(AMZN$Date) + 1, by = "day", length.out = future_days)

all_dates <- c(forecast_dates, future_dates)
all_actual <- c(test_actual, rep(NA, future_days))
all_arima <- c(arima_test_original, arima_future_original)
all_ets <- c(ets_test_original, ets_future_original)

# Create a final data frame
forecast_data <- data.frame(
  Date = all_dates,
  Actual = all_actual,
  ARIMA = all_arima,
  ETS = all_ets
)

# Print future predictions
cat("\nFuture Price Predictions (Original Scale):\n")
print(forecast_data[(nrow(forecast_data) - future_days + 1):nrow(forecast_data), ])

# Plotting
ggplot(forecast_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1, na.rm = TRUE) +
  geom_line(aes(y = ARIMA, color = "ARIMA Predicted"), size = 1) +
  geom_line(aes(y = ETS, color = "ETS Predicted"), size = 1) +
  labs(title = "Stock Price Prediction: Historical and Future Forecasts",
       y = "Stock Price", x = "Date") +
  scale_color_manual(values = c("Actual" = "black", 
                                "ARIMA Predicted" = "blue", 
                                "ETS Predicted" = "red"),
                     name = "Legend") +
  theme_minimal()

