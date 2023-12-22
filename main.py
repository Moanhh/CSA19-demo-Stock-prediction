import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go


# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning)


# Function to fetch historical stock data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to calculate Moving Average (MA) for a given window
def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to add Moving Average as a feature to the dataset
def add_moving_average_feature(data, window):
    data['MA'] = calculate_moving_average(data, window)
    return data

# Function to preprocess data for training the model
def preprocess_data(data, ma_window):
    data = add_moving_average_feature(data, ma_window)
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].apply(lambda x: x.toordinal())

    # Explicitly include the 'Close' column
    data['Close'] = data['Close']

    # Fill NaN values with 0
    data = data.fillna(0)

    return data

# Function to train a Random Forest Regressor model
def train_model(data):
    X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'MA']]
    y = data['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, scaler

# Function to predict the next day's closing price
def predict_price(model, last_date, scaler, open_price, high_price, low_price, volume):
    next_date = last_date + timedelta(days=1)
    next_date_ordinal = next_date.toordinal()
    input_data = np.array([[next_date_ordinal, open_price, high_price, low_price, volume, 0]])  # 0 is a placeholder for MA since it's not available for the next day
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]


# Streamlit UI
def main():

    # Sidebar for user input
    st.sidebar.header("User Input")
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(1990, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.now().date())
    ma_window = st.sidebar.slider("Moving Average Window", min_value=1, max_value=50, value=20)

    # Fetch historical stock data
    stock_data = get_stock_data(symbol, start_date, end_date)

    # Preprocess data
    processed_data = preprocess_data(stock_data, ma_window)

    # Train the model
    model, X_test, y_test, scaler = train_model(processed_data)

    # User Interface
    st.title(f"Stock Closing Price Predictor for {symbol}")

    st.markdown("---")

    # Prediction
    last_date = stock_data.index[-1].to_pydatetime().date()
    next_date = last_date + timedelta(days=1)
    open_price = stock_data['Open'].iloc[-1]
    high_price = stock_data['High'].iloc[-1]
    low_price = stock_data['Low'].iloc[-1]
    volume = stock_data['Volume'].iloc[-1]

    next_price = predict_price(model, last_date, scaler, open_price, high_price, low_price, volume)
    
    # Get the current day's closing price
    current_close_price = stock_data['Close'].iloc[-1]

    # Determine the color of the arrow based on predicted movement
    if next_price > current_close_price:
        arrow_color = 'green'  # Upward movement
    elif next_price < current_close_price:
        arrow_color = 'red'  # Downward movement
    else:
        arrow_color = 'black'  # No change

    # Display the arrow indicating the predicted movement
    arrow_html = f'<span style="color:{arrow_color}; font-size: 24px">&#8593;</span>' if arrow_color == 'green' else f'<span style="color:{arrow_color}; font-size: 24px">&#8595;</span>'

    # Display the predicted closing price and the directional arrow as a subheader
    next_date_str = next_date.strftime("%Y-%m-%d")
    prediction_text = f"### The predicted closing price for {next_date_str} is: **${next_price:.2f}** {arrow_html}"
    st.markdown(prediction_text, unsafe_allow_html=True)

    st.write("""
        The above value represents the predicted closing price for the next day, based on the trained model.
        The value is highlighted for emphasis.
    """)

    st.markdown("---")

    # Plot actual vs predicted closing prices with thinner lines
    st.subheader("Actual vs Predicted Closing Prices")
    plt.figure(figsize=(10, 8))

    # Plot predicted closing prices with thinner line
    predicted_prices = [predict_price(model, date.date(), scaler, stock_data['Open'].loc[date],
                                      stock_data['High'].loc[date], stock_data['Low'].loc[date],
                                      stock_data['Volume'].loc[date]) for date in processed_data.index]

    # Plot actual vs predicted closing prices
    fig_actual_vs_predicted = go.Figure()

    # Plot historical closing prices in blue
    fig_actual_vs_predicted.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], mode='lines', name='Historical Closing Price', line=dict(color='blue')))

    # Plot predicted closing prices in red
    fig_actual_vs_predicted.add_trace(go.Scatter(x=processed_data.index, y=predicted_prices, mode='lines', name='Predicted Closing Price', line=dict(color='red')))

    # Customize layout
    fig_actual_vs_predicted.update_layout(
        xaxis_title='Date',
        yaxis_title='Closing Price',
    )

    # Display the plot
    st.plotly_chart(fig_actual_vs_predicted)

    st.markdown("---")

    # Plot historical stock data
    st.subheader("Historical Stock Data")
    st.line_chart(stock_data['Close'])
    st.write("""
        This chart displays the historical closing prices of the selected stock over the specified time period.
    """)

    st.markdown("---")

    # Additional Time Series Analysis
    st.subheader("Time Series Analysis")

    # Plot Moving Average
    plt.figure(figsize=(10, 4))
    plt.plot(processed_data.index, processed_data['Close'], label="Closing Price", marker='o', linestyle='-')
    plt.plot(processed_data.index, processed_data['MA'], label=f"Moving Average ({ma_window} days)", linestyle='--', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Closing Price and Moving Average")
    plt.legend()
    st.pyplot(plt)
    st.write("""
        This chart shows the closing price of the stock along with a smoothed line called the Moving Average (MA).
        The MA helps to identify trends by reducing short-term fluctuations.
    """)

    st.markdown("---")


    # Plot Daily Price Changes
    st.subheader("Daily Price Changes")
    daily_changes = stock_data['Close'].pct_change()
    st.line_chart(daily_changes)
    st.write("""
        This chart illustrates the daily percentage changes in the closing prices of the stock. 
        It provides insights into the volatility and direction of price movements.
    """)

    st.markdown("---")


    # Plot Rolling Volatility
    st.subheader("Rolling Volatility")
    volatility_window = st.slider("Volatility Window", min_value=1, max_value=30, value=10)
    rolling_volatility = daily_changes.rolling(window=volatility_window).std()
    st.line_chart(rolling_volatility)
    st.write("""
        This chart displays the rolling volatility, which represents the degree of variation in daily price changes. 
        A higher rolling volatility indicates increased market uncertainty.
    """)

    st.markdown("---")

    # Display key components with a line separator
    st.subheader("Thank you for watching")
   
       

   
if __name__ == "__main__":
    main()
