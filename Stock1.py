import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Title and description
st.title("Stock Price Prediction")
st.markdown("""
This web app predicts stock prices using an LSTM model. 
Upload a CSV file containing stock data, and the app will visualize and predict closing prices.
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Select stock symbol
    symbols = df['Symbol'].unique()
    Symbol = st.selectbox("Select Stock Symbol", symbols)

    # Filter data by stock symbol
    df = df[df['Symbol'] == Symbol]

    # Convert 'date' column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Extract closing prices
    data = df['Close'].values.reshape(-1, 1)

    # Split the data into train and test sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Scale the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Function to create data windows
    def create_data_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    # Set window size
    window_size = 10

    # Create training and testing datasets
    X_train, y_train = create_data_windows(train_data, window_size)
    X_test, y_test = create_data_windows(test_data, window_size)

    # Reshape data for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test Loss: {test_loss:.4f}")

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions and actual values
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = scaler.inverse_transform(predictions)

    # Plot the results
    st.subheader(f"{Symbol} Stock Price Prediction")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_actual, label='Actual')
    ax.plot(predictions_actual, label='Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'{Symbol} Stock Price Prediction')
    ax.legend()
    st.pyplot(fig)
