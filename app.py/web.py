import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


st.title('📈 Stock Price Prediction App💸💰📊📈')


st.markdown("""
<style>
    .input-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .input-container input {
        padding: 10px;
        font-size: 16px;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        margin-right: 10px;
    }
    .input-container button {
        padding: 10px 20px;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .input-container button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


user_input = st.text_input('Enter stock ticker', 'AAPL', key='ticker_input')

start_date = st.date_input('Select start date', value=pd.to_datetime('2015-01-01'))
end_date = st.date_input('Select end date', value=pd.to_datetime('2025-01-01'))

if st.button('Fetch Data'):
    st.session_state.fetch_data = True

if 'fetch_data' in st.session_state and st.session_state.fetch_data:
    with st.spinner('Fetching stock data...'):
        stock = yf.download(user_input, start=start_date, end=end_date)

    
    st.subheader('Data from {} to {}'.format(start_date, end_date))
    st.write(stock.describe())

   
    st.subheader('Closing Price vs Time Chart')
    m100 = stock['Close'].rolling(100).mean()
    m200 = stock['Close'].rolling(200).mean()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock['Close'], label='Closing Price', color='blue')
    plt.plot(m100, 'g', label='100 Day MA')
    plt.plot(m200, 'r', label='200 Day MA')
    plt.title('Closing Price vs Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    
    data_training = stock['Close'][0:int(len(stock) * 0.70)]
    data_testing = stock['Close'][int(len(stock) * 0.70):]

    
    data_training = data_training.values.reshape(-1, 1)

    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_scaled = scaler.fit_transform(data_training)

    
    model = load_model('traning model/keras_model.h5')

    
    past_100_days = data_training_scaled[-100:]

    
    final_stock = np.concatenate((past_100_days, scaler.transform(data_testing.values.reshape(-1, 1))), axis=0)

    
    x_test = []
    y_test = []

    for i in range(100, final_stock.shape[0]):
        x_test.append(final_stock[i-100:i])
        y_test.append(final_stock[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    
    y_pred = model.predict(x_test)

   
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    
    st.subheader('Predicted vs Original Prices')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_pred, 'r', label='Predicted Price')
    plt.title('Predicted vs Original Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    st.pyplot(fig2)

    
    st.markdown("---")
    st.markdown("SOHAM VALUNJKAR")
    st.markdown("This application uses historical stock data to predict future prices using a trained model.")