import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

try:
    model = load_model(r'C:\Users\Lakshmipathi Rao\Desktop\Stock\Stock market predictions model.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

past_100_days_of_train_data = data_train.tail(100)
data_test = pd.concat([past_100_days_of_train_data, data_test], ignore_index=True)
data_test_scale = sc.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

if 'model' in locals():
    pred = model.predict(x)
    scale = 1 / sc.scale_
    pred = pred * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(pred, 'r', label='Original Price')
    plt.plot(y, 'g', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)
else:
    st.warning("Model couldn't be loaded. Please check the model file and try again.")
