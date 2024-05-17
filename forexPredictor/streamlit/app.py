# import module
import streamlit as st
 
# Title
st.title("Stock Market Predictor !!!")

st.sidebar.write("Select from below options")
side = st.sidebar.selectbox("Selcect one", ["Price Prediction", "Correlation Check", "Stock News", "LSTM"])
