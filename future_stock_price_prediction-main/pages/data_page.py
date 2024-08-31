import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from datetime import date

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def show_data_page():
    st.title('Data Page')
    selected_stock = st.session_state.get('selected_stock')
    if not selected_stock:
        st.write("No stock selected. Please go to the Home page and select a stock.")
        return
    
    data = load_data(selected_stock)
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_raw_data()
