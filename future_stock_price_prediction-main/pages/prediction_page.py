import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
import yfinance as yf

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def show_prediction_page():
    st.title('Prediction Page')
    selected_stock = st.session_state.get('selected_stock')
    n_years = st.session_state.get('n_years')
    
    if not selected_stock or not n_years:
        st.write("No stock or years selected. Please go to the Home page and select a stock and number of years.")
        return
    
    period = n_years * 365
    data = load_data(selected_stock)
    
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
