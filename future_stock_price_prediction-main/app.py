import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import mplfinance as mpf
from matplotlib import pyplot as plt

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app configuration
st.set_page_config(page_title='Stock Forecast App', layout='wide')

# Navbar CSS
css = """
<style>
* {
  box-sizing: border-box;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  padding: 0;
  margin: 0;
  font-family: "Poppins", sans-serif;
}

nav {
  padding: 5px 5%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px,
    rgba(0, 0, 0, 0.3) 0px 1px 3px -1px;
  z-index: 1;
}

nav .logo {
  display: flex;
  align-items: center;
}

nav .logo img {
  height: 25px;
  width: auto;
  margin-right: 10px;
}

nav .logo h1 {
  font-size: 1.1rem;
  background: linear-gradient(to right, #b927fc 0%, #2c64fc 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

nav ul {
  list-style: none;
  display: flex;
}

nav ul li {
  margin-left: 1.5rem;
}

nav ul li a {
  text-decoration: none;
  color: #000;
  font-size: 95%;
  font-weight: 400;
  padding: 4px 8px;
  border-radius: 5px;
}

nav ul li a:hover {
  background-color: #f5f5f5;
}

.hamburger {
  display: none;
  cursor: pointer;
}

.hamburger .line {
  width: 25px;
  height: 1px;
  background-color: #1f1f1f;
  display: block;
  margin: 7px auto;
  transition: all 0.3s ease-in-out;
}

.hamburger-active {
  transition: all 0.3s ease-in-out;
  transition-delay: 0.6s;
  transform: rotate(45deg);
}

.hamburger-active .line:nth-child(2) {
  width: 0px;
}

.hamburger-active .line:nth-child(1),
.hamburger-active .line:nth-child(3) {
  transition-delay: 0.3s;
}

.hamburger-active .line:nth-child(1) {
  transform: translateY(12px);
}

.hamburger-active .line:nth-child(3) {
  transform: translateY(-5px) rotate(90deg);
}

.menubar {
  position: absolute;
  top: 0;
  left: -60%;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 60%;
  height: 100vh;
  padding: 20% 0;
  background: rgba(255, 255, 255);
  transition: all 0.5s ease-in;
  z-index: 2;
}

.active {
  left: 0;
  box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
}

.menubar ul {
  padding: 0;
  list-style: none;
}

.menubar ul li {
  margin-bottom: 32px;
}

.menubar ul li a {
  text-decoration: none;
  color: #000;
  font-size: 95%;
  font-weight: 400;
  padding: 5px 10px;
  border-radius: 5px;
}

.menubar ul li a:hover {
  background-color: #f5f5f5;
}
</style>
"""

# Navbar HTML
html = """
<nav>
  <div class="logo">
    <h1>LOGO</h1>
  </div>
  <ul>
    <li><a href="#">Home</a></li>
    <li><a href="#">Services</a></li>
    <li><a href="#">Blog</a></li>
    <li><a href="#">Contact Us</a></li>
  </ul>
  <div class="hamburger">
    <span class="line"></span>
    <span class="line"></span>
    <span class="line"></span>
  </div>
</nav>
"""

# Display the navbar using st.markdown
st.markdown(css + html, unsafe_allow_html=True)

# Home Page Title
css_title = """
  <center><h1>HOME
  <p>Welcome to the Stock Forecast App. Navigate to the Prediction page for future forecasting.</p></h1></center>
"""
st.markdown(css_title, unsafe_allow_html=True)

# Divider
st.markdown('---')

# Load stock symbols
df = pd.read_csv('data/stock_name.csv')
symbol = df.iloc[:, 0]

# Data Page Title
css_data = """
  <center><h1>DATA</h1></center>
"""
st.markdown(css_data, unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header('User Input')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', symbol)
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Latest stock data info')
st.write(data.tail())

def plot_candlestick(ticker, start_date, end_date):
        # Download data
    data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate moving averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

        # Plot the candlestick chart with volume
    fig, axes = mpf.plot(
        data,
        type='candle',
        volume=True,
        mav=(50, 200),
        title=f'{ticker} Candlestick Chart with Volume',
        style='yahoo',
        ylabel='Price (USD)',
        ylabel_lower='Volume',
        returnfig=True
    )
    st.pyplot(fig)


# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["PAST DATA OF THE STOCKS", "FUTURE PREDICTION", "ANALYTICS OF STOCKS", "Candlestick Data"])

# Tab 1: Past Data
with tab1:
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

# Tab 2: Future Forecast Data
with tab2:
    st.subheader('Future Forecast Data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

# Tab 3: Forecast Components
with tab3:
    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Tab 4: Candlestick Data
with tab4:

  
    
  plot_candlestick(selected_stock, START, TODAY)
