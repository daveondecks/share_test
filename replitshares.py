import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="FT100 Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# Function to get UK market data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_uk_market_data():
    try:
        # Get FTSE 100 data
        ftse = yf.Ticker("^FTSE")
        current_price = ftse.history(period='1d')['Close'].iloc[-1]
        daily_change = ftse.history(period='2d')
        daily_change = ((daily_change['Close'].iloc[-1] - daily_change['Close'].iloc[-2]) / daily_change['Close'].iloc[-2]) * 100

        return {
            'FTSE 100': f"£{current_price:,.2f}",
            'Daily Change': f"{daily_change:+.2f}%",
            'Bank Rate': "5.25%",  # Bank of England Base Rate
            'Inflation Rate': "4.0%",  # UK CPI
            'GBP/USD': f"${ftse.info.get('regularMarketPrice', 'N/A')}",
            'Last Updated': datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return {
            'FTSE 100': "N/A",
            'Daily Change': "N/A",
            'Bank Rate': "5.25%",
            'Inflation Rate': "4.0%",
            'GBP/USD': "N/A",
            'Last Updated': datetime.now().strftime("%H:%M:%S")
        }

# FTSE 100 tickers
@st.cache_data
def get_ftse100_tickers():
    """Get comprehensive list of FTSE 100 companies"""
    return {
        'III.L': '3i Group',
        'ADM.L': 'Admiral Group',
        'AAF.L': 'Airtel Africa',
        'ANTO.L': 'Antofagasta',
        'AHT.L': 'Ashtead Group',
        'ABF.L': 'Associated British Foods',
        'AZN.L': 'AstraZeneca',
        'AUTO.L': 'Auto Trader Group',
        'AVV.L': 'Aveva Group',
        'BA.L': 'BAE Systems',
        'BARC.L': 'Barclays',
        'BDEV.L': 'Barratt Developments',
        'BKG.L': 'Berkeley Group Holdings',
        'BP.L': 'BP',
        'BATS.L': 'British American Tobacco',
        'BLND.L': 'British Land',
        'BT-A.L': 'BT Group',
        'BNZL.L': 'Bunzl',
        'BRBY.L': 'Burberry',
        'CCH.L': 'Coca-Cola HBC',
        'CPG.L': 'Compass Group',
        'CRH.L': 'CRH',
        'CRDA.L': 'Croda International',
        'DCC.L': 'DCC',
        'DGE.L': 'Diageo',
        'ENT.L': 'Entain',
        'EVR.L': 'Evraz',
        'EXPN.L': 'Experian',
        'FERG.L': 'Ferguson',
        'FLTR.L': 'Flutter Entertainment',
        'FRES.L': 'Fresnillo',
        'GSK.L': 'GlaxoSmithKline',
        'GLEN.L': 'Glencore',
        'HLN.L': 'Haleon',
        'HLMA.L': 'Halma',
        'HBR.L': 'Harbour Energy',
        'HSBA.L': 'HSBC Holdings',
        'IGG.L': 'IG Group Holdings',
        'IMB.L': 'Imperial Brands',
        'INF.L': 'Informa',
        'IHG.L': 'InterContinental Hotels Group',
        'IAG.L': 'International Consolidated Airlines Group',
        'ITRK.L': 'Intertek',
        'JD.L': 'JD Sports Fashion',
        'JMAT.L': 'Johnson Matthey',
        'KGF.L': 'Kingfisher',
        'LAND.L': 'Land Securities Group',
        'LGEN.L': 'Legal & General Group',
        'LLOY.L': 'Lloyds Banking Group',
        'LSEG.L': 'London Stock Exchange Group',
        'MNG.L': 'M&G',
        'MRO.L': 'Melrose Industries',
        'MNDI.L': 'Mondi',
        'NG.L': 'National Grid',
        'NWG.L': 'NatWest Group',
        'NXT.L': 'Next',
        'OCDO.L': 'Ocado Group',
        'PSON.L': 'Pearson',
        'PSH.L': 'Pershing Square Holdings',
        'PSN.L': 'Persimmon',
        'PHNX.L': 'Phoenix Group Holdings',
        'PRU.L': 'Prudential',
        'RKT.L': 'Reckitt Benckiser',
        'REL.L': 'RELX',
        'RTO.L': 'Rentokil Initial',
        'RIO.L': 'Rio Tinto',
        'RR.L': 'Rolls-Royce Holdings',
        'RS1.L': 'RS Group',
        'SGE.L': 'Sage Group',
        'SBRY.L': "Sainsbury's",
        'SDR.L': 'Schroders',
        'SMT.L': 'Scottish Mortgage Investment Trust',
        'SGRO.L': 'Segro',
        'SVT.L': 'Severn Trent',
        'SHEL.L': 'Shell',
        'SMDS.L': 'Smith (DS)',
        'SMIN.L': 'Smiths Group',
        'SN.L': 'Smith & Nephew',
        'SPX.L': 'Spirax-Sarco Engineering',
        'SSE.L': 'SSE',
        'STAN.L': 'Standard Chartered',
        'STJ.L': "St. James's Place",
        'TW.L': 'Taylor Wimpey',
        'TSCO.L': 'Tesco',
        'ULVR.L': 'Unilever',
        'UU.L': 'United Utilities Group',
        'VOD.L': 'Vodafone Group',
        'WEIR.L': 'Weir Group',
        'WTB.L': 'Whitbread',
        'WPP.L': 'WPP'
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

def calculate_projections(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Project 30 days into the future
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=31, freq='D')[1:]
    future_X = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_prices = model.predict(future_X)

    projection_df = pd.DataFrame({
        'Date': future_dates,
        'Projected_Close': future_prices
    })

    return projection_df

# Main title
st.title("📈 FTSE 100 Stock Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Stock Selection")

# Ticker selection
tickers = get_ftse100_tickers()
selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    options=list(tickers.keys()),
    format_func=lambda x: f"{tickers[x]} ({x})"
)

# Time period selection
time_periods = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}
selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=list(time_periods.keys())
)

st.sidebar.markdown("---")

# Add UK Economic Statistics to Sidebar
st.sidebar.header("🇬🇧 UK Market Statistics")
uk_stats = get_uk_market_data()

# Display statistics in an organized way
for metric, value in uk_stats.items():
    if metric != 'Last Updated':
        st.sidebar.metric(metric, value)

st.sidebar.caption(f"Last Updated: {uk_stats['Last Updated']}")


# Fetch data
df = fetch_stock_data(selected_ticker, time_periods[selected_period])

# Calculate statistics
current_price = df['Close'].iloc[-1]
avg_price = df['Close'].mean()
high_price = df['High'].max()
low_price = df['Low'].min()

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"£{current_price:.2f}")
with col2:
    st.metric("Average Price", f"£{avg_price:.2f}")
with col3:
    st.metric("Period High", f"£{high_price:.2f}")
with col4:
    st.metric("Period Low", f"£{low_price:.2f}")

# Charts section
st.markdown("## 📊 Price Analysis")
chart_tabs = st.tabs(["Price History", "Volume Analysis", "Price Projections"])

with chart_tabs[0]:
    # Historical price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    fig_price.update_layout(title='Stock Price History (Candlestick)')
    st.plotly_chart(fig_price, use_container_width=True)

with chart_tabs[1]:
    # Volume analysis
    fig_volume = px.bar(df, x='Date', y='Volume', title='Trading Volume')
    st.plotly_chart(fig_volume, use_container_width=True)

with chart_tabs[2]:
    # Price projections
    projection_df = calculate_projections(df)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Historical Price'
    ))
    fig_proj.add_trace(go.Scatter(
        x=projection_df['Date'],
        y=projection_df['Projected_Close'],
        name='Projected Price',
        line=dict(dash='dash')
    ))
    fig_proj.update_layout(title='Price History and 30-Day Projection')
    st.plotly_chart(fig_proj, use_container_width=True)

# Data table section
st.markdown("## 📋 Historical Data")
st.dataframe(
    df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']],
    use_container_width=True,
    hide_index=True
)

# Download data
csv = df.to_csv(index=False)
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name=f"{selected_ticker}_stock_data.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("### 📊 Dashboard Statistics")
col1, col2 = st.columns(2)
with col1:
    st.info(f"Selected Stock: {tickers[selected_ticker]} ({selected_ticker})")
with col2:
    st.info(f"Time Period: {selected_period}")
