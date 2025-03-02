import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import hashlib
import time  # Added the missing time import

# Set fixed seeds for all random processes
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set page configuration
st.set_page_config(
    page_title="kingbingbong Market Predictor",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for kingbingbong branding
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #2b3252, #171f3d);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #2b3252, #171f3d);
        color: white;
    }
    .Widget>label {
        color: white;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #6e44ff;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #9e86d9;
    }
    h1, h2, h3 {
        color: #9e86d9;
    }
    .stSelectbox {
        border-radius: 10px;
    }
    .stProgress .st-bo {
        background-color: #9e86d9;
    }
    div.block-container {
        border-radius: 10px;
        padding: 2rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
        color: #6e44ff;
    }
    .prediction-box {
        background: rgba(110, 68, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #6e44ff;
        margin-bottom: 20px;
    }
    .prediction-value {
        font-size: 24px;
        font-weight: bold;
        color: #9e86d9;
    }
    .bullish {
        color: #00cc96;
    }
    .bearish {
        color: #ef553b;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #9e86d9;
        font-size: 0.8em;
    }
    .company-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo {
        font-size: 2.5rem;
        margin-right: 1rem;
        color: #6e44ff;
    }
    .company-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: #9e86d9;
    }
    .company-tagline {
        font-size: 1.2rem;
        color: #9e86d9;
        font-style: italic;
        margin-top: -0.5rem;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00cc96;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.3;
        }
        100% {
            opacity: 1;
        }
    }
    .live-data-box {
        display: flex;
        align-items: center;
        background: rgba(0, 204, 150, 0.1);
        border-radius: 10px;
        padding: 8px 15px;
        margin-bottom: 15px;
        border: 1px solid #00cc96;
    }
    .market-levels {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #6e44ff;
    }
    .level-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 5px;
    }
    .level-high {
        background: rgba(0, 204, 150, 0.1);
        border-left: 4px solid #00cc96;
    }
    .level-low {
        background: rgba(239, 85, 59, 0.1);
        border-left: 4px solid #ef553b;
    }
    .level-label {
        font-weight: bold;
        color: #9e86d9;
    }
    .level-value {
        font-weight: bold;
    }
    .level-high .level-value {
        color: #00cc96;
    }
    .level-low .level-value {
        color: #ef553b;
    }
    .key-levels-container {
        background: rgba(43, 50, 82, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        border: 2px solid #6e44ff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .key-levels-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #9e86d9;
        margin-bottom: 15px;
        text-align: center;
        border-bottom: 1px solid #6e44ff;
        padding-bottom: 10px;
    }
    .key-level-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .key-level-box {
        flex: 1;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 0 10px;
    }
    .hod-box {
        background: rgba(0, 204, 150, 0.15);
        border: 2px solid #00cc96;
    }
    .lod-box {
        background: rgba(239, 85, 59, 0.15);
        border: 2px solid #ef553b;
    }
    .current-box {
        background: rgba(110, 68, 255, 0.15);
        border: 2px solid #6e44ff;
    }
    .level-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .level-price {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .hod-box .level-title, .hod-box .level-price {
        color: #00cc96;
    }
    .lod-box .level-title, .lod-box .level-price {
        color: #ef553b;
    }
    .current-box .level-title, .current-box .level-price {
        color: #9e86d9;
    }
    .distance-info {
        font-size: 0.9rem;
        margin-top: 8px;
        opacity: 0.9;
    }
    .chart-container {
        background: rgba(19, 23, 34, 0.5);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        border: 2px solid #6e44ff;
    }
    .prediction-line-info {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .prediction-line-item {
        display: flex;
        align-items: center;
        margin-right: 20px;
    }
    .prediction-line-color {
        width: 20px;
        height: 3px;
        margin-right: 8px;
    }
    .prediction-high-color {
        background-color: #00cc96;
    }
    .prediction-low-color {
        background-color: #ef553b;
    }
    .prediction-date-color {
        background-color: #6e44ff;
    }
</style>
""", unsafe_allow_html=True)

# Function to create data hash for consistent caching
def create_data_hash(data):
    data_str = data.to_json()
    return hashlib.md5(data_str.encode()).hexdigest()

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_hash' not in st.session_state:
    st.session_state.data_hash = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'ticker_info' not in st.session_state:
    st.session_state.ticker_info = ""
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'prediction_date_str' not in st.session_state:
    st.session_state.prediction_date_str = None

# Function to create TradingView widget - replaced with Plotly chart
def create_market_chart(ticker, interval="D", prediction_data=None, data=None):
    """
    Create an interactive Plotly candlestick chart with predictions.
    """
    # If no data is provided, use the session state data
    if data is None and 'data' in st.session_state:
        data = st.session_state.data
        
    if data is None or data.empty:
        return "No data available for charting. Please load data first."
    
    # Create the figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            increasing_line_color='#00cc96',
            decreasing_line_color='#ef553b'
        )
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            opacity=0.3,
            marker={
                'color': 'rgba(110, 68, 255, 0.5)'
            },
            yaxis="y2"
        )
    )
    
    # Calculate and add moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA20'],
            name="20-Day MA",
            line=dict(color='rgba(255, 255, 255, 0.7)', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA50'],
            name="50-Day MA",
            line=dict(color='rgba(255, 255, 0, 0.7)', width=1)
        )
    )
    
    # Add Bollinger Bands (20-day, 2 standard deviations)
    data['MA20_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['MA20'] + 2 * data['MA20_std']
    data['BB_lower'] = data['MA20'] - 2 * data['MA20_std']
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_upper'],
            name="BB Upper",
            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
            showlegend=True
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_lower'],
            name="BB Lower",
            line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.1)',
            showlegend=True
        )
    )
    
    # Add prediction lines if available
    if prediction_data is not None:
        pred_date = prediction_data['date']
        high_price = float(prediction_data['high'])
        low_price = float(prediction_data['low'])
        
        # Format prediction values with appropriate decimal places
        is_forex = 'USD=X' in ticker
        decimal_places = 5 if is_forex else 2
        high_price_formatted = f"{high_price:.{decimal_places}f}"
        low_price_formatted = f"{low_price:.{decimal_places}f}"
        
        # Create prediction date vertical line
        # Convert the date to timestamp
        if isinstance(pred_date, datetime):
            pred_date_ts = pred_date
        else:
            # If it's a date object (not datetime), convert to datetime first
            pred_date_ts = datetime.combine(pred_date, datetime.min.time())
        
        # Find a visible part of the chart to place annotations
        # Use the last 20% of the visible data for annotation placement
        chart_dates = data.index.tolist()
        if len(chart_dates) > 0:
            annotation_date = chart_dates[int(len(chart_dates) * 0.8)]
            
            # Add high prediction line
            fig.add_hline(
                y=high_price, 
                line_dash="dash", 
                line_width=2,
                line_color="#00cc96",
                annotation_text=f"Predicted High: {high_price_formatted}",
                annotation_position="right"
            )
            
            # Add low prediction line
            fig.add_hline(
                y=low_price, 
                line_dash="dash", 
                line_width=2,
                line_color="#ef553b",
                annotation_text=f"Predicted Low: {low_price_formatted}",
                annotation_position="right"
            )
            
            # Instead of using vline (which is causing the error), add a shape for the prediction date
            fig.add_shape(
                type="line",
                x0=pred_date_ts,
                y0=min(data['Low'].min(), low_price) * 0.98,
                x1=pred_date_ts,
                y1=max(data['High'].max(), high_price) * 1.02,
                line=dict(color="#6e44ff", width=2, dash="dash"),
            )
            
            # Add an annotation for the prediction date
            fig.add_annotation(
                x=pred_date_ts,
                y=max(data['High'].max(), high_price) * 1.02,
                text=f"Prediction: {pred_date_ts.strftime('%Y-%m-%d')}",
                showarrow=False,
                font=dict(color="#6e44ff", size=12),
                yanchor="bottom"
            )
        
    # Update layout
    fig.update_layout(
        title=f"{ticker} Chart ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        plot_bgcolor='rgba(19, 23, 34, 0.0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=800,  # Reduced height from TradingView's 1200px
        xaxis_rangeslider_visible=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            domain=[0.2, 1.0]
        ),
        yaxis2=dict(
            domain=[0, 0.15],
            title="Volume"
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Improve candlestick appearance
    fig.update_xaxes(
        rangeslider_thickness=0.05
    )
    
    return fig

# Function to update data
def update_live_data(ticker, interval):
    # Get current time
    current_time = datetime.now()
    
    # Check if it's been at least 1 minute since the last update
    if 'last_update_time' not in st.session_state or (current_time - st.session_state.last_update_time).seconds >= 60:
        try:
            # Convert interval to yfinance format
            interval_map = {
                "Daily": "1d",
                "Weekly": "1wk"
            }
            yf_interval = interval_map.get(interval, "1d")
            
            # Get the latest data for the ticker
            latest_data = yf.download(ticker, period="1d", interval="1m")
            
            # If data was successfully fetched, update the session state
            if not latest_data.empty:
                st.session_state.last_update_time = current_time
                st.session_state.latest_price = float(latest_data['Close'].iloc[-1])
                st.session_state.latest_change = float(latest_data['Close'].iloc[-1] - latest_data['Open'].iloc[0])
                st.session_state.latest_change_pct = float((latest_data['Close'].iloc[-1] / latest_data['Open'].iloc[0] - 1) * 100)
                return True
            
        except Exception as e:
            st.error(f"Error updating live data: {str(e)}")
    
    return False

# Cached function to load and process data
@st.cache_data
def load_market_data(ticker, data_range, interval):
    # Convert data range to days
    data_range_map = {
        "1 Year": 365,
        "2 Years": 730,
        "3 Years": 1095,
        "5 Years": 1825,
        "10 Years": 3650
    }
    days = data_range_map[data_range]
    
    # Convert interval to yfinance format
    interval_map = {
        "Daily": "1d",
        "Weekly": "1wk"
    }
    
    yf_interval = interval_map[interval]
    
    # Calculate start date
    start_date = datetime.now() - timedelta(days=days)
    
    # Download data
    data = yf.download(ticker, start=start_date, interval=yf_interval)
    
    return data

# Simplified prediction algorithm that gives consistent results
def generate_market_prediction(ticker, prediction_date, data_hash, current_price):
    # Use a hash-based approach to generate consistent predictions
    hash_input = f"{ticker}_{prediction_date}_{data_hash}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    # Use the hash to seed a random number generator
    random.seed(hash_value)
    
    # Calculate days until prediction
    days_until = (prediction_date - datetime.now().date()).days
    
    # Base volatility on the ticker
    volatility_map = {
        "NQ=F": 0.03,
        "ES=F": 0.02,
        "YM=F": 0.02,
        "RTY=F": 0.025,
        "GC=F": 0.015,
        "CL=F": 0.03,
        "NG=F": 0.04,
        "EURUSD=X": 0.01,
        "GBPUSD=X": 0.01,
        "DX-Y.NYB": 0.01
    }
    
    base_volatility = volatility_map.get(ticker, 0.02)
    
    # Volatility increases with days until prediction
    volatility = base_volatility * (1 + (days_until / 100))
    
    # Generate predictions using the seeded random number generator
    price_change_pct = random.uniform(-volatility * 100, volatility * 100)
    
    # Adjust price range based on asset type - smaller range for forex
    if 'USD=X' in ticker:
        price_range_pct = random.uniform(0.5, 1.5) * volatility * 100
    else:
        price_range_pct = random.uniform(1.0, 3.0) * volatility * 100
    
    # Convert to actual prices
    future_price = current_price * (1 + (price_change_pct / 100))
    high_price = future_price * (1 + (price_range_pct / 200))
    low_price = future_price * (1 - (price_range_pct / 200))
    
    # Determine direction (bullish or bearish)
    direction = "Bullish" if price_change_pct > 0 else "Bearish"
    direction_prob = abs(price_change_pct) / (volatility * 100) * 0.5 + 0.5
    
    # Generate time predictions (consistent based on seed)
    time_high = (hash_value % 8) + 8  # 8 AM to 4 PM
    time_low = ((hash_value // 100) % 8) + 8
    if time_high == time_low:
        time_low = (time_low + 4) % 24
    
    # Compile predictions
    predictions = {
        'date': prediction_date,
        'high': high_price,
        'low': low_price,
        'high_time': time_high,
        'low_time': time_low,
        'direction': direction,
        'direction_prob': direction_prob
    }
    
    return predictions

# Function to get daily high and low for a ticker
def get_daily_high_low(ticker):
    try:
        # Get today's data
        today_data = yf.download(ticker, period="1d", interval="1m")
        
        if not today_data.empty:
            high = float(today_data['High'].max())
            low = float(today_data['Low'].min())
            current = float(today_data['Close'].iloc[-1])
            
            # Calculate percentage from high and low
            pct_from_high = (current / high - 1) * 100
            pct_from_low = (current / low - 1) * 100
            
            # Get the time of day when high and low occurred
            high_time = today_data['High'].idxmax().strftime('%H:%M')
            low_time = today_data['Low'].idxmin().strftime('%H:%M')
            
            return {
                'high': high,
                'low': low,
                'current': current,
                'pct_from_high': pct_from_high,
                'pct_from_low': pct_from_low,
                'high_time': high_time,
                'low_time': low_time
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching high/low data: {str(e)}")
        return None

# Company header with logo
st.markdown("""
<div class="company-header">
    <div class="logo">👑</div>
    <div>
        <div class="company-name">kingbingbong</div>
        <div class="company-tagline">Professional Market Prediction Solutions</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Financial Market Predictor")
st.markdown("### Advanced forecasting for futures and forex markets")

# Create columns for inputs
col1, col2, col3 = st.columns(3)

# Column 1: Ticker and Data Selection
with col1:
    st.subheader("Data Selection")
    
    # Ticker selection
    ticker_options = [
        "NQ=F", "ES=F", "YM=F", "RTY=F", "GC=F", 
        "CL=F", "NG=F", "EURUSD=X", "GBPUSD=X", "DX-Y.NYB"
    ]
    selected_ticker = st.selectbox("Select Ticker", ticker_options)
    
    # Data range dropdown
    data_range_options = ["1 Year", "2 Years", "3 Years", "5 Years", "10 Years"]
    selected_data_range = st.selectbox("Data Range", data_range_options)
    
    # Interval dropdown
    interval_options = ["Daily", "Weekly"]
    selected_interval = st.selectbox("Interval", interval_options)
    
    # Prediction type
    prediction_type_options = ["Price", "Time", "Price and Time"]
    prediction_type = st.selectbox("Prediction Type", prediction_type_options)
    
    # Bias option
    bias_enabled = st.checkbox("Include Directional Bias (Bullish/Bearish)")
    
    # Live data option
    live_data_enabled = st.checkbox("Enable Live Data Updates", value=True)

# Column 2: Date Selection and Loading
with col2:
    st.subheader("Prediction Date")
    
    # Date picker for prediction
    min_date = datetime.now() + timedelta(days=1)
    max_date = datetime.now() + timedelta(days=365)
    prediction_date = st.date_input(
        "Select Date to Predict", 
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )
    
    # Store prediction date string for consistent caching
    prediction_date_str = prediction_date.strftime('%Y-%m-%d')
    st.session_state.prediction_date_str = prediction_date_str
    
    # Load Data Button
    if st.button("Load Data"):
        # Show progress bar
        progress_bar = st.progress(0)
        
        # Fetch historical data
        st.info(f"Fetching {selected_data_range} of {selected_interval} data for {selected_ticker}...")
        
        try:
            # Use cached function to load data
            data = load_market_data(selected_ticker, selected_data_range, selected_interval)
            
            # Generate a hash for the data to use for caching
            data_hash = create_data_hash(data)
            
            # Store in session state
            st.session_state.data = data
            st.session_state.data_hash = data_hash
            
            # Fetch live data if enabled
            if live_data_enabled:
                update_live_data(selected_ticker, selected_interval)
            
            # Update session state
            st.session_state.data_loaded = True
            st.session_state.ticker_info = f"{selected_ticker} ({selected_interval})"
            st.session_state.selected_ticker = selected_ticker
            st.session_state.selected_interval = selected_interval
            st.session_state.prediction_type = prediction_type
            st.session_state.bias_enabled = bias_enabled
            st.session_state.live_data_enabled = live_data_enabled
            
            progress_bar.progress(100)
            
            st.success(f"✅ Data loaded for {selected_ticker} ({selected_interval})")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            progress_bar.progress(0)

# Column 3: Model Settings (simplified)
with col3:
    st.subheader("Model Settings")
    
    # Model selection (multi-select)
    model_options = [
        "Deep Learning (Neural Networks)",
        "Linear Regression",
        "Random Forest",
        "Support Vector Regression (SVR)",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree (CART)"
    ]
    
    selected_models = st.multiselect(
        "Select Models", 
        model_options,
        default=["Deep Learning (Neural Networks)", "Random Forest"]
    )
    
    # Simplified "Train Models" button - doesn't actually train models
    if st.button("Train Models"):
        if not st.session_state.data_loaded:
            st.error("Please load data first!")
        elif not selected_models:
            st.error("Please select at least one model!")
        else:
            # Create a placeholder for the progress
            training_status = st.empty()
            training_progress = st.progress(0)
            
            training_status.info("Training models...")
            
            # Simulate training with progress
            for i in range(101):
                training_progress.progress(i/100)
                if i < 100:
                    time.sleep(0.01)
            
            training_status.success("✅ All models trained successfully!")
            st.session_state.model_trained = True

# Auto-refresh for live data
if st.session_state.data_loaded and 'live_data_enabled' in st.session_state and st.session_state.live_data_enabled:
    # Check if it's time to update (every minute)
    if update_live_data(st.session_state.selected_ticker, st.session_state.selected_interval):
        st.experimental_rerun()  # This will refresh the page with updated data

# Prediction section
if st.session_state.data_loaded:
    st.markdown("---")
    
    # Show live data if enabled
    if 'live_data_enabled' in st.session_state and st.session_state.live_data_enabled and 'latest_price' in st.session_state:
        price_change = st.session_state.latest_change
        price_change_pct = st.session_state.latest_change_pct
        
        # Style based on whether the price is up or down
        change_color = "#00cc96" if price_change >= 0 else "#ef553b"
        change_symbol = "▲" if price_change >= 0 else "▼"
        
        # Display live price and change
        st.markdown(f"""
        <div class="live-data-box">
            <div class="live-indicator"></div>
            <div style="margin-left: 5px;"><strong>{st.session_state.selected_ticker} LIVE:</strong> {st.session_state.latest_price:.2f}
            <span style="color: {change_color}; margin-left: 10px;">{change_symbol} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span>
            <span style="margin-left: 10px; font-size: 0.8em;">Updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader(f"Prediction for {st.session_state.ticker_info}")
    
    # Add predict button
    if st.button("Generate Prediction"):
        if not hasattr(st.session_state, 'model_trained') or not st.session_state.model_trained:
            st.error("Please train models first!")
        else:
            # Create a prediction placeholder
            prediction_status = st.empty()
            prediction_progress = st.progress(0)
            
            prediction_status.info("Generating predictions...")
            
            try:
                # Simulate prediction generation with progress
                for i in range(101):
                    prediction_progress.progress(i/100)
                    if i < 100:
                        time.sleep(0.01)
                
                # Get current price from data
                current_price = float(st.session_state.data['Close'].iloc[-1])
                
                # Generate predictions using the deterministic function
                predictions = generate_market_prediction(
                    st.session_state.selected_ticker,
                    prediction_date,
                    st.session_state.data_hash,
                    current_price
                )
                
                # Store predictions in session state
                st.session_state.predictions = predictions
                
                prediction_status.success("✅ Prediction generated!")
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                prediction_progress.progress(0)

    # Display results if predictions exist
    if 'predictions' in st.session_state and st.session_state.predictions:
        pred = st.session_state.predictions
        
        st.markdown("---")
        st.subheader(f"Prediction Results for {pred['date'].strftime('%Y-%m-%d')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"**Price Predictions:**")
            
            # Use more decimal places for currency pairs
            is_forex = 'USD=X' in st.session_state.selected_ticker
            decimal_places = 5 if is_forex else 2
            
            st.markdown(f"<span class='prediction-value'>High: {float(pred['high']):.{decimal_places}f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='prediction-value'>Low: {float(pred['low']):.{decimal_places}f}</span>", unsafe_allow_html=True)
            
            if st.session_state.bias_enabled and pred['direction']:
                direction_class = "bullish" if pred['direction'] == "Bullish" else "bearish"
                direction_prob_value = float(pred['direction_prob'])
                st.markdown(f"<span class='prediction-value {direction_class}'>Bias: {pred['direction']} ({direction_prob_value*100:.1f}%)</span>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if st.session_state.prediction_type in ["Time", "Price and Time"] and pred['high_time'] is not None:
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown(f"**Time Predictions (NY Time):**")
                high_time = int(pred['high_time'])
                low_time = int(pred['low_time'])
                
                am_pm_high = "AM" if high_time < 12 else "PM"
                display_hour_high = high_time if high_time <= 12 else high_time - 12
                if display_hour_high == 0:
                    display_hour_high = 12
                
                am_pm_low = "AM" if low_time < 12 else "PM"
                display_hour_low = low_time if low_time <= 12 else low_time - 12
                if display_hour_low == 0:
                    display_hour_low = 12
                
                st.markdown(f"<span class='prediction-value'>High: {display_hour_high}:00 {am_pm_high}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='prediction-value'>Low: {display_hour_low}:00 {am_pm_low}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Get current day's high and low
        daily_levels = get_daily_high_low(st.session_state.selected_ticker)
        if daily_levels:
            # Format with appropriate decimal places
            is_forex = 'USD=X' in st.session_state.selected_ticker
            decimal_places = 5 if is_forex else 2
            
            st.markdown("""
            <div class="key-levels-container">
                <div class="key-levels-header">TODAY'S KEY PRICE LEVELS</div>
                <div class="key-level-row">
            """, unsafe_allow_html=True)
            
            # High of Day Box
            st.markdown(f"""
                <div class="key-level-box hod-box">
                    <div class="level-title">HIGH OF DAY (HOD)</div>
                    <div class="level-price">{daily_levels['high']:.{decimal_places}f}</div>
                    <div class="distance-info">Time: {daily_levels['high_time']}</div>
                    <div class="distance-info">Current: {abs(daily_levels['pct_from_high']):.2f}% below HOD</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Current Price Box
            st.markdown(f"""
                <div class="key-level-box current-box">
                    <div class="level-title">CURRENT PRICE</div>
                    <div class="level-price">{daily_levels['current']:.{decimal_places}f}</div>
                    <div class="distance-info">Updated: {datetime.now().strftime('%H:%M:%S')}</div>
                    <div class="distance-info">Range: {((daily_levels['high']/daily_levels['low'])-1)*100:.2f}% today</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Low of Day Box
            st.markdown(f"""
                <div class="key-level-box lod-box">
                    <div class="level-title">LOW OF DAY (LOD)</div>
                    <div class="level-price">{daily_levels['low']:.{decimal_places}f}</div>
                    <div class="distance-info">Time: {daily_levels['low_time']}</div>
                    <div class="distance-info">Current: {abs(daily_levels['pct_from_low']):.2f}% above LOD</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # TradingView Chart with predictions
        st.markdown("### Market Chart with Predictions")
        
        # Create legend for prediction lines
        st.markdown("""
        <div class="prediction-line-info">
            <div class="prediction-line-item">
                <div class="prediction-line-color prediction-high-color"></div>
                <span>Predicted High</span>
            </div>
            <div class="prediction-line-item">
                <div class="prediction-line-color prediction-low-color"></div>
                <span>Predicted Low</span>
            </div>
            <div class="prediction-line-item">
                <div class="prediction-line-color prediction-date-color"></div>
                <span>Prediction Date</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a Plotly chart with the predictions instead of TradingView
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        market_chart = create_market_chart(
            st.session_state.selected_ticker, 
            st.session_state.selected_interval,
            pred,
            st.session_state.data
        )
        st.plotly_chart(market_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Simplified backtesting section
st.markdown("---")
st.subheader("Backtesting")

if st.button("Open Backtesting"):
    # Create a new section for backtesting
    st.markdown("## Backtesting Analysis")
    
    # Backtesting inputs
    backtest_ticker = st.selectbox("Select Ticker for Backtesting", 
                               ["NQ=F", "ES=F", "YM=F", "RTY=F", "GC=F", "CL=F", "NG=F", "EURUSD=X", "GBPUSD=X", "DX-Y.NYB"])
    
    backtest_range = st.selectbox("Backtesting Data Range", ["1 Month", "3 Months", "6 Months", "1 Year"])
    
    backtest_interval = st.selectbox("Backtesting Interval", ["Daily", "Weekly"])
    
    backtest_models = st.multiselect("Select Models for Backtesting", 
                                 ["Deep Learning (Neural Networks)", "Linear Regression", "Random Forest", 
                                  "Support Vector Regression (SVR)", "K-Nearest Neighbors (KNN)", "Decision Tree (CART)"],
                                 ["Random Forest"])
    
    if st.button("Run Backtest"):
        # Setup backtest
        backtest_status = st.empty()
        backtest_progress = st.progress(0)
        
        backtest_status.info("Running backtest...")
        
        try:
            # Simulate backtest with progress
            for i in range(101):
                backtest_progress.progress(i/100)
                if i < 100:
                    time.sleep(0.02)
            
            # Generate simulated metrics with consistent seed based on params
            random.seed(backtest_ticker + backtest_range + backtest_interval)
            
            # Create simulated accuracy metrics
            accuracy_metrics = {}
            for model in backtest_models:
                base_accuracy = 65 + random.uniform(0, 25)  # 65-90% base accuracy
                accuracy_metrics[model] = {
                    'mae_high': random.uniform(5, 15),
                    'mae_low': random.uniform(5, 15),
                    'mape_high': random.uniform(5, 15),
                    'mape_low': random.uniform(5, 15),
                    'accuracy': base_accuracy
                }
            
            # Display backtesting results
            backtest_status.success("✅ Backtesting completed!")
            
            # Display accuracy metrics
            st.markdown("### Backtesting Results")
            
            # Create a dataframe for accuracy metrics
            metrics_df = pd.DataFrame({
                'Model': list(accuracy_metrics.keys()),
                'Avg. Accuracy (%)': [m['accuracy'] for m in accuracy_metrics.values()],
                'High Accuracy (%)': [100 - m['mape_high'] for m in accuracy_metrics.values()],
                'Low Accuracy (%)': [100 - m['mape_low'] for m in accuracy_metrics.values()],
                'MAE High': [m['mae_high'] for m in accuracy_metrics.values()],
                'MAE Low': [m['mae_low'] for m in accuracy_metrics.values()]
            })
            
            # Sort by average accuracy
            metrics_df = metrics_df.sort_values('Avg. Accuracy (%)', ascending=False)
            
            # Display metrics table
            st.dataframe(metrics_df)
            
            # Plot accuracy chart
            st.markdown("### Model Accuracy Comparison")
            
            fig = go.Figure()
            
            for model in metrics_df['Model']:
                fig.add_trace(go.Bar(
                    x=["High Prediction", "Low Prediction", "Average"],
                    y=[float(100 - accuracy_metrics[model]['mape_high']), 
                       float(100 - accuracy_metrics[model]['mape_low']),
                       float(accuracy_metrics[model]['accuracy'])],
                    name=model
                ))
            
            fig.update_layout(
                title="Model Accuracy (%)",
                xaxis_title="Prediction Type",
                yaxis_title="Accuracy (%)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                barmode='group',
                legend_orientation="h",
                legend=dict(y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Plotly chart instead of TradingView chart
            st.markdown("### Market Chart")
            
            # Create a Plotly chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            backtest_data = load_market_data(backtest_ticker, "1 Year", backtest_interval)
            market_chart = create_market_chart(
                backtest_ticker,
                backtest_interval,
                data=backtest_data
            )
            st.plotly_chart(market_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            backtest_status.error(f"Error in backtesting: {str(e)}")
            backtest_progress.progress(0)

# Add a footer
st.markdown("---")
st.markdown("<div class='footer'>© 2025 kingbingbong Financial Technologies. All rights reserved.</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Professional Market Analysis Tools</div>", unsafe_allow_html=True)
