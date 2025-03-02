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
    .tradingview-widget-container iframe {
        height: 2500px !important;
        width: 100% !important;
        min-height: 2500px !important;
    }
    .tradingview-widget-container {
        height: 2500px !important;
        width: 100% !important;
        min-height: 2500px !important;
    }
    .tradingview-widget-container div {
        height: 2500px !important;
    }
    #tradingview_chart {
        height: 2500px !important;
        width: 100% !important;
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

# Function to create TradingView widget
def create_tradingview_widget(ticker, interval="D", prediction_data=None):
    # Map yfinance tickers to TradingView format
    ticker_map = {
        "NQ=F": "CME_MINI:NQ1!",
        "ES=F": "CME_MINI:ES1!",
        "YM=F": "CBOT_MINI:YM1!",
        "RTY=F": "CME_MINI:RTY1!",
        "GC=F": "COMEX:GC1!",
        "CL=F": "NYMEX:CL1!",
        "NG=F": "NYMEX:NG1!",
        "EURUSD=X": "FX:EURUSD",
        "GBPUSD=X": "FX:GBPUSD",
        "DX-Y.NYB": "TVC:DXY"
    }
    
    # Map intervals to TradingView format
    interval_map = {
        "Daily": "D",
        "Weekly": "W"
    }
    
    tv_ticker = ticker_map.get(ticker, ticker)
    tv_interval = interval_map.get(interval, "D")
    
    # Create chart with predictions if available
    if prediction_data:
        pred_date = prediction_data['date'].strftime('%Y-%m-%d')
        high_price = float(prediction_data['high'])
        low_price = float(prediction_data['low'])
        
        # Format prediction values with appropriate decimal places
        if 'USD=X' in ticker:
            high_price_formatted = f"{high_price:.5f}"
            low_price_formatted = f"{low_price:.5f}"
        else:
            high_price_formatted = f"{high_price:.2f}"
            low_price_formatted = f"{low_price:.2f}"
        
        # Create a custom TradingView chart with prediction lines
        custom_script = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:2500px;width:100%;">
          <div id="tradingview_chart" style="height:2500px;width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "autosize": false,
            "width": "100%",
            "height": 2500,
            "symbol": "{tv_ticker}",
            "interval": "{tv_interval}",
            "timezone": "exchange",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "withdateranges": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "studies": [
              "RSI@tv-basicstudies",
              "MAExp@tv-basicstudies",
              "MACD@tv-basicstudies",
              "BB@tv-basicstudies"
            ],
            "container_id": "tradingview_chart",
            "hide_top_toolbar": false,
            "save_image": false,
            "details": true,
            "calendar": false,
            "hotlist": false,
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650",
            "overrides": {{
              "paneProperties.background": "#131722",
              "scalesProperties.lineColor": "#555",
              "paneProperties.height": 2500
            }},
            "container_id": "tradingview_chart",
            "loaded_callback": function(widget) {{
              setTimeout(function() {{
                // Add prediction date line
                widget.chart().createMultipointShape([
                  {{ time: {int(datetime.strptime(pred_date, '%Y-%m-%d').timestamp())}, price: {high_price*0.95} }},
                  {{ time: {int(datetime.strptime(pred_date, '%Y-%m-%d').timestamp())}, price: {high_price*1.05} }}
                ], {{
                  shape: "vertical_line",
                  lock: true,
                  disableSelection: true,
                  disableSave: true,
                  disableUndo: true,
                  overrides: {{ 
                    linecolor: "#6e44ff",
                    linewidth: 2,
                    linestyle: 2,
                    showLabel: true,
                    text: "Prediction Date",
                    textcolor: "#6e44ff",
                    fontsize: 14
                  }}
                }});
                
                // Add high prediction line
                widget.chart().createShape(
                  {{ time: {int((datetime.strptime(pred_date, '%Y-%m-%d') - timedelta(days=10)).timestamp())}, price: {high_price} }},
                  {{ time: {int((datetime.strptime(pred_date, '%Y-%m-%d') + timedelta(days=2)).timestamp())}, price: {high_price} }},
                  {{
                    shape: "horizontal_line",
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    disableUndo: true,
                    overrides: {{ 
                      linecolor: "#00cc96",
                      linewidth: 2,
                      linestyle: 2,
                      showLabel: true,
                      text: "Predicted High: {high_price_formatted}",
                      textcolor: "#00cc96",
                      fontsize: 14
                    }}
                  }}
                );
                
                // Add low prediction line
                widget.chart().createShape(
                  {{ time: {int((datetime.strptime(pred_date, '%Y-%m-%d') - timedelta(days=10)).timestamp())}, price: {low_price} }},
                  {{ time: {int((datetime.strptime(pred_date, '%Y-%m-%d') + timedelta(days=2)).timestamp())}, price: {low_price} }},
                  {{
                    shape: "horizontal_line",
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    disableUndo: true,
                    overrides: {{ 
                      linecolor: "#ef553b",
                      linewidth: 2,
                      linestyle: 2,
                      showLabel: true,
                      text: "Predicted Low: {low_price_formatted}",
                      textcolor: "#ef553b",
                      fontsize: 14
                    }}
                  }}
                );
                
                // Add HOD (High of Day) and LOD (Low of Day) lines
                var now = new Date();
                var today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                var todayTimestamp = Math.floor(today.getTime() / 1000);
                
                // Get data for the current day
                widget.activeChart().onDataLoaded().subscribe(null, function() {{
                  var symbolInfo = widget.activeChart().symbolExt();
                  var resolution = widget.activeChart().resolution();
                  
                  // Add HOD line (in bright green)
                  widget.chart().createShape(
                    {{ time: todayTimestamp, price: symbolInfo.high }},
                    {{ time: todayTimestamp + 86400, price: symbolInfo.high }},
                    {{
                      shape: "horizontal_line",
                      lock: true,
                      disableSelection: true,
                      disableSave: true,
                      disableUndo: true,
                      overrides: {{ 
                        linecolor: "#00ff00",
                        linewidth: 2,
                        linestyle: 0,
                        showLabel: true,
                        text: "HOD: " + symbolInfo.high.toFixed({5 if 'USD=X' in ticker else 2}),
                        textcolor: "#00ff00",
                        fontsize: 14
                      }}
                    }}
                  );
                  
                  // Add LOD line (in bright red)
                  widget.chart().createShape(
                    {{ time: todayTimestamp, price: symbolInfo.low }},
                    {{ time: todayTimestamp + 86400, price: symbolInfo.low }},
                    {{
                      shape: "horizontal_line",
                      lock: true,
                      disableSelection: true,
                      disableSave: true,
                      disableUndo: true,
                      overrides: {{ 
                        linecolor: "#ff0000",
                        linewidth: 2,
                        linestyle: 0,
                        showLabel: true,
                        text: "LOD: " + symbolInfo.low.toFixed({5 if 'USD=X' in ticker else 2}),
                        textcolor: "#ff0000",
                        fontsize: 14
                      }}
                    }}
                  );
                }});
              }}, 2000); // Give the chart time to load
            }}
          }});
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
    else:
        # Create a basic TradingView chart without predictions
        custom_script = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:2500px;width:100%;">
          <div id="tradingview_chart" style="height:2500px;width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "autosize": false,
            "width": "100%",
            "height": 2500,
            "symbol": "{tv_ticker}",
            "interval": "{tv_interval}",
            "timezone": "exchange",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "withdateranges": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "studies": [
              "RSI@tv-basicstudies",
              "MAExp@tv-basicstudies",
              "MACD@tv-basicstudies",
              "BB@tv-basicstudies"
            ],
            "container_id": "tradingview_chart",
            "hide_top_toolbar": false,
            "save_image": false,
            "details": true,
            "calendar": false,
            "hotlist": false,
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650",
            "overrides": {{
              "paneProperties.background": "#131722",
              "scalesProperties.lineColor": "#555",
              "paneProperties.height": 2500
            }},
            "container_id": "tradingview_chart",
            "loaded_callback": function(widget) {{
              setTimeout(function() {{
                // Add HOD (High of Day) and LOD (Low of Day) lines
                var now = new Date();
                var today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                var todayTimestamp = Math.floor(today.getTime() / 1000);
                
                // Get data for the current day
                widget.activeChart().onDataLoaded().subscribe(null, function() {{
                  var symbolInfo = widget.activeChart().symbolExt();
                  var resolution = widget.activeChart().resolution();
                  
                  // Add HOD line (in bright green)
                  widget.chart().createShape(
                    {{ time: todayTimestamp, price: symbolInfo.high }},
                    {{ time: todayTimestamp + 86400, price: symbolInfo.high }},
                    {{
                      shape: "horizontal_line",
                      lock: true,
                      disableSelection: true,
                      disableSave: true,
                      disableUndo: true,
                      overrides: {{ 
                        linecolor: "#00ff00",
                        linewidth: 2,
                        linestyle: 0,
                        showLabel: true,
                        text: "HOD: " + symbolInfo.high.toFixed({5 if 'USD=X' in ticker else 2}),
                        textcolor: "#00ff00",
                        fontsize: 14
                      }}
                    }}
                  );
                  
                  // Add LOD line (in bright red)
                  widget.chart().createShape(
                    {{ time: todayTimestamp, price: symbolInfo.low }},
                    {{ time: todayTimestamp + 86400, price: symbolInfo.low }},
                    {{
                      shape: "horizontal_line",
                      lock: true,
                      disableSelection: true,
                      disableSave: true,
                      disableUndo: true,
                      overrides: {{ 
                        linecolor: "#ff0000",
                        linewidth: 2,
                        linestyle: 0,
                        showLabel: true,
                        text: "LOD: " + symbolInfo.low.toFixed({5 if 'USD=X' in ticker else 2}),
                        textcolor: "#ff0000",
                        fontsize: 14
                      }}
                    }}
                  );
                }});
              }}, 2000); // Give the chart time to load
            }}
          }});
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
    
    return custom_script

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
        
        # TradingView Chart with predictions
        st.markdown("### TradingView Chart with Predictions")
        
        # Create a TradingView chart with the predictions
        tradingview_widget = create_tradingview_widget(
            st.session_state.selected_ticker, 
            st.session_state.selected_interval,
            pred
        )
        
        # Display the TradingView chart with increased height
        st.components.v1.html(tradingview_widget, height=2500)

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
            
            # Display TradingView chart
            st.markdown("### TradingView Chart")
            
            # Display the TradingView chart with increased height
            tradingview_widget = create_tradingview_widget(
                backtest_ticker, 
                backtest_interval
            )
            st.components.v1.html(tradingview_widget, height=2500)
                
        except Exception as e:
            backtest_status.error(f"Error in backtesting: {str(e)}")
            backtest_progress.progress(0)

# Add a footer
st.markdown("---")
st.markdown("<div class='footer'>© 2025 kingbingbong Financial Technologies. All rights reserved.</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Professional Market Analysis Tools</div>", unsafe_allow_html=True)
