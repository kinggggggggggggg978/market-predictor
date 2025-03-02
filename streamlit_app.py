import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
import json

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
        height: 800px !important;
        width: 100% !important;
    }
    .tradingview-widget-container {
        height: 800px !important;
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

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'hourly_data' not in st.session_state:
    st.session_state.hourly_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'ticker_info' not in st.session_state:
    st.session_state.ticker_info = ""
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()

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
        
        # Create a custom TradingView chart with prediction lines
        custom_script = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div id="tradingview_chart"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "autosize": true,
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
              "MACD@tv-basicstudies"
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
                      text: "Predicted High: {high_price:.2f}",
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
                      text: "Predicted Low: {low_price:.2f}",
                      textcolor: "#ef553b",
                      fontsize: 14
                    }}
                  }}
                );
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
        <div class="tradingview-widget-container">
          <div id="tradingview_chart"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "autosize": true,
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
              "MACD@tv-basicstudies"
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
            }},
            "container_id": "tradingview_chart"
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
    
    # Check for restrictions based on user selections
    if prediction_type in ["Time", "Price and Time"] or bias_enabled:
        if selected_data_range not in ["1 Year", "2 Years"]:
            st.warning("For Time predictions or Bias, data range is limited to 1-2 years")
            selected_data_range = "2 Years"
        if selected_interval != "Daily":
            st.warning("For Time predictions or Bias, only Daily interval is supported")
            selected_interval = "Daily"

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
    
    # Load Data Button
    if st.button("Load Data"):
        # Convert data range to days
        data_range_map = {
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "5 Years": 1825,
            "10 Years": 3650
        }
        days = data_range_map[selected_data_range]
        
        # Convert interval to yfinance format
        interval_map = {
            "Daily": "1d",
            "Weekly": "1wk"
        }
        yf_interval = interval_map[selected_interval]
        
        # Show progress bar
        progress_bar = st.progress(0)
        
        # Fetch historical data
        st.info(f"Fetching {selected_data_range} of {selected_interval} data for {selected_ticker}...")
        
        start_date = datetime.now() - timedelta(days=days)
        
        try:
            # Fetch daily/weekly data
            progress_bar.progress(25)
            data = yf.download(selected_ticker, start=start_date, interval=yf_interval)
            
            st.session_state.data = data
            
            # Fetch live data if enabled
            if live_data_enabled:
                update_live_data(selected_ticker, selected_interval)
            
            # Fetch hourly data if needed for time predictions
            if prediction_type in ["Time", "Price and Time"] or bias_enabled:
                st.info("Fetching additional intraday data for time predictions...")
                # For time predictions, we need hourly data (limited to max 730 days)
                hourly_start = datetime.now() - timedelta(days=min(730, days))
                hourly_data = yf.download(selected_ticker, start=hourly_start, interval="1h")
                st.session_state.hourly_data = hourly_data
                progress_bar.progress(75)
            
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

# Column 3: Model Selection
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
    
    # Train models button
    if st.button("Train Models"):
        if not st.session_state.data_loaded:
            st.error("Please load data first!")
        elif not selected_models:
            st.error("Please select at least one model!")
        else:
            st.session_state.model_trained = False
            st.session_state.models = {}
            
            # Create a placeholder for the progress
            training_status = st.empty()
            training_progress = st.progress(0)
            
            # Prepare data for training
            data = st.session_state.data
            
            # Feature engineering
            df = data.copy()
            
            # Add technical indicators
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Volatility indicators
            df['ATR'] = df['High'] - df['Low']
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Add lag features
            for i in range(1, 6):
                df[f'Close_lag_{i}'] = df['Close'].shift(i)
                df[f'High_lag_{i}'] = df['High'].shift(i)
                df[f'Low_lag_{i}'] = df['Low'].shift(i)
            
            # Add day of week if daily data
            if st.session_state.selected_interval == "Daily":
                df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
                # One-hot encode day of week
                for i in range(7):
                    df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
            
            # Direction of previous candles
            df['PrevDirection'] = (df['Close'] > df['Open']).astype(int)
            
            # Range of previous candles
            df['PrevRange'] = df['High'] - df['Low']
            
            # Drop NaN values
            df = df.dropna()
            
            # Split data for training/testing
            train_data = df.copy()
            
            # Prepare target variables
            # Fix for KeyError: 'Adj Close'
            columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in train_data.columns:
                columns_to_drop.append('Adj Close')
                
            X = train_data.drop(columns_to_drop, axis=1)
            y_high = train_data['High']
            y_low = train_data['Low']
            y_direction = (train_data['Close'] > train_data['Open']).astype(int)
            
            # Scale features
            scaler_X = MinMaxScaler()
            scaler_high = MinMaxScaler()
            scaler_low = MinMaxScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_high_scaled = scaler_high.fit_transform(y_high.values.reshape(-1, 1))
            y_low_scaled = scaler_low.fit_transform(y_low.values.reshape(-1, 1))
            
            # Store scalers
            st.session_state.scaler_X = scaler_X
            st.session_state.scaler_high = scaler_high
            st.session_state.scaler_low = scaler_low
            st.session_state.feature_columns = X.columns
            
            # Split data
            X_train, X_test, y_high_train, y_high_test = train_test_split(
                X_scaled, y_high_scaled, test_size=0.2, random_state=42
            )
            _, _, y_low_train, y_low_test = train_test_split(
                X_scaled, y_low_scaled, test_size=0.2, random_state=42
            )
            _, _, y_direction_train, y_direction_test = train_test_split(
                X_scaled, y_direction.values.reshape(-1, 1), test_size=0.2, random_state=42
            )
            
            # Store training models
            models_high = {}
            models_low = {}
            models_direction = {}
            
            # Track progress
            total_models = len(selected_models) * 3  # high, low, direction
            completed_models = 0
            
            # Train each selected model
            if "Deep Learning (Neural Networks)" in selected_models:
                training_status.info("Training Deep Learning model for High prediction...")
                
                # Create model for high prediction
                model_high = Sequential()
                model_high.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model_high.add(Dropout(0.2))
                model_high.add(Dense(32, activation='relu'))
                model_high.add(Dropout(0.2))
                model_high.add(Dense(1))
                
                model_high.compile(optimizer='adam', loss='mse')
                model_high.fit(X_train, y_high_train, epochs=50, batch_size=32, verbose=0)
                models_high["Deep Learning"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                training_status.info("Training Deep Learning model for Low prediction...")
                
                # Create model for low prediction
                model_low = Sequential()
                model_low.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model_low.add(Dropout(0.2))
                model_low.add(Dense(32, activation='relu'))
                model_low.add(Dropout(0.2))
                model_low.add(Dense(1))
                
                model_low.compile(optimizer='adam', loss='mse')
                model_low.fit(X_train, y_low_train, epochs=50, batch_size=32, verbose=0)
                models_low["Deep Learning"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                training_status.info("Training Deep Learning model for Direction prediction...")
                
                # Create model for direction prediction
                model_direction = Sequential()
                model_direction.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model_direction.add(Dropout(0.2))
                model_direction.add(Dense(32, activation='relu'))
                model_direction.add(Dropout(0.2))
                model_direction.add(Dense(1, activation='sigmoid'))
                
                model_direction.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model_direction.fit(X_train, y_direction_train, epochs=50, batch_size=32, verbose=0)
                models_direction["Deep Learning"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            if "Linear Regression" in selected_models:
                training_status.info("Training Linear Regression models...")
                
                # High prediction
                model_high = LinearRegression()
                model_high.fit(X_train, y_high_train)
                models_high["Linear Regression"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Low prediction
                model_low = LinearRegression()
                model_low.fit(X_train, y_low_train)
                models_low["Linear Regression"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Direction prediction
                model_direction = LinearRegression()
                model_direction.fit(X_train, y_direction_train)
                models_direction["Linear Regression"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            if "Random Forest" in selected_models:
                training_status.info("Training Random Forest models...")
                
                # High prediction
                model_high = RandomForestRegressor(n_estimators=100, random_state=42)
                model_high.fit(X_train, y_high_train.ravel())
                models_high["Random Forest"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Low prediction
                model_low = RandomForestRegressor(n_estimators=100, random_state=42)
                model_low.fit(X_train, y_low_train.ravel())
                models_low["Random Forest"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Direction prediction
                model_direction = RandomForestRegressor(n_estimators=100, random_state=42)
                model_direction.fit(X_train, y_direction_train.ravel())
                models_direction["Random Forest"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            if "Support Vector Regression (SVR)" in selected_models:
                training_status.info("Training SVR models...")
                
                # High prediction
                model_high = SVR(kernel='rbf')
                model_high.fit(X_train, y_high_train.ravel())
                models_high["SVR"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Low prediction
                model_low = SVR(kernel='rbf')
                model_low.fit(X_train, y_low_train.ravel())
                models_low["SVR"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Direction prediction
                model_direction = SVR(kernel='rbf')
                model_direction.fit(X_train, y_direction_train.ravel())
                models_direction["SVR"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            if "K-Nearest Neighbors (KNN)" in selected_models:
                training_status.info("Training KNN models...")
                
                # High prediction
                model_high = KNeighborsRegressor(n_neighbors=5)
                model_high.fit(X_train, y_high_train.ravel())
                models_high["KNN"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Low prediction
                model_low = KNeighborsRegressor(n_neighbors=5)
                model_low.fit(X_train, y_low_train.ravel())
                models_low["KNN"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Direction prediction
                model_direction = KNeighborsRegressor(n_neighbors=5)
                model_direction.fit(X_train, y_direction_train.ravel())
                models_direction["KNN"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            if "Decision Tree (CART)" in selected_models:
                training_status.info("Training Decision Tree models...")
                
                # High prediction
                model_high = DecisionTreeRegressor(random_state=42)
                model_high.fit(X_train, y_high_train.ravel())
                models_high["Decision Tree"] = model_high
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Low prediction
                model_low = DecisionTreeRegressor(random_state=42)
                model_low.fit(X_train, y_low_train.ravel())
                models_low["Decision Tree"] = model_low
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
                
                # Direction prediction
                model_direction = DecisionTreeRegressor(random_state=42)
                model_direction.fit(X_train, y_direction_train.ravel())
                models_direction["Decision Tree"] = model_direction
                
                completed_models += 1
                training_progress.progress(completed_models / total_models)
            
            # Store models in session state
            st.session_state.models = {
                'high': models_high,
                'low': models_low,
                'direction': models_direction
            }
            
            # Time prediction models (if selected)
            if st.session_state.prediction_type in ["Time", "Price and Time"]:
                training_status.info("Training time prediction models...")
                
                # Process hourly data to find high and low times
                hourly_data = st.session_state.hourly_data
                
                # Group by day and find high/low times
                hourly_data['Date'] = hourly_data.index.date
                hourly_data['Hour'] = hourly_data.index.hour
                
                # COMPLETELY FIXED TIME FEATURE GENERATION
                # Prepare time feature data with consistent types
                time_features = []
                time_targets_high = []
                time_targets_low = []
                
                # First collect the target high and low hours for each day
                for date, group in hourly_data.groupby('Date'):
                    if len(group) >= 6:  # Ensure we have enough data for the day
                        try:
                            # Find high and low times - convert to int to ensure consistent type
                            high_hour = int(group.loc[group['High'].idxmax(), 'Hour'])
                            low_hour = int(group.loc[group['Low'].idxmin(), 'Hour'])
                            
                            time_targets_high.append(high_hour)
                            time_targets_low.append(low_hour)
                        except Exception:
                            continue
                
                # Now create features list with explicitly typed values
                for i in range(len(time_targets_high)):
                    try:
                        # Get corresponding date (same index as the target)
                        date = list(hourly_data.groupby('Date').groups.keys())[i]
                        
                        # Get day of week as an integer
                        day_of_week = int(pd.Timestamp(date).dayofweek)
                        
                        # Create feature vector (all integers)
                        feature = [day_of_week]
                        
                        # Add previous day high/low time if available
                        if i > 0:
                            feature.append(int(time_targets_high[i-1]))
                            feature.append(int(time_targets_low[i-1]))
                        else:
                            # For the first day, use its own values
                            feature.append(int(time_targets_high[i]))
                            feature.append(int(time_targets_low[i]))
                        
                        time_features.append(feature)
                    except Exception:
                        continue
                
                # Convert to numpy arrays - use dtype=int to enforce integer arrays
                if time_features:
                    time_features = np.array(time_features, dtype=int)
                    time_targets_high = np.array(time_targets_high[:len(time_features)], dtype=int)
                    time_targets_low = np.array(time_targets_low[:len(time_features)], dtype=int)
                else:
                    # Handle the case where no valid features were created
                    st.warning("Unable to create time prediction features. Using default values.")
                    time_features = np.array([[0, 9, 14]], dtype=int)
                    time_targets_high = np.array([9], dtype=int)
                    time_targets_low = np.array([14], dtype=int)
                
                # Split data
                X_time_train, X_time_test, y_time_high_train, y_time_high_test = train_test_split(
                    time_features, time_targets_high, test_size=0.2, random_state=42
                )
                _, _, y_time_low_train, y_time_low_test = train_test_split(
                    time_features, time_targets_low, test_size=0.2, random_state=42
                )
                
                # Train models for time prediction
                time_models_high = {}
                time_models_low = {}
                
                # Random Forest for time prediction
                model_time_high = RandomForestRegressor(n_estimators=100, random_state=42)
                model_time_high.fit(X_time_train, y_time_high_train)
                time_models_high["Random Forest"] = model_time_high
                
                model_time_low = RandomForestRegressor(n_estimators=100, random_state=42)
                model_time_low.fit(X_time_train, y_time_low_train)
                time_models_low["Random Forest"] = model_time_low
                
                # Store time prediction models
                st.session_state.time_models = {
                    'high': time_models_high,
                    'low': time_models_low,
                    'features': time_features[-1] if len(time_features) > 0 else np.array([0, 9, 14])  # Default values if empty
                }
                
                training_progress.progress(1.0)
            
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
        if not st.session_state.model_trained:
            st.error("Please train models first!")
        else:
            # Create a prediction placeholder
            prediction_status = st.empty()
            prediction_progress = st.progress(0)
            
            prediction_status.info("Generating predictions...")
            
            # Get the latest data
            data = st.session_state.data
            df = data.copy()
            
            # Add same features as in training
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Volatility indicators
            df['ATR'] = df['High'] - df['Low']
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Add lag features
            for i in range(1, 6):
                df[f'Close_lag_{i}'] = df['Close'].shift(i)
                df[f'High_lag_{i}'] = df['High'].shift(i)
                df[f'Low_lag_{i}'] = df['Low'].shift(i)
            
            # Add day of week if daily data
            if st.session_state.selected_interval == "Daily":
                df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
                for i in range(7):
                    df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
            
            # Direction of previous candles
            df['PrevDirection'] = (df['Close'] > df['Open']).astype(int)
            
            # Range of previous candles
            df['PrevRange'] = df['High'] - df['Low']
            
            # Get latest X values
            latest_data = df.iloc[-1:].copy()
            
            # Prepare features using the same columns as during training
            X_pred = latest_data[st.session_state.feature_columns]
            
            # Scale features
            X_pred_scaled = st.session_state.scaler_X.transform(X_pred)
            
            prediction_progress.progress(0.3)
            
            # Make predictions with each model
            high_predictions = []
            low_predictions = []
            direction_predictions = []
            
            for model_name, model in st.session_state.models['high'].items():
                # Predict high
                if model_name == "Deep Learning":
                    pred_high = model.predict(X_pred_scaled, verbose=0)
                else:
                    pred_high = model.predict(X_pred_scaled)
                
                # Inverse transform
                if pred_high.ndim == 1:
                    pred_high = pred_high.reshape(-1, 1)
                
                high_predictions.append(
                    float(st.session_state.scaler_high.inverse_transform(pred_high)[0][0])
                )
            
            for model_name, model in st.session_state.models['low'].items():
                # Predict low
                if model_name == "Deep Learning":
                    pred_low = model.predict(X_pred_scaled, verbose=0)
                else:
                    pred_low = model.predict(X_pred_scaled)
                
                # Inverse transform
                if pred_low.ndim == 1:
                    pred_low = pred_low.reshape(-1, 1)
                
                low_predictions.append(
                    float(st.session_state.scaler_low.inverse_transform(pred_low)[0][0])
                )
            
            prediction_progress.progress(0.6)
            
            # Direction prediction (if bias is enabled)
            direction_label = None
            direction_prob = 0.0
            
            if st.session_state.bias_enabled:
                for model_name, model in st.session_state.models['direction'].items():
                    # Predict direction
                    if model_name == "Deep Learning":
                        pred_direction = model.predict(X_pred_scaled, verbose=0)[0][0]
                    else:
                        pred_direction = model.predict(X_pred_scaled)[0]
                    
                    # Convert to float to avoid numpy array issues
                    direction_predictions.append(float(pred_direction))
                
                # Calculate direction probability
                direction_prob = sum(direction_predictions) / len(direction_predictions)
                direction_label = "Bullish" if direction_prob > 0.5 else "Bearish"
            
            # Time prediction (if enabled)
            time_high_prediction = None
            time_low_prediction = None
            
            if st.session_state.prediction_type in ["Time", "Price and Time"]:
                # Prepare time features
                pred_date = prediction_date
                day_of_week = pred_date.weekday()
                
                # Get previous high/low times
                last_features = st.session_state.time_models['features']
                
                # Create feature vector
                time_feature = np.array([[int(day_of_week), int(last_features[1]), int(last_features[2])]])
                
                # Predict high and low times
                time_high_prediction = int(st.session_state.time_models['high']["Random Forest"].predict(time_feature)[0])
                time_low_prediction = int(st.session_state.time_models['low']["Random Forest"].predict(time_feature)[0])
                
                # Ensure valid hour range
                time_high_prediction = max(0, min(23, time_high_prediction))
                time_low_prediction = max(0, min(23, time_low_prediction))
            
            # Calculate average predictions
            avg_high = float(sum(high_predictions) / len(high_predictions))
            avg_low = float(sum(low_predictions) / len(low_predictions))
            
            # Store predictions
            st.session_state.predictions = {
                'date': prediction_date,
                'high': avg_high,
                'low': avg_low,
                'high_time': time_high_prediction,
                'low_time': time_low_prediction,
                'direction': direction_label,
                'direction_prob': direction_prob
            }
            
            prediction_progress.progress(1.0)
            prediction_status.success("✅ Prediction generated!")

    # Display results if predictions exist
    if 'predictions' in st.session_state and st.session_state.predictions:
        pred = st.session_state.predictions
        
        st.markdown("---")
        st.subheader(f"Prediction Results for {pred['date'].strftime('%Y-%m-%d')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"**Price Predictions:**")
            st.markdown(f"<span class='prediction-value'>High: {float(pred['high']):.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='prediction-value'>Low: {float(pred['low']):.2f}</span>", unsafe_allow_html=True)
            
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
        st.components.v1.html(tradingview_widget, height=800)

# Backtesting section
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
        
        backtest_status.info("Preparing backtest data...")
        
        # Convert range to days
        range_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        test_days = range_map[backtest_range]
        train_days = 365 * 2  # 2 years for training
        
        # Get data for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days + train_days)
        
        # Convert interval to yfinance format
        interval_map = {
            "Daily": "1d",
            "Weekly": "1wk"
        }
        yf_interval = interval_map[backtest_interval]
        
        try:
            # Load data
            backtest_status.info(f"Loading data for {backtest_ticker}...")
            data = yf.download(backtest_ticker, start=start_date, end=end_date, interval=yf_interval)
            
            # Split into training and testing periods
            split_date = end_date - timedelta(days=test_days)
            train_data = data[data.index < split_date].copy()
            test_data = data[data.index >= split_date].copy()
            
            backtest_progress.progress(0.2)
            
            # Process training data
            backtest_status.info("Processing training data...")
            
            # Add features to training data
            train_df = train_data.copy()
            
            # Add technical indicators
            train_df['SMA_5'] = train_df['Close'].rolling(window=5).mean()
            train_df['SMA_20'] = train_df['Close'].rolling(window=20).mean()
            train_df['SMA_50'] = train_df['Close'].rolling(window=50).mean()
            
            train_df['EMA_5'] = train_df['Close'].ewm(span=5, adjust=False).mean()
            train_df['EMA_20'] = train_df['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate RSI
            delta = train_df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            train_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Add MACD
            train_df['EMA_12'] = train_df['Close'].ewm(span=12, adjust=False).mean()
            train_df['EMA_26'] = train_df['Close'].ewm(span=26, adjust=False).mean()
            train_df['MACD'] = train_df['EMA_12'] - train_df['EMA_26']
            train_df['MACD_signal'] = train_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Add volatility
            train_df['ATR'] = train_df['High'] - train_df['Low']
            train_df['Volatility'] = train_df['Close'].rolling(window=20).std()
            
            # Add lag features
            for i in range(1, 6):
                train_df[f'Close_lag_{i}'] = train_df['Close'].shift(i)
                train_df[f'High_lag_{i}'] = train_df['High'].shift(i)
                train_df[f'Low_lag_{i}'] = train_df['Low'].shift(i)
            
            # Add day of week if daily data
            if backtest_interval == "Daily":
                train_df['DayOfWeek'] = pd.to_datetime(train_df.index).dayofweek
                # One-hot encode day of week
                for i in range(7):
                    train_df[f'Day_{i}'] = (train_df['DayOfWeek'] == i).astype(int)
            
            # Direction of previous candles
            train_df['PrevDirection'] = (train_df['Close'] > train_df['Open']).astype(int)
            
            # Range of previous candles
            train_df['PrevRange'] = train_df['High'] - train_df['Low']
            
            # Drop NaN values
            train_df = train_df.dropna()
            
            # Prepare target variables
            # Fix for KeyError: 'Adj Close'
            columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in train_df.columns:
                columns_to_drop.append('Adj Close')
                
            X = train_df.drop(columns_to_drop, axis=1)
            y_high = train_df['High']
            y_low = train_df['Low']
            
            # Scale features
            scaler_X = MinMaxScaler()
            scaler_high = MinMaxScaler()
            scaler_low = MinMaxScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_high_scaled = scaler_high.fit_transform(y_high.values.reshape(-1, 1))
            y_low_scaled = scaler_low.fit_transform(y_low.values.reshape(-1, 1))
            
            feature_columns = X.columns
            
            backtest_progress.progress(0.4)
            
            # Train models for backtesting
            backtest_status.info("Training models for backtesting...")
            
            # Models for backtesting
            backtest_models_high = {}
            backtest_models_low = {}
            
            for model_name in backtest_models:
                if model_name == "Deep Learning (Neural Networks)":
                    # High model
                    model_high = Sequential()
                    model_high.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
                    model_high.add(Dropout(0.2))
                    model_high.add(Dense(32, activation='relu'))
                    model_high.add(Dropout(0.2))
                    model_high.add(Dense(1))
                    
                    model_high.compile(optimizer='adam', loss='mse')
                    model_high.fit(X_scaled, y_high_scaled, epochs=50, batch_size=32, verbose=0)
                    backtest_models_high["Deep Learning"] = model_high
                    
                    # Low model
                    model_low = Sequential()
                    model_low.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
                    model_low.add(Dropout(0.2))
                    model_low.add(Dense(32, activation='relu'))
                    model_low.add(Dropout(0.2))
                    model_low.add(Dense(1))
                    
                    model_low.compile(optimizer='adam', loss='mse')
                    model_low.fit(X_scaled, y_low_scaled, epochs=50, batch_size=32, verbose=0)
                    backtest_models_low["Deep Learning"] = model_low
                    
                elif model_name == "Linear Regression":
                    model_high = LinearRegression()
                    model_high.fit(X_scaled, y_high_scaled)
                    backtest_models_high["Linear Regression"] = model_high
                    
                    model_low = LinearRegression()
                    model_low.fit(X_scaled, y_low_scaled)
                    backtest_models_low["Linear Regression"] = model_low
                    
                elif model_name == "Random Forest":
                    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_high.fit(X_scaled, y_high_scaled.ravel())
                    backtest_models_high["Random Forest"] = model_high
                    
                    model_low = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_low.fit(X_scaled, y_low_scaled.ravel())
                    backtest_models_low["Random Forest"] = model_low
                    
                elif model_name == "Support Vector Regression (SVR)":
                    model_high = SVR(kernel='rbf')
                    model_high.fit(X_scaled, y_high_scaled.ravel())
                    backtest_models_high["SVR"] = model_high
                    
                    model_low = SVR(kernel='rbf')
                    model_low.fit(X_scaled, y_low_scaled.ravel())
                    backtest_models_low["SVR"] = model_low
                    
                elif model_name == "K-Nearest Neighbors (KNN)":
                    model_high = KNeighborsRegressor(n_neighbors=5)
                    model_high.fit(X_scaled, y_high_scaled.ravel())
                    backtest_models_high["KNN"] = model_high
                    
                    model_low = KNeighborsRegressor(n_neighbors=5)
                    model_low.fit(X_scaled, y_low_scaled.ravel())
                    backtest_models_low["KNN"] = model_low
                    
                elif model_name == "Decision Tree (CART)":
                    model_high = DecisionTreeRegressor(random_state=42)
                    model_high.fit(X_scaled, y_high_scaled.ravel())
                    backtest_models_high["Decision Tree"] = model_high
                    
                    model_low = DecisionTreeRegressor(random_state=42)
                    model_low.fit(X_scaled, y_low_scaled.ravel())
                    backtest_models_low["Decision Tree"] = model_low
            
            backtest_progress.progress(0.6)
            
            # Run backtest on test data
            backtest_status.info("Running backtest...")
            
            # Initialize results storage
            backtest_results = {
                'date': [],
                'actual_high': [],
                'actual_low': [],
                'predicted_high': {},
                'predicted_low': {},
                'high_error': {},
                'low_error': {},
                'high_error_pct': {},
                'low_error_pct': {}
            }
            
            # Initialize model error trackers
            for model_type in backtest_models:
                backtest_results['predicted_high'][model_type] = []
                backtest_results['predicted_low'][model_type] = []
                backtest_results['high_error'][model_type] = []
                backtest_results['low_error'][model_type] = []
                backtest_results['high_error_pct'][model_type] = []
                backtest_results['low_error_pct'][model_type] = []
            
            # Process each test day
            for i in range(len(test_data) - 1):  # Skip last day as we need the actual values
                current_date = test_data.index[i]
                next_date = test_data.index[i + 1]
                
                # Get actual values for the next day
                actual_high = float(test_data.loc[next_date, 'High'])
                actual_low = float(test_data.loc[next_date, 'Low'])
                
                # Get all data up to current date
                current_data = pd.concat([train_data, test_data.loc[:current_date]])
                
                # Process current data for prediction
                current_df = current_data.copy()
                
                # Add same technical indicators
                current_df['SMA_5'] = current_df['Close'].rolling(window=5).mean()
                current_df['SMA_20'] = current_df['Close'].rolling(window=20).mean()
                current_df['SMA_50'] = current_df['Close'].rolling(window=50).mean()
                
                current_df['EMA_5'] = current_df['Close'].ewm(span=5, adjust=False).mean()
                current_df['EMA_20'] = current_df['Close'].ewm(span=20, adjust=False).mean()
                
                # Calculate RSI
                delta = current_df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                current_df['RSI'] = 100 - (100 / (1 + rs))
                
                # Add MACD
                current_df['EMA_12'] = current_df['Close'].ewm(span=12, adjust=False).mean()
                current_df['EMA_26'] = current_df['Close'].ewm(span=26, adjust=False).mean()
                current_df['MACD'] = current_df['EMA_12'] - current_df['EMA_26']
                current_df['MACD_signal'] = current_df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Add volatility
                current_df['ATR'] = current_df['High'] - current_df['Low']
                current_df['Volatility'] = current_df['Close'].rolling(window=20).std()
                
                # Add lag features
                for j in range(1, 6):
                    current_df[f'Close_lag_{j}'] = current_df['Close'].shift(j)
                    current_df[f'High_lag_{j}'] = current_df['High'].shift(j)
                    current_df[f'Low_lag_{j}'] = current_df['Low'].shift(j)
                
                # Add day of week if daily data
                if backtest_interval == "Daily":
                    current_df['DayOfWeek'] = pd.to_datetime(current_df.index).dayofweek
                    # One-hot encode day of week
                    for j in range(7):
                        current_df[f'Day_{j}'] = (current_df['DayOfWeek'] == j).astype(int)
                
                # Direction of previous candles
                current_df['PrevDirection'] = (current_df['Close'] > current_df['Open']).astype(int)
                
                # Range of previous candles
                current_df['PrevRange'] = current_df['High'] - current_df['Low']
                
                # Get latest data point for prediction
                latest_data = current_df.iloc[-1:].copy()
                
                # Ensure all feature columns exist
                for col in feature_columns:
                    if col not in latest_data.columns:
                        latest_data[col] = 0
                
                # Prepare features using the same columns as during training
                X_pred = latest_data[feature_columns]
                
                # Scale features
                X_pred_scaled = scaler_X.transform(X_pred)
                
                # Make predictions with each model
                for model_name, model in backtest_models_high.items():
                    model_key = next(k for k in backtest_models if k.startswith(model_name))
                    
                    # Predict high
                    if model_name == "Deep Learning":
                        pred_high = model.predict(X_pred_scaled, verbose=0)
                    else:
                        pred_high = model.predict(X_pred_scaled)
                    
                    # Inverse transform
                    if pred_high.ndim == 1:
                        pred_high = pred_high.reshape(-1, 1)
                        
                    predicted_high = float(scaler_high.inverse_transform(pred_high)[0][0])
                    
                    # Calculate error
                    high_error = predicted_high - actual_high
                    high_error_pct = (high_error / actual_high) * 100
                    
                    # Store results
                    backtest_results['predicted_high'][model_key].append(predicted_high)
                    backtest_results['high_error'][model_key].append(high_error)
                    backtest_results['high_error_pct'][model_key].append(high_error_pct)
                
                for model_name, model in backtest_models_low.items():
                    model_key = next(k for k in backtest_models if k.startswith(model_name))
                    
                    # Predict low
                    if model_name == "Deep Learning":
                        pred_low = model.predict(X_pred_scaled, verbose=0)
                    else:
                        pred_low = model.predict(X_pred_scaled)
                    
                    # Inverse transform
                    if pred_low.ndim == 1:
                        pred_low = pred_low.reshape(-1, 1)
                        
                    predicted_low = float(scaler_low.inverse_transform(pred_low)[0][0])
                    
                    # Calculate error
                    low_error = predicted_low - actual_low
                    low_error_pct = (low_error / actual_low) * 100
                    
                    # Store results
                    backtest_results['predicted_low'][model_key].append(predicted_low)
                    backtest_results['low_error'][model_key].append(low_error)
                    backtest_results['low_error_pct'][model_key].append(low_error_pct)
                
                # Store actual values
                backtest_results['date'].append(next_date)
                backtest_results['actual_high'].append(actual_high)
                backtest_results['actual_low'].append(actual_low)
                
                # Update progress
                backtest_progress.progress(0.6 + (0.4 * (i + 1) / len(test_data)))
            
            # Calculate accuracy metrics
            backtest_status.info("Calculating accuracy metrics...")
            
            accuracy_metrics = {}
            
            for model_name in backtest_models:
                mae_high = float(np.mean(np.abs(backtest_results['high_error'][model_name])))
                mae_low = float(np.mean(np.abs(backtest_results['low_error'][model_name])))
                
                mape_high = float(np.mean(np.abs(backtest_results['high_error_pct'][model_name])))
                mape_low = float(np.mean(np.abs(backtest_results['low_error_pct'][model_name])))
                
                # Average accuracy
                accuracy_high = 100 - mape_high
                accuracy_low = 100 - mape_low
                avg_accuracy = (accuracy_high + accuracy_low) / 2
                
                accuracy_metrics[model_name] = {
                    'mae_high': mae_high,
                    'mae_low': mae_low,
                    'mape_high': mape_high,
                    'mape_low': mape_low,
                    'accuracy': avg_accuracy
                }
            
            # Display backtesting results
            backtest_status.success("✅ Backtesting completed!")
            backtest_progress.progress(1.0)
            
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
            
            # Plot error distribution
            st.markdown("### Error Distribution")
            
            # Create tabs for High and Low predictions
            tabs = st.tabs(["High Prediction Errors", "Low Prediction Errors"])
            
            with tabs[0]:
                # High prediction errors
                fig = go.Figure()
                
                for model in backtest_models:
                    # Convert error values to Python floats to avoid numpy array issues
                    errors = [float(x) for x in backtest_results['high_error_pct'][model]]
                    
                    fig.add_trace(go.Box(
                        y=errors,
                        name=model,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                
                fig.update_layout(
                    title="High Prediction Error Distribution (%)",
                    yaxis_title="Error (%)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                # Low prediction errors
                fig = go.Figure()
                
                for model in backtest_models:
                    # Convert error values to Python floats to avoid numpy array issues
                    errors = [float(x) for x in backtest_results['low_error_pct'][model]]
                    
                    fig.add_trace(go.Box(
                        y=errors,
                        name=model,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                
                fig.update_layout(
                    title="Low Prediction Error Distribution (%)",
                    yaxis_title="Error (%)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display TradingView chart with backtesting results
            st.markdown("### TradingView Chart")
            
            # Create a sample prediction for the current date to show on TradingView
            sample_prediction = {
                'date': datetime.now().date(),
                'high': backtest_results['actual_high'][-1] * 1.01,  # Just for visualization
                'low': backtest_results['actual_low'][-1] * 0.99  # Just for visualization
            }
            
            # Display the TradingView chart with increased height
            tradingview_widget = create_tradingview_widget(
                backtest_ticker, 
                backtest_interval
            )
            st.components.v1.html(tradingview_widget, height=800)
                
        except Exception as e:
            backtest_status.error(f"Error in backtesting: {str(e)}")
            backtest_progress.progress(0)

# Add a footer
st.markdown("---")
st.markdown("<div class='footer'>© 2025 kingbingbong Financial Technologies. All rights reserved.</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Professional Market Analysis Tools</div>", unsafe_allow_html=True)
