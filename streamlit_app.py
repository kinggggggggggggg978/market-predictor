import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import datetime
import time
import math

# Set page config
st.set_page_config(
    page_title="Market Prediction Simulator",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'traders' not in st.session_state:
    st.session_state.traders = [
        {"id": 1, "name": "Trend Follower", "balance": 10000, "color": "#4338ca", "wins": 0, 
         "strategy": "trend", "learningRate": 0.05, "accuracy": {"hod": 0, "lod": 0}, 
         "predictions": [], "maxDrawdown": 0, "consecutiveWins": 0, "consecutiveLosses": 0},
        {"id": 2, "name": "Mean Reversal", "balance": 10000, "color": "#059669", "wins": 0, 
         "strategy": "reversal", "learningRate": 0.08, "accuracy": {"hod": 0, "lod": 0}, 
         "predictions": [], "maxDrawdown": 0, "consecutiveWins": 0, "consecutiveLosses": 0},
        {"id": 3, "name": "Volatility Breakout", "balance": 10000, "color": "#d97706", "wins": 0, 
         "strategy": "volatility", "learningRate": 0.07, "accuracy": {"hod": 0, "lod": 0}, 
         "predictions": [], "maxDrawdown": 0, "consecutiveWins": 0, "consecutiveLosses": 0}
    ]
if 'active_trader' not in st.session_state:
    st.session_state.active_trader = 1
if 'day' not in st.session_state:
    st.session_state.day = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'selected_ai' not in st.session_state:
    st.session_state.selected_ai = None
if 'current_price' not in st.session_state:
    st.session_state.current_price = 0
if 'round_ended' not in st.session_state:
    st.session_state.round_ended = False
if 'pattern_type' not in st.session_state:
    st.session_state.pattern_type = 'all'
if 'market_label' not in st.session_state:
    st.session_state.market_label = 'Mixed Market'
if 'current_regime' not in st.session_state:
    st.session_state.current_regime = 'mixed'
if 'show_all' not in st.session_state:
    st.session_state.show_all = False
if 'auto_training' not in st.session_state:
    st.session_state.auto_training = False
if 'trader_networks' not in st.session_state:
    st.session_state.trader_networks = {
        1: {
            "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "biasHod": np.random.uniform(-0.1, 0.1),
            "biasLod": np.random.uniform(-0.1, 0.1)
        },
        2: {
            "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "biasHod": np.random.uniform(-0.1, 0.1),
            "biasLod": np.random.uniform(-0.1, 0.1)
        },
        3: {
            "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
            "biasHod": np.random.uniform(-0.1, 0.1),
            "biasLod": np.random.uniform(-0.1, 0.1)
        }
    }
if 'system_messages' not in st.session_state:
    st.session_state.system_messages = []
if 'training_cycles' not in st.session_state:
    st.session_state.training_cycles = 0
if 'regimes_completed' not in st.session_state:
    st.session_state.regimes_completed = []

# Add CSS styling
st.markdown("""
<style>
    .trader-card {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        transition: all 0.3s;
    }
    .trader-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .trader-active {
        background-color: #1e293b;
        color: white;
        transform: scale(1.02);
    }
    .trader-selected {
        background-color: #1d4ed8;
        color: white;
    }
    .trader-normal {
        background-color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .win-badge {
        background-color: #e2e8f0;
        color: #475569;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .prediction-success {
        color: #22c55e;
        font-weight: 600;
    }
    .prediction-failure {
        color: #ef4444;
        font-weight: 600;
    }
    .strategy-tag {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 5px;
    }
    .select-button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        cursor: pointer;
    }
    .log-container {
        height: 200px;
        overflow-y: auto;
        background-color: white;
        border-radius: 4px;
        padding: 8px;
    }
    .message-info {
        background-color: #e0f2fe;
        color: #0c4a6e;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .message-success {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .message-warning {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Function to add system messages
def add_system_message(message, message_type="info"):
    new_message = {
        "id": int(time.time() * 1000),
        "text": message,
        "type": message_type,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.system_messages.insert(0, new_message)
    # Limit to the most recent 50 messages
    st.session_state.system_messages = st.session_state.system_messages[:50]

# Detect market structure/regime
def detect_market_structure(data, lookback=50):
    if len(data) < lookback:
        return "unknown"
    
    recent_data = data.tail(lookback)
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    closes = recent_data['close'].values
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_highs.append({"index": i, "value": highs[i]})
        
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swing_lows.append({"index": i, "value": lows[i]})
    
    # Determine market structure based on swing highs and lows
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]
        
        if (last_two_highs[1]["value"] > last_two_highs[0]["value"] and 
            last_two_lows[1]["value"] > last_two_lows[0]["value"]):
            return "uptrend"  # Higher highs and higher lows
        elif (last_two_highs[1]["value"] < last_two_highs[0]["value"] and 
              last_two_lows[1]["value"] < last_two_lows[0]["value"]):
            return "downtrend"  # Lower highs and lower lows
        elif (abs(last_two_highs[1]["value"] - last_two_highs[0]["value"]) / last_two_highs[0]["value"] < 0.01 and
              abs(last_two_lows[1]["value"] - last_two_lows[0]["value"]) / last_two_lows[0]["value"] < 0.01):
            return "range"  # Flat highs and lows
    
    # Check for range-bound market
    highest_high = np.max(highs)
    lowest_low = np.min(lows)
    range_pct = (highest_high - lowest_low) / lowest_low
    
    if range_pct < 0.03:
        return "range"
    
    # Check for choppy market
    changes = 0
    direction = 1 if closes[1] > closes[0] else -1
    
    for i in range(2, len(closes)):
        new_direction = 1 if closes[i] > closes[i-1] else -1
        if new_direction != direction:
            changes += 1
            direction = new_direction
    
    if changes > lookback * 0.4:
        return "choppy"
    
    return "mixed"

# Generate realistic market data
def generate_data(forced_pattern=None):
    # Create market cycles and patterns with 1000 candles
    data_length = 1000
    data = []
    price = 100
    
    # Pattern to use (from parameter or state)
    selected_pattern = forced_pattern if forced_pattern else st.session_state.pattern_type
    
    # Default market label
    market_label = "Mixed Market"
    regime_type = "mixed"
    
    if selected_pattern == 'uptrend':
        market_label = "Strong Uptrend Market"
        regime_type = "uptrend"
    elif selected_pattern == 'downtrend':
        market_label = "Strong Downtrend Market"
        regime_type = "downtrend"
    elif selected_pattern == 'range':
        market_label = "Range-Bound Market"
        regime_type = "range"
    elif selected_pattern == 'choppy':
        market_label = "Choppy Market"
        regime_type = "choppy"
    
    st.session_state.market_label = market_label
    st.session_state.current_regime = regime_type
    
    # Pre-defined patterns
    def generate_uptrend_candle(i, base_price):
        volatility = 1.5 + math.sin(i/30) * 0.5
        trend_strength = 0.15
        change = (random.random() - 0.35) * volatility + trend_strength
        open_price = base_price
        close_price = base_price + change
        high_price = max(open_price, close_price) + random.random() * volatility * 0.5
        low_price = min(open_price, close_price) - random.random() * volatility * 0.3
        return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
    
    def generate_downtrend_candle(i, base_price):
        volatility = 1.5 + math.cos(i/20) * 0.5
        trend_strength = -0.18
        change = (random.random() - 0.65) * volatility + trend_strength
        open_price = base_price
        close_price = base_price + change
        high_price = max(open_price, close_price) + random.random() * volatility * 0.3
        low_price = min(open_price, close_price) - random.random() * volatility * 0.5
        return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
    
    def generate_range_candle(i, base_price):
        center_price = 100  # Center of the range
        range_size = 8      # Size of the range (plus/minus from center)
        
        # Calculate a mean-reverting price
        distance_from_center = base_price - center_price
        reversion = distance_from_center * 0.1  # Strength of reversion
        
        volatility = 1.0
        change = (random.random() - 0.5) * volatility - reversion
        
        open_price = base_price
        close_price = base_price + change
        high_price = max(open_price, close_price) + random.random() * volatility * 0.4
        low_price = min(open_price, close_price) - random.random() * volatility * 0.4
        return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
    
    def generate_choppy_candle(i, base_price):
        # Create zigzag pattern
        cycle = math.sin(i/5) * math.cos(i/3)  # Creates frequent reversals
        volatility = 1.2
        change = cycle * volatility + (random.random() - 0.5) * volatility * 0.5
        
        open_price = base_price
        close_price = base_price + change
        high_price = max(open_price, close_price) + random.random() * volatility * 0.3
        low_price = min(open_price, close_price) - random.random() * volatility * 0.3
        return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
    
    def generate_default_candle(i, base_price):
        volatility = 1.0
        change = (random.random() - 0.5) * volatility
        open_price = base_price
        close_price = base_price + change
        high_price = max(open_price, close_price) + random.random() * volatility * 0.3
        low_price = min(open_price, close_price) - random.random() * volatility * 0.3
        return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
    
    # Map pattern types to candle generation functions
    pattern_functions = {
        'uptrend': generate_uptrend_candle,
        'downtrend': generate_downtrend_candle,
        'range': generate_range_candle,
        'choppy': generate_choppy_candle,
        'default': generate_default_candle
    }
    
    # Market structure with multiple segments
    segments = []
    current_position = 0
    
    if selected_pattern == 'all' or selected_pattern == 'mixed':
        # Create a complex, realistic market with multiple segments
        while current_position < data_length:
            segment_length = random.randint(50, 200)  # 50-200 candles per segment
            pattern_keys = ['uptrend', 'downtrend', 'range', 'choppy']
            pattern = random.choice(pattern_keys)
            segments.append({"start": current_position, "length": segment_length, "pattern": pattern})
            current_position += segment_length
    else:
        # Create a single pattern type market
        segments.append({"start": 0, "length": data_length, "pattern": selected_pattern})
    
    # Generate the actual data
    for i in range(data_length):
        # Find which segment contains this position
        active_segment = None
        for segment in segments:
            if i >= segment["start"] and i < segment["start"] + segment["length"]:
                active_segment = segment
                break
        
        # Generate candle
        candle_function = pattern_functions.get('default')
        if active_segment:
            candle_function = pattern_functions.get(active_segment["pattern"], pattern_functions['default'])
        
        candle = candle_function(i, price)
        
        # Apply the price change
        price += candle["change"]
        
        # Ensure price doesn't go below 10
        price = max(price, 10)
        
        # Generate volume based on price movement
        volume_base = 1000
        volume_variation = abs(candle["change"]) * 1000
        volume = int(volume_base + volume_variation + random.random() * 500)
        
        # Create the candle data
        new_candle = {
            "time": i,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": price,
            "volume": volume
        }
        
        data.append(new_candle)
    
    # Convert to DataFrame and update state
    df = pd.DataFrame(data)
    st.session_state.data = df
    st.session_state.current_price = df['close'].iloc[0]

# Extract features for predictions
def extract_features(data):
    if len(data) < 20:
        return None
    
    # Get most recent candles
    recent_data = data.tail(20)
    
    # Calculate technical indicators and other features
    closes = recent_data['close'].values
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    volumes = recent_data['volume'].values
    
    # Moving averages
    sma5 = calculate_sma(closes, 5)
    sma10 = calculate_sma(closes, 10)
    sma20 = calculate_sma(closes, 20)
    
    # Price momentum
    momentum5 = closes[-1] / closes[-6] if len(closes) >= 6 else 1.0
    momentum10 = closes[-1] / closes[-11] if len(closes) >= 11 else 1.0
    
    # Volatility (ATR-like)
    volatility = calculate_atr(recent_data, 14)
    
    # Volume trend
    volume_change = volumes[-1] / calculate_sma(volumes, 5) if len(volumes) >= 5 else 1.0
    
    # Calculate relative position of current price within recent range
    recent_highest = np.max(highs)
    recent_lowest = np.min(lows)
    price_position = (closes[-1] - recent_lowest) / (recent_highest - recent_lowest) if recent_highest != recent_lowest else 0.5
    
    # Calculate recent high/low relationship to predict if new highs/lows are likely
    highest_high_idx = np.argmax(highs)
    lowest_low_idx = np.argmin(lows)
    days_since_high = len(highs) - 1 - highest_high_idx
    days_since_low = len(lows) - 1 - lowest_low_idx
    
    # Create feature vector
    features = [
        closes[-1] / sma5,                  # Price relative to SMA5
        closes[-1] / sma10,                 # Price relative to SMA10
        closes[-1] / sma20,                 # Price relative to SMA20
        momentum5,                          # 5-day momentum
        momentum10,                         # 10-day momentum
        volatility / closes[-1],            # Normalized volatility
        volume_change,                      # Volume trend
        price_position,                     # Relative price position
        1.0 / (days_since_high + 1),        # Recency of highest high
        1.0 / (days_since_low + 1)          # Recency of lowest low
    ]
    
    return features

# Calculate Simple Moving Average
def calculate_sma(data, period):
    if len(data) < period:
        return data[-1] if len(data) > 0 else 0
    
    return np.mean(data[-period:])

# Calculate Average True Range
def calculate_atr(candles, period):
    if len(candles) < 2:
        return 0
    
    true_ranges = []
    for i in range(1, len(candles)):
        high_low = candles['high'].iloc[i] - candles['low'].iloc[i]
        high_prev_close = abs(candles['high'].iloc[i] - candles['close'].iloc[i-1])
        low_prev_close = abs(candles['low'].iloc[i] - candles['close'].iloc[i-1])
        
        true_ranges.append(max(high_low, high_prev_close, low_prev_close))
    
    if len(true_ranges) < period:
        return np.mean(true_ranges) if true_ranges else 0
    
    return np.mean(true_ranges[-period:])

# Neural network prediction function
def predict_with_network(features, weights, bias):
    if not features:
        return 0
    
    result = bias
    for i in range(len(features)):
        if i < len(weights):
            result += features[i] * weights[i]
    
    return result

# Make predictions for HOD/LOD
def make_predictions(trader_id, data, look_ahead=5):
    if len(data) < 20:
        return None
    
    features = extract_features(data)
    if not features:
        return None
    
    network = st.session_state.trader_networks[trader_id]
    trader = next((t for t in st.session_state.traders if t["id"] == trader_id), None)
    
    # Get current close price for reference
    current_close = data['close'].iloc[-1]
    
    # Get recent highs/lows for context
    recent_data = data.tail(20)
    recent_highs = recent_data['high'].values
    recent_lows = recent_data['low'].values
    recent_highest = np.max(recent_highs)
    recent_lowest = np.min(recent_lows)
    
    # Adjust prediction strategy based on trader's specialty
    hod_multiplier = 1.0
    lod_multiplier = 1.0
    
    if trader and trader["strategy"] == 'trend':
        # Trend followers expect more extreme values in the trending direction
        market_trend = detect_market_structure(data)
        if market_trend == 'uptrend':
            hod_multiplier = 1.2
            lod_multiplier = 0.9
        elif market_trend == 'downtrend':
            hod_multiplier = 0.9
            lod_multiplier = 1.2
    elif trader and trader["strategy"] == 'reversal':
        # Mean reversal expects moves counter to recent extremes
        if current_close > recent_highest * 0.95:
            hod_multiplier = 0.9
            lod_multiplier = 1.1
        elif current_close < recent_lowest * 1.05:
            hod_multiplier = 1.1
            lod_multiplier = 0.9
    elif trader and trader["strategy"] == 'volatility':
        # Volatility strategies expect larger ranges in choppy or volatile markets
        volatility = calculate_atr(recent_data, 14) / current_close
        if volatility > 0.01:
            hod_multiplier = 1.1
            lod_multiplier = 1.1
    
    # Make raw predictions using neural network weights
    raw_hod_prediction = predict_with_network(features, network["hodWeights"], network["biasHod"])
    raw_lod_prediction = predict_with_network(features, network["lodWeights"], network["biasLod"])
    
    # Apply strategy-based adjustments and scaling
    # Normalize to get percentage change from current price
    hod_pct_change = 0.02 * raw_hod_prediction * hod_multiplier  # Max ~2% move
    lod_pct_change = -0.02 * raw_lod_prediction * lod_multiplier  # Max ~2% move
    
    # Calculate actual price predictions
    hod_prediction = current_close * (1 + hod_pct_change)
    lod_prediction = current_close * (1 + lod_pct_change)
    
    # Ensure LOD is lower than HOD
    if lod_prediction >= hod_prediction:
        mid_point = (hod_prediction + lod_prediction) / 2
        hod_prediction = mid_point * 1.01
        lod_prediction = mid_point * 0.99
    
    return {
        "hodPrediction": hod_prediction,
        "lodPrediction": lod_prediction,
        "features": features,
        "currentClose": current_close
    }

# Update neural network weights through reinforcement learning
def update_network_weights(trader_id, features, actual, prediction, is_hod):
    # Calculate error
    error = actual - prediction
    error_pct = abs(error / actual)
    
    # Only reward/penalize if significant error
    if error_pct < 0.0025:
        return True  # Consider as correct if within 0.25%
    
    # Get current weights
    network = st.session_state.trader_networks[trader_id].copy()
    trader = next((t for t in st.session_state.traders if t["id"] == trader_id), None)
    weights = network["hodWeights"] if is_hod else network["lodWeights"]
    bias = network["biasHod"] if is_hod else network["biasLod"]
    
    # Adjust weights and bias based on error
    learning_rate = trader["learningRate"] * (1.2 if error_pct > 0.01 else 0.8)  # Adjust rate based on error size
    
    # Reinforcement learning: only update weights if prediction was incorrect
    if error_pct > 0.005:
        # Update weights based on error and features
        updated_weights = []
        for i in range(len(weights)):
            if i < len(features):
                sign = 1 if actual > prediction else -1
                new_weight = weights[i] + sign * learning_rate * features[i] * error_pct
                updated_weights.append(new_weight)
            else:
                updated_weights.append(weights[i])
        
        # Update bias
        updated_bias = bias + learning_rate * (1 if actual > prediction else -1) * error_pct
        
        # Update the network
        if is_hod:
            network["hodWeights"] = updated_weights
            network["biasHod"] = updated_bias
        else:
            network["lodWeights"] = updated_weights
            network["biasLod"] = updated_bias
        
        st.session_state.trader_networks[trader_id] = network
    
    return error_pct <= 0.01  # Success if error is less than 1%

# Evaluate prediction accuracy
def evaluate_prediction(trader_id, prediction, actual_data):
    if not prediction or actual_data.empty:
        return None
    
    actual_hod = actual_data['high'].max()
    actual_lod = actual_data['low'].min()
    
    # Calculate accuracy as percentage error
    hod_error = abs((prediction["hodPrediction"] - actual_hod) / actual_hod)
    lod_error = abs((prediction["lodPrediction"] - actual_lod) / actual_lod)
    
    # Success if error is less than thresholds (1% for now)
    hod_success = update_network_weights(trader_id, prediction["features"], actual_hod, prediction["hodPrediction"], True)
    lod_success = update_network_weights(trader_id, prediction["features"], actual_lod, prediction["lodPrediction"], False)
    
    # Calculate accuracy scores (100% - error%)
    hod_accuracy = max(0, 100 - (hod_error * 100))
    lod_accuracy = max(0, 100 - (lod_error * 100))
    
    # Return evaluation results
    return {
        "hodSuccess": hod_success,
        "lodSuccess": lod_success,
        "hodAccuracy": hod_accuracy,
        "lodAccuracy": lod_accuracy,
        "actualHod": actual_hod,
        "actualLod": actual_lod,
        "hodError": hod_error,
        "lodError": lod_error
    }

# Update trader statistics
def update_trader_stats(trader, evaluation):
    if not evaluation:
        return trader
    
    hod_success = evaluation["hodSuccess"]
    lod_success = evaluation["lodSuccess"]
    hod_accuracy = evaluation["hodAccuracy"]
    lod_accuracy = evaluation["lodAccuracy"]
    
    # Calculate success rate for this prediction
    overall_success = (hod_success and lod_success) or (hod_success and lod_accuracy > 95) or (lod_success and hod_accuracy > 95)
    
    # Update consecutive win/loss streaks
    consecutive_wins = trader["consecutiveWins"]
    consecutive_losses = trader["consecutiveLosses"]
    
    if overall_success:
        consecutive_wins += 1
        consecutive_losses = 0
    else:
        consecutive_wins = 0
        consecutive_losses += 1
    
    # Update accuracy metrics with exponential moving average
    alpha = 0.05  # Weight for newest data
    new_hod_accuracy = alpha * hod_accuracy + (1 - alpha) * trader["accuracy"]["hod"]
    new_lod_accuracy = alpha * lod_accuracy + (1 - alpha) * trader["accuracy"]["lod"]
    
    # Update trader object
    updated_trader = trader.copy()
    updated_trader["wins"] = trader["wins"] + (1 if overall_success else 0)
    updated_trader["accuracy"] = {
        "hod": new_hod_accuracy,
        "lod": new_lod_accuracy
    }
    updated_trader["consecutiveWins"] = consecutive_wins
    updated_trader["consecutiveLosses"] = consecutive_losses
    
    return updated_trader

# Reset simulation
def reset_simulation():
    st.session_state.day = 0
    st.session_state.round_ended = False
    
    # Reset traders
    updated_traders = []
    for trader in st.session_state.traders:
        updated_trader = trader.copy()
        updated_trader["predictions"] = []
        updated_trader["wins"] = 0
        updated_trader["consecutiveWins"] = 0
        updated_trader["consecutiveLosses"] = 0
        updated_traders.append(updated_trader)
    
    st.session_state.traders = updated_traders
    st.session_state.running = False

# Select an AI trader
def select_ai(trader_id):
    st.session_state.selected_ai = trader_id
    trader = next((t for t in st.session_state.traders if t["id"] == trader_id), None)
    if trader:
        add_system_message(f"Selected {trader['name']} as primary prediction AI", "success")

# Start auto-training
def start_auto_training():
    st.session_state.auto_training = True
    st.session_state.training_cycles = 0
    st.session_state.regimes_completed = []
    add_system_message("Starting automated training across all market conditions...", "info")
    generate_data("uptrend")
    reset_simulation()
    st.session_state.running = True

# Main app layout
st.title("Advanced Market Prediction Simulator")

# Top section with controls and market type
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    pattern_options = {
        'all': 'All Market Types',
        'mixed': 'Mixed Market',
        'uptrend': 'Strong Uptrend',
        'downtrend': 'Strong Downtrend',
        'range': 'Range-Bound',
        'choppy': 'Choppy Market'
    }
    
    selected_pattern = st.selectbox(
        "Market Type",
        options=list(pattern_options.keys()),
        format_func=lambda x: pattern_options[x],
        index=list(pattern_options.keys()).index(st.session_state.pattern_type),
        disabled=st.session_state.auto_training
    )
    
    if selected_pattern != st.session_state.pattern_type:
        st.session_state.pattern_type = selected_pattern
        generate_data()
        reset_simulation()

with col2:
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        st.metric("Current Day", f"{st.session_state.day}/{1000}")
    
    with col2_2:
        st.metric("Market Regime", st.session_state.market_label)

with col3:
    col3_1, col3_2 = st.columns(2)
    
    with col3_1:
        if st.button("New Market Data", disabled=st.session_state.auto_training):
            generate_data()
            reset_simulation()
    
    with col3_2:
        if not st.session_state.auto_training:
            if st.button("Start Auto-Training"):
                start_auto_training()
        else:
            st.info("Auto-Training in Progress...")

# Trader cards
st.subheader("AI Traders")
trader_cols = st.columns(3)

for i, trader in enumerate(st.session_state.traders):
    with trader_cols[i]:
        is_active = trader["id"] == st.session_state.active_trader
        is_selected = trader["id"] == st.session_state.selected_ai
        
        # Calculate stats
        total_predictions = len(trader["predictions"])
        evaluated_predictions = max(0, total_predictions - 5)
        
        # Card container with dynamic styling
        card_class = "trader-active" if is_active else "trader-selected" if is_selected else "trader-normal"
        st.markdown(f"""
        <div class="trader-card {card_class}" onclick="selectTrader({trader['id']})">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {trader['color']}; margin-right: 8px;"></div>
                    <h3 style="font-weight: 600; margin: 0;">{trader['name']}</h3>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="win-badge">Wins: {trader['wins']}</span>
                    {'' if is_selected else '<button class="select-button" onclick="selectAI('+str(trader['id'])+')">Select</button>'}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
                <div>
                    <div class="metric-label">HOD Accuracy</div>
                    <div class="metric-value">{trader['accuracy']['hod']:.2f}%</div>
                </div>
                <div>
                    <div class="metric-label">LOD Accuracy</div>
                    <div class="metric-value">{trader['accuracy']['lod']:.2f}%</div>
                </div>
                <div>
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{(trader['wins'] / max(1, evaluated_predictions) * 100):.2f}%</div>
                </div>
                <div>
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{(trader['consecutiveLosses'] * 5):.2f}%</div>
                </div>
            </div>
            
            <div class="strategy-tag">
                Strategy: {
                    'Trend Following' if trader['strategy'] == 'trend' else
                    'Mean Reversal' if trader['strategy'] == 'reversal' else
                    'Volatility Breakout'
                }
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button to handle click event (workaround for div onclick)
        if st.button(f"Select {trader['name']}", key=f"trader_btn_{trader['id']}", help="Click to view this trader's details"):
            st.session_state.active_trader = trader["id"]
            st.experimental_rerun()

# Chart section
st.subheader("Market Chart with Predictions")

# Get data for the chart
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Determine the visible window (70 candles)
    visible_candles = 70
    start_idx = max(0, st.session_state.day - visible_candles + 5)
    end_idx = st.session_state.day + 1
    display_data = data.iloc[start_idx:end_idx].copy()
    
    # Get active trader predictions for the chart
    active_traders = []
    for trader in st.session_state.traders:
        if st.session_state.show_all or trader["id"] == st.session_state.active_trader or trader["id"] == st.session_state.selected_ai:
            # Find the most recent prediction
            latest_prediction = None
            if trader["predictions"]:
                for pred in reversed(trader["predictions"]):
                    if pred["day"] == st.session_state.day - 1:
                        latest_prediction = pred
                        break
            
            if latest_prediction and latest_prediction["prediction"]:
                active_traders.append({
                    "traderId": trader["id"],
                    "name": trader["name"],
                    "color": trader["color"],
                    "prediction": latest_prediction["prediction"]
                })
    
    # Create the candlestick chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         row_heights=[0.8, 0.2], 
                         vertical_spacing=0.02,
                         specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=display_data.index,
            open=display_data['open'],
            high=display_data['high'],
            low=display_data['low'],
            close=display_data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume trace
    fig.add_trace(
        go.Bar(
            x=display_data.index,
            y=display_data['volume'],
            name="Volume",
            marker=dict(color='rgba(100, 100, 100, 0.5)')
        ),
        row=2, col=1
    )
    
    # Add prediction lines for each active trader
    for trader in active_traders:
        # Add HOD prediction line
        fig.add_trace(
            go.Scatter(
                x=[display_data.index[-1], display_data.index[-1] + 5],
                y=[trader["prediction"]["hodPrediction"], trader["prediction"]["hodPrediction"]],
                mode='lines',
                line=dict(color=trader["color"], width=2, dash='dash'),
                name=f"{trader['name']} HOD"
            ),
            row=1, col=1
        )
        
        # Add HOD label
        fig.add_annotation(
            x=display_data.index[-1] + 5,
            y=trader["prediction"]["hodPrediction"],
            text=f"HOD: {trader['prediction']['hodPrediction']:.2f}",
            showarrow=False,
            font=dict(color=trader["color"]),
            xanchor="left",
            row=1, col=1
        )
        
        # Add LOD prediction line
        fig.add_trace(
            go.Scatter(
                x=[display_data.index[-1], display_data.index[-1] + 5],
                y=[trader["prediction"]["lodPrediction"], trader["prediction"]["lodPrediction"]],
                mode='lines',
                line=dict(color=trader["color"], width=2, dash='dash'),
                name=f"{trader['name']} LOD"
            ),
            row=1, col=1
        )
        
        # Add LOD label
        fig.add_annotation(
            x=display_data.index[-1] + 5,
            y=trader["prediction"]["lodPrediction"],
            text=f"LOD: {trader['prediction']['lodPrediction']:.2f}",
            showarrow=False,
            font=dict(color=trader["color"]),
            xanchor="left",
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        title="Market Price Chart with Predictions",
        xaxis_title="Day",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Fix y-axis scaling
    if not display_data.empty:
        y_min = display_data['low'].min() * 0.98
        y_max = display_data['high'].max() * 1.02
        
        # Adjust for prediction lines that might be outside the range
        for trader in active_traders:
            y_min = min(y_min, trader["prediction"]["lodPrediction"] * 0.98)
            y_max = max(y_max, trader["prediction"]["hodPrediction"] * 1.02)
        
        fig.update_layout(yaxis_range=[y_min, y_max])
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Generate data to see the chart")

# Control panel
st.subheader("Simulation Controls")
control_cols = st.columns([1, 1, 1, 1])

with control_cols[0]:
    if st.button("▶️ Start" if not st.session_state.running else "⏸️ Pause", disabled=st.session_state.auto_training and not st.session_state.running):
        st.session_state.running = not st.session_state.running

with control_cols[1]:
    if st.button("🔄 Reset", disabled=st.session_state.auto_training):
        reset_simulation()

with control_cols[2]:
    # Simulation speed slider
    speed_options = {
        1000: "Slow",
        500: "Medium",
        200: "Fast",
        50: "Ultra Fast",
        10: "Maximum"
    }
    
    speed = st.selectbox(
        "Simulation Speed",
        options=list(speed_options.keys()),
        format_func=lambda x: speed_options[x],
        index=1  # Default to "Medium"
    )

with control_cols[3]:
    st.checkbox("Show All Traders", value=st.session_state.show_all, key="show_all_checkbox", 
               on_change=lambda: setattr(st.session_state, "show_all", st.session_state.show_all_checkbox))

# Logs and prediction details
log_cols = st.columns(2)

with log_cols[0]:
    st.subheader(f"Prediction Log: {next((t['name'] for t in st.session_state.traders if t['id'] == st.session_state.active_trader), '')}")
    
    active_trader = next((t for t in st.session_state.traders if t["id"] == st.session_state.active_trader), None)
    
    if active_trader and active_trader["predictions"]:
        # Create a DataFrame for the predictions
        pred_data = []
        for pred in active_trader["predictions"]:
            if "prediction" in pred and pred["prediction"]:
                # Find actual values if available (5 days after prediction)
                actual_data = None
                if pred["day"] + 5 < st.session_state.day and pred["day"] + 5 < len(st.session_state.data):
                    actual_data = st.session_state.data.iloc[pred["day"] + 1:pred["day"] + 6]
                
                actual_hod = actual_data['high'].max() if actual_data is not None else None
                actual_lod = actual_data['low'].min() if actual_data is not None else None
                
                # Determine if prediction was successful
                evaluated = actual_data is not None
                success = False
                if evaluated:
                    hod_success = abs(actual_hod - pred["prediction"]["hodPrediction"]) / actual_hod <= 0.01
                    lod_success = abs(actual_lod - pred["prediction"]["lodPrediction"]) / actual_lod <= 0.01
                    success = hod_success and lod_success
                
                pred_data.append({
                    "Day": pred["day"],
                    "HOD Pred": f"{pred['prediction']['hodPrediction']:.2f}",
                    "LOD Pred": f"{pred['prediction']['lodPrediction']:.2f}",
                    "Actual HOD": f"{actual_hod:.2f}" if actual_hod is not None else "-",
                    "Actual LOD": f"{actual_lod:.2f}" if actual_lod is not None else "-",
                    "Result": "✅" if success else "❌" if evaluated else "-"
                })
        
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, height=200)
        else:
            st.info("No predictions available for this trader yet")
    else:
        st.info("No predictions yet. Start the simulation to see prediction activity.")

with log_cols[1]:
    st.subheader("System Messages")
    
    # Display system messages
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.system_messages:
        message_class = f"message-{msg['type']}"
        st.markdown(f"""
        <div class="{message_class}">
            <div style="display: flex; justify-content: space-between;">
                <span>{msg['text']}</span>
                <span style="font-size: 0.8rem; opacity: 0.7;">{msg['time']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.system_messages:
        st.markdown('<div style="color: #64748b; padding: 8px;">No system messages yet.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with explanations
st.markdown("""
### How It Works

- **Prediction Approach:** Each AI predicts High of Day (HOD) and Low of Day (LOD) for the coming market days.
- **Learning Method:** Reinforcement learning is used to reward accurate predictions and improve future forecasts.
- **AI Strategies:**
  - **Trend Follower:** Specializes in trending markets, expecting continuation of momentum.
  - **Mean Reversal:** Looks for price reversals when markets reach extremes.
  - **Volatility Breakout:** Focuses on price breakouts during volatile market conditions.
- **Usage:** After training, select the best performing AI to use for real market predictions.
""")

# Run simulation (update on each rerun)
if st.session_state.running and st.session_state.data is not None:
    if st.session_state.day >= len(st.session_state.data) - 5:  # Need at least 5 days for prediction eval
        if not st.session_state.round_ended:
            st.session_state.round_ended = True
            # Determine the winner
            winner = max(st.session_state.traders, key=lambda t: t["accuracy"]["hod"] + t["accuracy"]["lod"])
            
            # Log end of round
            add_system_message(f"{st.session_state.market_label} prediction training complete. Best performer: {winner['name']}", "success")
            
            # Handle auto-training progression
            if st.session_state.auto_training:
                # A round has ended, evaluate and move to next regime or cycle
                completed_regime = st.session_state.current_regime
                
                if completed_regime not in st.session_state.regimes_completed:
                    st.session_state.regimes_completed.append(completed_regime)
                
                training_regimes = ['uptrend', 'downtrend', 'range', 'choppy', 'mixed']
                
                if len(st.session_state.regimes_completed) >= len(training_regimes):
                    # All regimes completed, start a new cycle
                    st.session_state.training_cycles += 1
                    st.session_state.regimes_completed = []
                    
                    add_system_message(f"Training cycle {st.session_state.training_cycles} of 5 completed", "success")
                    
                    if st.session_state.training_cycles >= 5:
                        # Training complete
                        st.session_state.auto_training = False
                        add_system_message("Auto-training complete! All AI traders have been trained on market conditions.", "success")
                        st.experimental_rerun()
                        st.stop()
                
                # Pick next regime that hasn't been completed in this cycle
                next_regime = next((r for r in training_regimes if r not in st.session_state.regimes_completed), None)
                
                if next_regime:
                    # Start next round
                    add_system_message(f"Starting training on {next_regime} market conditions...", "info")
                    generate_data(next_regime)
                    reset_simulation()
                    st.session_state.running = True
                    st.experimental_rerun()
                    st.stop()
        
        st.experimental_rerun()
        st.stop()
    
    # Process current day
    window_data = st.session_state.data.iloc[:st.session_state.day + 1]
    current_candle = st.session_state.data.iloc[st.session_state.day]
    future_data = st.session_state.data.iloc[st.session_state.day + 1:st.session_state.day + 6]  # Next 5 days for evaluation
    st.session_state.current_price = current_candle['close']
    
    # Update each trader
    updated_traders = []
    for trader in st.session_state.traders:
        # Make predictions for next 5 days
        prediction = make_predictions(trader["id"], window_data, 5)
        
        # Create a copy to update
        updated_trader = trader.copy()
        
        # Store prediction
        updated_predictions = trader["predictions"].copy() if "predictions" in trader else []
        updated_predictions.append({
            "day": st.session_state.day,
            "prediction": prediction
        })
        updated_trader["predictions"] = updated_predictions
        
        # Evaluate previous predictions (from 5 days ago)
        if st.session_state.day >= 5:
            prev_pred_index = -1
            for i, p in enumerate(updated_trader["predictions"]):
                if p["day"] == st.session_state.day - 5:
                    prev_pred_index = i
                    break
            
            if prev_pred_index >= 0:
                prev_prediction = updated_trader["predictions"][prev_pred_index]["prediction"]
                actual_data_for_eval = st.session_state.data.iloc[updated_trader["predictions"][prev_pred_index]["day"] + 1:
                                                                 updated_trader["predictions"][prev_pred_index]["day"] + 6]
                
                # Evaluate prediction
                evaluation = evaluate_prediction(trader["id"], prev_prediction, actual_data_for_eval)
                
                if evaluation:
                    # Update trader stats
                    updated_trader = update_trader_stats(updated_trader, evaluation)
        
        updated_traders.append(updated_trader)
    
    # Update state
    st.session_state.traders = updated_traders
    st.session_state.day += 1
    
    # Add a delay based on simulation speed
    time.sleep(speed / 1000)
    
    # Rerun to update the UI
    st.experimental_rerun()

# JavaScript for interactivity (workaround for some event handling)
st.markdown("""
<script>
function selectTrader(traderId) {
    // Using Streamlit's postMessage API
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        value: traderId,
        dataType: "json"
    }, "*");
}

function selectAI(traderId) {
    // Using Streamlit's postMessage API
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        value: traderId,
        dataType: "json"
    }, "*");
    event.stopPropagation();
}
</script>
""", unsafe_allow_html=True)
