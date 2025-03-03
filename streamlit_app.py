import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import random
import math

# Function to add to your Streamlit app
def render_enhanced_backtesting():
    st.title("Advanced HOD/LOD Market Prediction Backtesting")
    
    # Initialize backtesting session state
    if 'bt_data' not in st.session_state:
        st.session_state.bt_data = None
    if 'bt_running' not in st.session_state:
        st.session_state.bt_running = False
    if 'bt_day' not in st.session_state:
        st.session_state.bt_day = 0
    if 'bt_traders' not in st.session_state:
        st.session_state.bt_traders = [
            {"id": 1, "name": "Trend Follower", "balance": 10000, "color": "#4338ca", "wins": 0, 
             "strategy": "trend", "learningRate": 0.05, "accuracy": {"hod": 0.0, "lod": 0.0}, 
             "predictions": [], "totalValue": 10000},
            {"id": 2, "name": "Mean Reversal", "balance": 10000, "color": "#059669", "wins": 0, 
             "strategy": "reversal", "learningRate": 0.08, "accuracy": {"hod": 0.0, "lod": 0.0}, 
             "predictions": [], "totalValue": 10000},
            {"id": 3, "name": "Volatility Breakout", "balance": 10000, "color": "#d97706", "wins": 0, 
             "strategy": "volatility", "learningRate": 0.07, "accuracy": {"hod": 0.0, "lod": 0.0}, 
             "predictions": [], "totalValue": 10000}
        ]
    if 'bt_trader_networks' not in st.session_state:
        st.session_state.bt_trader_networks = {
            1: {
                "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "biasHod": float(np.random.uniform(-0.1, 0.1)),
                "biasLod": float(np.random.uniform(-0.1, 0.1))
            },
            2: {
                "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "biasHod": float(np.random.uniform(-0.1, 0.1)),
                "biasLod": float(np.random.uniform(-0.1, 0.1))
            },
            3: {
                "hodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "lodWeights": np.random.uniform(-0.1, 0.1, 10).tolist(),
                "biasHod": float(np.random.uniform(-0.1, 0.1)),
                "biasLod": float(np.random.uniform(-0.1, 0.1))
            }
        }
    if 'bt_selected_ai' not in st.session_state:
        st.session_state.bt_selected_ai = None
    if 'bt_messages' not in st.session_state:
        st.session_state.bt_messages = []
    if 'bt_active_trader' not in st.session_state:
        st.session_state.bt_active_trader = 1
    if 'bt_market_type' not in st.session_state:
        st.session_state.bt_market_type = 'all'
    
    # Function to add system message
    def add_bt_message(message, message_type="info"):
        new_message = {
            "id": int(time.time() * 1000),
            "text": message,
            "type": message_type,
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.bt_messages.insert(0, new_message)
        # Limit to the most recent 50 messages
        st.session_state.bt_messages = st.session_state.bt_messages[:50]
    
    # Function to generate backtesting data
    def generate_bt_data(market_type='all'):
        # Create market cycles and patterns with 1000 candles
        data_length = 1000
        price = 100
        dates = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(data_length)]
        
        # Pre-defined patterns
        def uptrend_candle(i, base_price):
            volatility = 1.5 + math.sin(i/30) * 0.5
            trend_strength = 0.15
            change = (random.random() - 0.35) * volatility + trend_strength
            open_price = base_price
            close_price = base_price + change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.5
            low_price = min(open_price, close_price) - random.random() * volatility * 0.3
            return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
        
        def downtrend_candle(i, base_price):
            volatility = 1.5 + math.cos(i/20) * 0.5
            trend_strength = -0.18
            change = (random.random() - 0.65) * volatility + trend_strength
            open_price = base_price
            close_price = base_price + change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.3
            low_price = min(open_price, close_price) - random.random() * volatility * 0.5
            return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
        
        def range_candle(i, base_price):
            center_price = 100
            reversion = (base_price - center_price) * 0.1
            volatility = 1.0
            change = (random.random() - 0.5) * volatility - reversion
            open_price = base_price
            close_price = base_price + change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.4
            low_price = min(open_price, close_price) - random.random() * volatility * 0.4
            return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
        
        def choppy_candle(i, base_price):
            cycle = math.sin(i/5) * math.cos(i/3)
            volatility = 1.2
            change = cycle * volatility + (random.random() - 0.5) * volatility * 0.5
            open_price = base_price
            close_price = base_price + change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.3
            low_price = min(open_price, close_price) - random.random() * volatility * 0.3
            return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
        
        def default_candle(i, base_price):
            volatility = 1.0
            change = (random.random() - 0.5) * volatility
            open_price = base_price
            close_price = base_price + change
            high_price = max(open_price, close_price) + random.random() * volatility * 0.3
            low_price = min(open_price, close_price) - random.random() * volatility * 0.3
            return {"change": change, "open": open_price, "close": close_price, "high": high_price, "low": low_price}
        
        # Select pattern function based on market type
        pattern_func = default_candle
        if market_type == 'uptrend':
            pattern_func = uptrend_candle
        elif market_type == 'downtrend':
            pattern_func = downtrend_candle
        elif market_type == 'range':
            pattern_func = range_candle
        elif market_type == 'choppy':
            pattern_func = choppy_candle
        
        # Generate data
        data = []
        
        if market_type == 'all' or market_type == 'mixed':
            # Create segments of different patterns
            segments = []
            current_pos = 0
            while current_pos < data_length:
                segment_len = random.randint(50, 200)
                pattern = random.choice(['uptrend', 'downtrend', 'range', 'choppy'])
                segments.append({"start": current_pos, "length": segment_len, "pattern": pattern})
                current_pos += segment_len
                
            # Generate candles based on segments
            for i in range(data_length):
                # Find active segment
                active_segment = next((s for s in segments if i >= s["start"] and i < s["start"] + s["length"]), None)
                
                # Use appropriate pattern
                if active_segment:
                    if active_segment["pattern"] == 'uptrend':
                        candle = uptrend_candle(i, price)
                    elif active_segment["pattern"] == 'downtrend':
                        candle = downtrend_candle(i, price)
                    elif active_segment["pattern"] == 'range':
                        candle = range_candle(i, price)
                    elif active_segment["pattern"] == 'choppy':
                        candle = choppy_candle(i, price)
                    else:
                        candle = default_candle(i, price)
                else:
                    candle = default_candle(i, price)
                
                # Update price
                price += candle["change"]
                price = max(price, 10)  # Ensure price doesn't go below 10
                
                # Create volume
                volume = int(1000 + abs(candle["change"]) * 1000 + random.random() * 500)
                
                # Add to data
                data.append({
                    "date": dates[i],
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": price,
                    "volume": volume
                })
        else:
            # Single pattern throughout
            for i in range(data_length):
                candle = pattern_func(i, price)
                price += candle["change"]
                price = max(price, 10)
                volume = int(1000 + abs(candle["change"]) * 1000 + random.random() * 500)
                
                data.append({
                    "date": dates[i],
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": price,
                    "volume": volume
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    # Function to extract features for prediction
    def extract_features(data):
        if len(data) < 20:
            return None
        
        # Get most recent candles
        recent_data = data.tail(20)
        
        # Calculate technical indicators
        closes = recent_data['close'].values
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        volumes = recent_data['volume'].values
        
        # Simple moving averages
        sma5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        sma10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
        sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        
        # Price momentum
        momentum5 = closes[-1] / closes[-6] if len(closes) >= 6 else 1.0
        momentum10 = closes[-1] / closes[-11] if len(closes) >= 11 else 1.0
        
        # Volatility (approximation of ATR)
        ranges = [max(highs[i] - lows[i], 
                      abs(highs[i] - closes[i-1]), 
                      abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
        volatility = np.mean(ranges) if ranges else 0
        
        # Volume trend
        volume_sma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        volume_change = volumes[-1] / volume_sma5 if volume_sma5 > 0 else 1.0
        
        # Price position
        recent_highest = np.max(highs)
        recent_lowest = np.min(lows)
        price_position = (closes[-1] - recent_lowest) / (recent_highest - recent_lowest) if (recent_highest - recent_lowest) > 0 else 0.5
        
        # Days since extremes
        highest_idx = np.argmax(highs)
        lowest_idx = np.argmin(lows) 
        days_since_high = len(highs) - 1 - highest_idx
        days_since_low = len(lows) - 1 - lowest_idx
        
        # Create feature vector
        features = [
            closes[-1] / sma5,
            closes[-1] / sma10,
            closes[-1] / sma20,
            momentum5,
            momentum10,
            volatility / closes[-1] if closes[-1] > 0 else 0,
            volume_change,
            price_position,
            1.0 / (days_since_high + 1),
            1.0 / (days_since_low + 1)
        ]
        
        return features
    
    # Function for neural network prediction
    def predict_with_network(features, weights, bias):
        if not features:
            return 0
        
        result = bias
        for i in range(min(len(features), len(weights))):
            result += features[i] * weights[i]
        
        return result
    
    # Make HOD/LOD predictions
    def make_predictions(trader_id, data):
        if len(data) < 20:
            return None
        
        features = extract_features(data)
        if not features:
            return None
        
        network = st.session_state.bt_trader_networks[trader_id]
        trader = next((t for t in st.session_state.bt_traders if t["id"] == trader_id), None)
        
        # Get current close price
        current_close = data['close'].iloc[-1]
        
        # Strategy multipliers
        hod_multiplier = 1.0
        lod_multiplier = 1.0
        
        if trader["strategy"] == 'trend':
            # Check if recent trend is up or down
            closes = data['close'].tail(10).values
            if closes[-1] > closes[0]:  # Uptrend
                hod_multiplier = 1.2
                lod_multiplier = 0.9
            else:  # Downtrend
                hod_multiplier = 0.9
                lod_multiplier = 1.2
                
        elif trader["strategy"] == 'reversal':
            # Check if price is near recent extremes
            highs = data['high'].tail(10).values
            lows = data['low'].tail(10).values
            if current_close > np.max(highs) * 0.95:  # Near highs
                hod_multiplier = 0.9
                lod_multiplier = 1.1
            elif current_close < np.min(lows) * 1.05:  # Near lows
                hod_multiplier = 1.1
                lod_multiplier = 0.9
                
        elif trader["strategy"] == 'volatility':
            # Check recent volatility
            highs = data['high'].tail(10).values
            lows = data['low'].tail(10).values
            volatility = (np.max(highs) - np.min(lows)) / current_close
            if volatility > 0.02:  # Higher volatility
                hod_multiplier = 1.1
                lod_multiplier = 1.1
        
        # Make predictions using neural network
        raw_hod = predict_with_network(features, network["hodWeights"], network["biasHod"])
        raw_lod = predict_with_network(features, network["lodWeights"], network["biasLod"])
        
        # Convert to price movements
        hod_pct_change = 0.02 * raw_hod * hod_multiplier
        lod_pct_change = -0.02 * raw_lod * lod_multiplier
        
        # Calculate predicted prices
        hod_prediction = current_close * (1 + hod_pct_change)
        lod_prediction = current_close * (1 + lod_pct_change)
        
        # Ensure HOD > LOD
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
        error_pct = abs(error / actual) if actual != 0 else 0
        
        # Only reward/penalize if significant error
        if error_pct < 0.0025:
            return True  # Correct enough
        
        # Get network and trader
        network = st.session_state.bt_trader_networks[trader_id].copy()
        trader = next((t for t in st.session_state.bt_traders if t["id"] == trader_id), None)
        
        if not trader:
            return False
            
        # Get weights and bias to update
        weights = network["hodWeights"] if is_hod else network["lodWeights"]
        bias = network["biasHod"] if is_hod else network["biasLod"]
        
        # Learning rate adjustment based on error
        learning_rate = trader["learningRate"] * (1.2 if error_pct > 0.01 else 0.8)
        
        # Apply reinforcement learning
        if error_pct > 0.005:
            # Update weights
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
            
            # Save to session state
            if is_hod:
                network["hodWeights"] = updated_weights
                network["biasHod"] = updated_bias
            else:
                network["lodWeights"] = updated_weights
                network["biasLod"] = updated_bias
                
            st.session_state.bt_trader_networks[trader_id] = network
        
        return error_pct <= 0.01  # Success if error is less than 1%
        
    # Evaluate prediction accuracy
    def evaluate_prediction(trader_id, prediction, actual_data):
        if not prediction or actual_data.empty:
            return None
            
        # Get actual HOD/LOD
        actual_hod = actual_data['high'].max()
        actual_lod = actual_data['low'].min()
        
        # Calculate errors
        hod_error = abs((prediction["hodPrediction"] - actual_hod) / actual_hod) if actual_hod != 0 else 1.0
        lod_error = abs((prediction["lodPrediction"] - actual_lod) / actual_lod) if actual_lod != 0 else 1.0
        
        # Update weights and get success status
        hod_success = update_network_weights(trader_id, prediction["features"], actual_hod, prediction["hodPrediction"], True)
        lod_success = update_network_weights(trader_id, prediction["features"], actual_lod, prediction["lodPrediction"], False)
        
        # Calculate accuracy
        hod_accuracy = max(0, 100 - (hod_error * 100))
        lod_accuracy = max(0, 100 - (lod_error * 100))
        
        return {
            "hodSuccess": hod_success,
            "lodSuccess": lod_success,
            "hodAccuracy": hod_accuracy,
            "lodAccuracy": lod_accuracy,
            "actualHod": actual_hod,
            "actualLod": actual_lod
        }
        
    # Update trader stats
    def update_trader_stats(trader, evaluation):
        if not evaluation:
            return trader
            
        # Get success metrics
        hod_success = evaluation["hodSuccess"]
        lod_success = evaluation["lodSuccess"]
        hod_accuracy = evaluation["hodAccuracy"]
        lod_accuracy = evaluation["lodAccuracy"]
        
        # Overall success
        overall_success = (hod_success and lod_success) or (hod_success and lod_accuracy > 95) or (lod_success and hod_accuracy > 95)
        
        # Update accuracy using exponential moving average
        alpha = 0.05  # Weight for new data
        trader_copy = trader.copy()
        trader_copy["accuracy"]["hod"] = alpha * hod_accuracy + (1 - alpha) * trader["accuracy"]["hod"]
        trader_copy["accuracy"]["lod"] = alpha * lod_accuracy + (1 - alpha) * trader["accuracy"]["lod"]
        trader_copy["wins"] = trader["wins"] + (1 if overall_success else 0)
        
        return trader_copy
        
    # Reset simulation
    def reset_simulation():
        st.session_state.bt_day = 0
        st.session_state.bt_running = False
        
        # Reset traders
        updated_traders = []
        for trader in st.session_state.bt_traders:
            updated_trader = trader.copy()
            updated_trader["predictions"] = []
            updated_trader["wins"] = 0
            updated_trader["accuracy"] = {"hod": 0.0, "lod": 0.0}
            updated_traders.append(updated_trader)
            
        st.session_state.bt_traders = updated_traders
    
    # Main UI elements
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        market_options = {
            'all': 'All Market Types',
            'uptrend': 'Strong Uptrend',
            'downtrend': 'Strong Downtrend',
            'range': 'Range-Bound',
            'choppy': 'Choppy Market'
        }
        
        selected_market = st.selectbox(
            "Market Type",
            options=list(market_options.keys()),
            format_func=lambda x: market_options[x],
            key="bt_market_select"
        )
        
        if selected_market != st.session_state.bt_market_type:
            st.session_state.bt_market_type = selected_market
            st.session_state.bt_data = generate_bt_data(selected_market)
            reset_simulation()
            add_bt_message(f"Generated new {market_options[selected_market]} data", "info")
    
    with col2:
        st.metric("Day", f"{st.session_state.bt_day}/{1000 if st.session_state.bt_data is not None else 0}")
        
    with col3:
        if st.button("Generate New Data"):
            st.session_state.bt_data = generate_bt_data(st.session_state.bt_market_type)
            reset_simulation()
            add_bt_message(f"Generated new market data", "info")
    
    # Trader cards
    st.subheader("AI Prediction Engines")
    trader_cols = st.columns(3)
    
    for i, trader in enumerate(st.session_state.bt_traders):
        with trader_cols[i]:
            is_active = trader["id"] == st.session_state.bt_active_trader
            is_selected = trader["id"] == st.session_state.bt_selected_ai
            
            # Card background
            card_bg = "rgb(30, 41, 59)" if is_active else "rgb(29, 78, 216)" if is_selected else "white"
            card_text = "white" if is_active or is_selected else "black"
            
            # Create a card with CSS
            st.markdown(f"""
            <div style="background-color: {card_bg}; color: {card_text}; padding: 15px; border-radius: 8px; 
                        margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s; 
                        transform: scale({1.02 if is_active else 1.0});">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {trader['color']}; margin-right: 8px;"></div>
                        <div style="font-weight: bold;">{trader['name']}</div>
                    </div>
                    <div style="background-color: {'rgba(255,255,255,0.2)' if is_active or is_selected else 'rgba(0,0,0,0.1)'}; 
                              padding: 3px 8px; border-radius: 12px; font-size: 0.7rem;">
                        Wins: {trader['wins']}
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
                    <div>
                        <div style="font-size: 0.7rem; color: {'rgba(255,255,255,0.7)' if is_active or is_selected else 'rgba(0,0,0,0.5)'};">
                            HOD Accuracy
                        </div>
                        <div style="font-weight: bold;">{trader['accuracy']['hod']:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: {'rgba(255,255,255,0.7)' if is_active or is_selected else 'rgba(0,0,0,0.5)'};">
                            LOD Accuracy
                        </div>
                        <div style="font-weight: bold;">{trader['accuracy']['lod']:.2f}%</div>
                    </div>
                </div>
                <div style="font-size: 0.7rem; margin-top: 5px; opacity: 0.7;">
                    Strategy: {
                        'Trend Following' if trader['strategy'] == 'trend' else
                        'Mean Reversal' if trader['strategy'] == 'reversal' else
                        'Volatility Breakout'
                    }
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Buttons for selection
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Details", key=f"details_{trader['id']}"):
                    st.session_state.bt_active_trader = trader["id"]
                    st.experimental_rerun()
            with col2:
                if not is_selected:
                    if st.button(f"Select", key=f"select_{trader['id']}"):
                        st.session_state.bt_selected_ai = trader["id"]
                        add_bt_message(f"Selected {trader['name']} as primary prediction AI", "success")
                        st.experimental_rerun()
    
    # Chart with predictions
    if st.session_state.bt_data is not None:
        st.subheader("Market Chart with HOD/LOD Predictions")
        
        # Get data for display
        visible_window = 70
        start_idx = max(0, st.session_state.bt_day - visible_window)
        end_idx = st.session_state.bt_day + 1
        display_data = st.session_state.bt_data.iloc[start_idx:end_idx].copy()
        
        # Get active traders for prediction display
        active_traders = []
        for trader in st.session_state.bt_traders:
            if trader["id"] == st.session_state.bt_active_trader or trader["id"] == st.session_state.bt_selected_ai:
                # Find most recent prediction
                latest_prediction = None
                for pred in reversed(trader.get("predictions", [])):
                    if pred["day"] == st.session_state.bt_day - 1:
                        latest_prediction = pred
                        break
                
                if latest_prediction and "prediction" in latest_prediction:
                    active_traders.append({
                        "traderId": trader["id"],
                        "name": trader["name"],
                        "color": trader["color"],
                        "prediction": latest_prediction["prediction"]
                    })
        
        # Create candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.8, 0.2], 
                           vertical_spacing=0.02)
        
        # Add candlesticks
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
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=display_data.index,
                y=display_data['volume'],
                name="Volume",
                marker=dict(color='rgba(100, 100, 100, 0.5)')
            ),
            row=2, col=1
        )
        
        # Add prediction lines
        for trader in active_traders:
            if "hodPrediction" in trader["prediction"]:
                # HOD prediction line
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
                
                # LOD prediction line
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
                
                # Annotations for prediction values
                fig.add_annotation(
                    x=display_data.index[-1] + 5,
                    y=trader["prediction"]["hodPrediction"],
                    text=f"HOD: {trader['prediction']['hodPrediction']:.2f}",
                    showarrow=False,
                    font=dict(color=trader["color"]),
                    xanchor="left",
                    row=1, col=1
                )
                
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Fix y-axis range
        if not display_data.empty:
            y_min = display_data['low'].min() * 0.98
            y_max = display_data['high'].max() * 1.02
            
            # Include prediction lines in range
            for trader in active_traders:
                if "prediction" in trader and "hodPrediction" in trader["prediction"]:
                    y_max = max(y_max, trader["prediction"]["hodPrediction"] * 1.02)
                    y_min = min(y_min, trader["prediction"]["lodPrediction"] * 0.98)
            
            fig.update_layout(yaxis_range=[y_min, y_max])
            
        st.plotly_chart(fig, use_container_width=True)
    
    # Control panel
    st.subheader("Simulation Controls")
    control_cols = st.columns([1, 1, 1, 1])
    
    with control_cols[0]:
        if st.button("▶️ Start" if not st.session_state.bt_running else "⏸️ Pause"):
            st.session_state.bt_running = not st.session_state.bt_running
            
    with control_cols[1]:
        if st.button("🔄 Reset"):
            reset_simulation()
            
    with control_cols[2]:
        speed_options = {
            1000: "Slow",
            500: "Medium",
            200: "Fast",
            50: "Ultra Fast",
            10: "Maximum"
        }
        
        speed = st.selectbox(
            "Speed",
            options=list(speed_options.keys()),
            format_func=lambda x: speed_options[x],
            index=1,  # Default to Medium
            key="bt_speed_select"
        )
    
    with control_cols[3]:
        show_all = st.checkbox("Show All Traders", key="bt_show_all")
    
    # Logs and predictions
    log_cols = st.columns(2)
    
    with log_cols[0]:
        active_trader = next((t for t in st.session_state.bt_traders if t["id"] == st.session_state.bt_active_trader), None)
        st.subheader(f"Prediction Log: {active_trader['name'] if active_trader else ''}")
        
        if active_trader and "predictions" in active_trader and active_trader["predictions"]:
            # Create DataFrame for display
            pred_data = []
            for pred in active_trader["predictions"]:
                if "prediction" in pred:
                    # Find actual values if available
                    actual_data = None
                    if pred["day"] + 5 < st.session_state.bt_day and pred["day"] + 5 < len(st.session_state.bt_data):
                        actual_data = st.session_state.bt_data.iloc[pred["day"] + 1:pred["day"] + 6]
                    
                    actual_hod = actual_data['high'].max() if actual_data is not None and not actual_data.empty else None
                    actual_lod = actual_data['low'].min() if actual_data is not None and not actual_data.empty else None
                    
                    # Success evaluation
                    success = "-"
                    if actual_hod is not None and actual_lod is not None:
                        hod_error = abs(actual_hod - pred["prediction"]["hodPrediction"]) / actual_hod if actual_hod != 0 else 1.0
                        lod_error = abs(actual_lod - pred["prediction"]["lodPrediction"]) / actual_lod if actual_lod != 0 else 1.0
                        if hod_error <= 0.01 and lod_error <= 0.01:
                            success = "✅"
                        else:
                            success = "❌"
                    
                    pred_data.append({
                        "Day": pred["day"],
                        "HOD Pred": f"{pred['prediction']['hodPrediction']:.2f}",
                        "LOD Pred": f"{pred['prediction']['lodPrediction']:.2f}",
                        "Actual HOD": f"{actual_hod:.2f}" if actual_hod is not None else "-",
                        "Actual LOD": f"{actual_lod:.2f}" if actual_lod is not None else "-",
                        "Result": success
                    })
            
            if pred_data:
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, height=200)
            else:
                st.info("No predictions available for this trader yet.")
        else:
            st.info("No predictions yet. Start the simulation to see prediction activity.")
    
    with log_cols[1]:
        st.subheader("System Messages")
        
        # Display system messages
        message_container = st.container()
        with message_container:
            for msg in st.session_state.bt_messages:
                if msg["type"] == "success":
                    st.success(f"{msg['text']} - {msg['time']}")
                elif msg["type"] == "warning":
                    st.warning(f"{msg['text']} - {msg['time']}")
                else:
                    st.info(f"{msg['text']} - {msg['time']}")
    
    # Info about backtesting
    st.markdown("""
    ### HOD/LOD Backtesting Approach
    
    This system uses machine learning to predict the **High of Day (HOD)** and **Low of Day (LOD)** for market prices.
    
    Each AI engine uses a different strategy:
    - **Trend Follower**: Excels at identifying directional momentum
    - **Mean Reversal**: Specializes in detecting price reversals
    - **Volatility Breakout**: Focuses on price moves during high volatility
    
    The system uses reinforcement learning to improve predictions over time. More accurate predictions lead to
    higher accuracy scores and better overall performance.
    
    After training, select the best performing AI for your own market predictions.
    """)
    
    # Run simulation if active
    if st.session_state.bt_running and st.session_state.bt_data is not None:
        # Check if we're at the end
        if st.session_state.bt_day >= len(st.session_state.bt_data) - 5:
            st.session_state.bt_running = False
            # Determine winner
            winner = max(st.session_state.bt_traders, key=lambda t: t["accuracy"]["hod"] + t["accuracy"]["lod"])
            add_bt_message(f"Backtesting complete! Best performer: {winner['name']}", "success")
            st.experimental_rerun()
        else:
            # Process current day
            current_day = st.session_state.bt_day
            window_data = st.session_state.bt_data.iloc[:current_day + 1]
            future_data = st.session_state.bt_data.iloc[current_day + 1:current_day + 6]
            
            # Update traders
            updated_traders = []
            for trader in st.session_state.bt_traders:
                # Make new prediction
                prediction = make_predictions(trader["id"], window_data)
                
                # Store prediction
                updated_trader = trader.copy()
                updated_predictions = trader.get("predictions", []).copy()
                updated_predictions.append({
                    "day": current_day,
                    "prediction": prediction
                })
                updated_trader["predictions"] = updated_predictions
                
                # Evaluate earlier predictions (from 5 days ago)
                if current_day >= 5:
                    # Find prediction from 5 days ago
                    prev_pred = None
                    for p in updated_trader["predictions"]:
                        if p["day"] == current_day - 5:
                            prev_pred = p
                            break
                    
                    if prev_pred and "prediction" in prev_pred:
                        # Get actual data for that period
                        eval_data = st.session_state.bt_data.iloc[prev_pred["day"] + 1:prev_pred["day"] + 6]
                        
                        # Evaluate
                        evaluation = evaluate_prediction(trader["id"], prev_pred["prediction"], eval_data)
                        
                        if evaluation:
                            # Update stats
                            updated_trader = update_trader_stats(updated_trader, evaluation)
                
                updated_traders.append(updated_trader)
            
            # Update session state
            st.session_state.bt_traders = updated_traders
            st.session_state.bt_day += 1
            
            # Add a message occasionally
            if current_day % 100 == 0:
                add_bt_message(f"Processing day {current_day}...", "info")
                
            # Rerun to update UI
            time.sleep(speed / 1000)  # Delay based on speed setting
            st.experimental_rerun()
