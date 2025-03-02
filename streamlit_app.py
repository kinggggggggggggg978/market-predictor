import React, { useState, useEffect, useRef } from 'react';

const EnhancedBacktesting = () => {
  const [data, setData] = useState([]);
  const [traders, setTraders] = useState([
    { id: 1, name: "Trend Follower", balance: 10000, holdings: 0, trades: [], color: "#4338ca", learningRate: 0.05, adaptiveRate: 0.02, strategy: "trend", totalValue: 10000, wins: 0, performanceHistory: [], lastEvaluation: 0, predictions: [], accuracy: { hod: 0, lod: 0 }, maxDrawdown: 0, consecutiveWins: 0, consecutiveLosses: 0 },
    { id: 2, name: "Mean Reversal", balance: 10000, holdings: 0, trades: [], color: "#059669", learningRate: 0.08, adaptiveRate: 0.02, strategy: "reversal", totalValue: 10000, wins: 0, performanceHistory: [], lastEvaluation: 0, predictions: [], accuracy: { hod: 0, lod: 0 }, maxDrawdown: 0, consecutiveWins: 0, consecutiveLosses: 0 },
    { id: 3, name: "Volatility Breakout", balance: 10000, holdings: 0, trades: [], color: "#d97706", learningRate: 0.07, adaptiveRate: 0.02, strategy: "volatility", totalValue: 10000, wins: 0, performanceHistory: [], lastEvaluation: 0, predictions: [], accuracy: { hod: 0, lod: 0 }, maxDrawdown: 0, consecutiveWins: 0, consecutiveLosses: 0 }
  ]);
  const [activeTrader, setActiveTrader] = useState(1);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [day, setDay] = useState(0);
  const [patternType, setPatternType] = useState('all');
  const [showAll, setShowAll] = useState(false);
  const [roundEnded, setRoundEnded] = useState(false);
  const [marketLabel, setMarketLabel] = useState("Mixed Market");
  const [currentRegimeType, setCurrentRegimeType] = useState("mixed");
  const [systemMessages, setSystemMessages] = useState([]);
  const [predictionStats, setPredictionStats] = useState({
    1: { hodAccuracy: 0, lodAccuracy: 0, totalAccuracy: 0, winRate: 0, drawdown: 0 },
    2: { hodAccuracy: 0, lodAccuracy: 0, totalAccuracy: 0, winRate: 0, drawdown: 0 },
    3: { hodAccuracy: 0, lodAccuracy: 0, totalAccuracy: 0, winRate: 0, drawdown: 0 }
  });
  const [selectedAI, setSelectedAI] = useState(null);
  const [autoTraining, setAutoTraining] = useState(false);
  const trainingCyclesRef = useRef(0);
  const maxTrainingCycles = 5;
  const regimesCompleted = useRef([]);
  const messageListRef = useRef(null);

  // Neural network weights for each trader's prediction model
  const [traderNetworks, setTraderNetworks] = useState({
    1: {
      hodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      lodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      biasHod: Math.random() * 0.2 - 0.1,
      biasLod: Math.random() * 0.2 - 0.1
    },
    2: {
      hodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      lodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      biasHod: Math.random() * 0.2 - 0.1,
      biasLod: Math.random() * 0.2 - 0.1
    },
    3: {
      hodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      lodWeights: Array(10).fill().map(() => Math.random() * 0.2 - 0.1),
      biasHod: Math.random() * 0.2 - 0.1,
      biasLod: Math.random() * 0.2 - 0.1
    }
  });

  // Market regimes for automatic training
  const trainingRegimes = ['uptrend', 'downtrend', 'range', 'choppy', 'mixed'];

  // Add system message
  const addSystemMessage = (message, type = "info") => {
    const newMessage = {
      id: Date.now(),
      text: message,
      type: type,
      time: new Date().toLocaleTimeString()
    };
    setSystemMessages(prev => [newMessage, ...prev].slice(0, 50));
  };

  // Detect market structure
  const detectMarketStructure = (windowData, lookback = 50) => {
    if (windowData.length < lookback) return "unknown";
    
    const recentData = windowData.slice(-lookback);
    const highs = recentData.map(d => d.high);
    const lows = recentData.map(d => d.low);
    const closes = recentData.map(d => d.close);
    
    // Find swing highs and lows
    const swingHighs = [];
    const swingLows = [];
    
    for (let i = 2; i < highs.length - 2; i++) {
      if (highs[i] > highs[i-1] && highs[i] > highs[i-2] && 
          highs[i] > highs[i+1] && highs[i] > highs[i+2]) {
        swingHighs.push({ index: i, value: highs[i] });
      }
      
      if (lows[i] < lows[i-1] && lows[i] < lows[i-2] && 
          lows[i] < lows[i+1] && lows[i] < lows[i+2]) {
        swingLows.push({ index: i, value: lows[i] });
      }
    }
    
    // Determine market structure based on swing highs and lows
    if (swingHighs.length >= 3 && swingLows.length >= 3) {
      const lastTwoHighs = swingHighs.slice(-2);
      const lastTwoLows = swingLows.slice(-2);
      
      if (lastTwoHighs[1].value > lastTwoHighs[0].value && 
          lastTwoLows[1].value > lastTwoLows[0].value) {
        return "uptrend"; // Higher highs and higher lows
      } else if (lastTwoHighs[1].value < lastTwoHighs[0].value && 
                lastTwoLows[1].value < lastTwoLows[0].value) {
        return "downtrend"; // Lower highs and lower lows
      } else if (Math.abs(lastTwoHighs[1].value - lastTwoHighs[0].value) / lastTwoHighs[0].value < 0.01 &&
                Math.abs(lastTwoLows[1].value - lastTwoLows[0].value) / lastTwoLows[0].value < 0.01) {
        return "range"; // Flat highs and lows
      }
    }
    
    // Check for range-bound market
    const highestHigh = Math.max(...highs);
    const lowestLow = Math.min(...lows);
    const range = (highestHigh - lowestLow) / lowestLow;
    
    if (range < 0.03) {
      return "range";
    }
    
    // Check for choppy market
    let changes = 0;
    let direction = closes[1] > closes[0] ? 1 : -1;
    
    for (let i = 2; i < closes.length; i++) {
      const newDirection = closes[i] > closes[i-1] ? 1 : -1;
      if (newDirection !== direction) {
        changes++;
        direction = newDirection;
      }
    }
    
    if (changes > lookback * 0.4) {
      return "choppy";
    }
    
    return "mixed";
  };

  // Generate realistic market data
  const generateData = (forcedPattern = null) => {
    // Create market cycles and patterns with 1000 candles
    const dataLength = 1000;
    const newData = [];
    let price = 100;
    
    // Pattern to use (from parameter or state)
    const selectedPattern = forcedPattern || patternType;
    
    // Default market label
    let marketLabelText = "Mixed Market";
    let regimeType = "mixed";
    
    if (selectedPattern === 'uptrend') {
      marketLabelText = "Strong Uptrend Market";
      regimeType = "uptrend";
    } else if (selectedPattern === 'downtrend') {
      marketLabelText = "Strong Downtrend Market";
      regimeType = "downtrend";
    } else if (selectedPattern === 'range') {
      marketLabelText = "Range-Bound Market";
      regimeType = "range";
    } else if (selectedPattern === 'choppy') {
      marketLabelText = "Choppy Market";
      regimeType = "choppy";
    } 
    
    setMarketLabel(marketLabelText);
    setCurrentRegimeType(regimeType);
    
    // Pre-defined patterns
    const patterns = {
      uptrend: (i, basePrice) => {
        const volatility = 1.5 + Math.sin(i/30) * 0.5;
        const trendStrength = 0.15;
        const change = (Math.random() - 0.35) * volatility + trendStrength;
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.5;
        const low = Math.min(open, close) - Math.random() * volatility * 0.3;
        return { change, open, close, high, low };
      },
      downtrend: (i, basePrice) => {
        const volatility = 1.5 + Math.cos(i/20) * 0.5;
        const trendStrength = -0.18;
        const change = (Math.random() - 0.65) * volatility + trendStrength;
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.3;
        const low = Math.min(open, close) - Math.random() * volatility * 0.5;
        return { change, open, close, high, low };
      },
      range: (i, basePrice) => {
        const centerPrice = 100; // Center of the range
        const rangeSize = 8;     // Size of the range (plus/minus from center)
        
        // Calculate a mean-reverting price
        const distanceFromCenter = basePrice - centerPrice;
        const reversion = distanceFromCenter * 0.1; // Strength of reversion
        
        const volatility = 1.0;
        const change = (Math.random() - 0.5) * volatility - reversion;
        
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.4;
        const low = Math.min(open, close) - Math.random() * volatility * 0.4;
        return { change, open, close, high, low };
      },
      choppy: (i, basePrice) => {
        // Create zigzag pattern
        const cycle = Math.sin(i/5) * Math.cos(i/3); // Creates frequent reversals
        const volatility = 1.2;
        const change = cycle * volatility + (Math.random() - 0.5) * volatility * 0.5;
        
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.3;
        const low = Math.min(open, close) - Math.random() * volatility * 0.3;
        return { change, open, close, high, low };
      },
      // Default pattern as fallback
      default: (i, basePrice) => {
        const volatility = 1.0;
        const change = (Math.random() - 0.5) * volatility;
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.3;
        const low = Math.min(open, close) - Math.random() * volatility * 0.3;
        return { change, open, close, high, low };
      }
    };
    
    // Market structure with multiple segments
    const segments = [];
    let currentPosition = 0;
    
    if (selectedPattern === 'all' || selectedPattern === 'mixed') {
      // Create a complex, realistic market with multiple segments
      while (currentPosition < dataLength) {
        const segmentLength = Math.floor(Math.random() * 150) + 50; // 50-200 candles per segment
        const patternKeys = ['uptrend', 'downtrend', 'range', 'choppy'];
        const pattern = patternKeys[Math.floor(Math.random() * patternKeys.length)];
        segments.push({ start: currentPosition, length: segmentLength, pattern });
        currentPosition += segmentLength;
      }
    } else {
      // Create a single pattern type market
      segments.push({ start: 0, length: dataLength, pattern: selectedPattern });
    }
    
    // Generate the actual data
    for (let i = 0; i < dataLength; i++) {
      // Find which segment contains this position
      const activeSegment = segments.find(seg => i >= seg.start && i < seg.start + seg.length);
      
      let candle;
      if (activeSegment) {
        // Use regular segment pattern
        candle = patterns[activeSegment.pattern](i, price);
      } else {
        // Fallback to default pattern if no segment covers this position
        candle = patterns.default(i, price);
      }
      
      // Apply the price change
      price += candle.change;
      
      // Ensure price doesn't go below 10
      price = Math.max(price, 10);
      
      // Generate volume based on price movement
      const volumeBase = 1000;
      const volumeVariation = Math.abs(candle.change) * 1000;
      const volume = Math.floor(volumeBase + volumeVariation + Math.random() * 500);
      
      // Create the candle data
      const newCandle = {
        time: i,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: price,
        volume
      };
      
      newData.push(newCandle);
    }
    
    // Update state with the generated data
    setData(newData);
    setCurrentPrice(newData[0].close);
  };

  // Initialize data on component mount and pattern type change
  useEffect(() => {
    if (!autoTraining) {
      generateData();
      resetSimulation();
    }
  }, [patternType]);
  
  // Auto-training logic
  useEffect(() => {
    if (!autoTraining) return;
    
    if (roundEnded) {
      // A round has ended, evaluate and move to next regime or cycle
      const completedRegime = currentRegimeType;
      
      if (!regimesCompleted.current.includes(completedRegime)) {
        regimesCompleted.current.push(completedRegime);
      }
      
      if (regimesCompleted.current.length >= trainingRegimes.length) {
        // All regimes completed, start a new cycle
        trainingCyclesRef.current += 1;
        regimesCompleted.current = [];
        
        addSystemMessage(`Training cycle ${trainingCyclesRef.current} of ${maxTrainingCycles} completed`, "success");
        
        if (trainingCyclesRef.current >= maxTrainingCycles) {
          // Training complete
          setAutoTraining(false);
          addSystemMessage("Auto-training complete! All AI traders have been trained on market conditions.", "success");
          return;
        }
      }
      
      // Pick next regime that hasn't been completed in this cycle
      const nextRegime = trainingRegimes.find(regime => !regimesCompleted.current.includes(regime));
      
      // Start next round
      addSystemMessage(`Starting training on ${nextRegime} market conditions...`, "info");
      generateData(nextRegime);
      resetSimulation();
      setTimeout(() => setRunning(true), 500);
    }
  }, [roundEnded, autoTraining, currentRegimeType]);

  // Regenerate data function
  const regenerateData = () => {
    setRunning(false);
    generateData();
    resetSimulation();
  };
  
  // Start auto-training
  const startAutoTraining = () => {
    setAutoTraining(true);
    trainingCyclesRef.current = 0;
    regimesCompleted.current = [];
    addSystemMessage("Starting automated training across all market conditions...", "info");
    generateData("uptrend");
    resetSimulation();
    setTimeout(() => setRunning(true), 500);
  };

  // Extract features for HOD/LOD prediction
  const extractFeatures = (windowData) => {
    if (windowData.length < 20) return null;
    
    // Get most recent candles
    const recentCandles = windowData.slice(-20);
    
    // Calculate technical indicators and other features
    const closes = recentCandles.map(d => d.close);
    const highs = recentCandles.map(d => d.high);
    const lows = recentCandles.map(d => d.low);
    const volumes = recentCandles.map(d => d.volume);
    
    // Moving averages
    const sma5 = calculateSMA(closes, 5);
    const sma10 = calculateSMA(closes, 10);
    const sma20 = calculateSMA(closes, 20);
    
    // Price momentum
    const momentum5 = closes[closes.length - 1] / closes[closes.length - 6];
    const momentum10 = closes[closes.length - 1] / closes[closes.length - 11];
    
    // Volatility (ATR-like)
    const volatility = calculateATR(recentCandles, 14);
    
    // Volume trend
    const volumeChange = volumes[volumes.length - 1] / calculateSMA(volumes, 5);
    
    // Calculate relative position of current price within recent range
    const recentHighest = Math.max(...highs);
    const recentLowest = Math.min(...lows);
    const pricePosition = (closes[closes.length - 1] - recentLowest) / (recentHighest - recentLowest);
    
    // Calculate recent high/low relationship to predict if new highs/lows are likely
    const highestHighIdx = highs.indexOf(recentHighest);
    const lowestLowIdx = lows.indexOf(recentLowest);
    const daysSinceHigh = closes.length - 1 - highestHighIdx;
    const daysSinceLow = closes.length - 1 - lowestLowIdx;
    
    // Create feature vector
    return [
      closes[closes.length - 1] / sma5,                  // Price relative to SMA5
      closes[closes.length - 1] / sma10,                 // Price relative to SMA10
      closes[closes.length - 1] / sma20,                 // Price relative to SMA20
      momentum5,                                         // 5-day momentum
      momentum10,                                        // 10-day momentum
      volatility / closes[closes.length - 1],            // Normalized volatility
      volumeChange,                                      // Volume trend
      pricePosition,                                     // Relative price position
      1 / (daysSinceHigh + 1),                          // Recency of highest high
      1 / (daysSinceLow + 1)                            // Recency of lowest low
    ];
  };
  
  // Helper function to calculate Simple Moving Average
  const calculateSMA = (data, period) => {
    if (data.length < period) return data[data.length - 1];
    
    const sum = data.slice(-period).reduce((a, b) => a + b, 0);
    return sum / period;
  };
  
  // Helper function to calculate Average True Range
  const calculateATR = (candles, period) => {
    if (candles.length < 2) return 0;
    
    const trueRanges = [];
    for (let i = 1; i < candles.length; i++) {
      const highLow = candles[i].high - candles[i].low;
      const highPrevClose = Math.abs(candles[i].high - candles[i-1].close);
      const lowPrevClose = Math.abs(candles[i].low - candles[i-1].close);
      
      trueRanges.push(Math.max(highLow, highPrevClose, lowPrevClose));
    }
    
    if (trueRanges.length < period) {
      return trueRanges.reduce((a, b) => a + b, 0) / trueRanges.length;
    }
    
    return trueRanges.slice(-period).reduce((a, b) => a + b, 0) / period;
  };
  
  // Simple neural network prediction using features and weights
  const predictWithNetwork = (features, weights, bias) => {
    let sum = bias;
    for (let i = 0; i < features.length; i++) {
      sum += features[i] * weights[i];
    }
    return sum;
  };
  
  // Make HOD/LOD predictions for a trader
  const makePredictions = (trader, windowData, lookAhead = 1) => {
    if (windowData.length < 20) return null;
    
    const features = extractFeatures(windowData);
    if (!features) return null;
    
    const network = traderNetworks[trader.id];
    
    // Get current close price for reference
    const currentClose = windowData[windowData.length - 1].close;
    
    // Get recent highs/lows for context
    const recentCandles = windowData.slice(-20);
    const recentHighs = recentCandles.map(c => c.high);
    const recentLows = recentCandles.map(c => c.low);
    const recentHighest = Math.max(...recentHighs);
    const recentLowest = Math.min(...recentLows);
    
    // Adjust prediction strategy based on trader's specialty
    let hodMultiplier = 1.0;
    let lodMultiplier = 1.0;
    
    if (trader.strategy === 'trend') {
      // Trend followers expect more extreme values in the trending direction
      const trend = detectMarketStructure(windowData);
      if (trend === 'uptrend') {
        hodMultiplier = 1.2;
        lodMultiplier = 0.9;
      } else if (trend === 'downtrend') {
        hodMultiplier = 0.9;
        lodMultiplier = 1.2;
      }
    } else if (trader.strategy === 'reversal') {
      // Mean reversal expects moves counter to recent extremes
      if (currentClose > recentHighest * 0.95) {
        hodMultiplier = 0.9;
        lodMultiplier = 1.1;
      } else if (currentClose < recentLowest * 1.05) {
        hodMultiplier = 1.1;
        lodMultiplier = 0.9;
      }
    } else if (trader.strategy === 'volatility') {
      // Volatility strategies expect larger ranges in choppy or volatile markets
      const volatility = calculateATR(recentCandles, 14) / currentClose;
      if (volatility > 0.01) {
        hodMultiplier = 1.1;
        lodMultiplier = 1.1;
      }
    }
    
    // Make raw predictions using neural network weights
    const rawHodPrediction = predictWithNetwork(features, network.hodWeights, network.biasHod);
    const rawLodPrediction = predictWithNetwork(features, network.lodWeights, network.biasLod);
    
    // Apply strategy-based adjustments and scaling
    // Normalize to get percentage change from current price
    const hodPctChange = 0.02 * rawHodPrediction * hodMultiplier; // Max ~2% move
    const lodPctChange = -0.02 * rawLodPrediction * lodMultiplier; // Max ~2% move
    
    // Calculate actual price predictions
    let hodPrediction = currentClose * (1 + hodPctChange);
    let lodPrediction = currentClose * (1 + lodPctChange);
    
    // Ensure LOD is lower than HOD
    if (lodPrediction >= hodPrediction) {
      const midPoint = (hodPrediction + lodPrediction) / 2;
      hodPrediction = midPoint * 1.01;
      lodPrediction = midPoint * 0.99;
    }
    
    return {
      hodPrediction,
      lodPrediction,
      features,
      currentClose
    };
  };
  
  // Update neural network weights through reinforcement learning
  const updateNetworkWeights = (traderId, features, actual, prediction, isHod) => {
    // Calculate error
    const error = actual - prediction;
    const errorPct = Math.abs(error / actual);
    
    // Only reward/penalize if significant error
    if (errorPct < 0.0025) return true; // Consider as correct if within 0.25%
    
    // Get current weights
    const network = {...traderNetworks[traderId]};
    const trader = traders.find(t => t.id === traderId);
    const weights = isHod ? network.hodWeights : network.lodWeights;
    const bias = isHod ? network.biasHod : network.biasLod;
    
    // Adjust weights and bias based on error
    const learningRate = trader.learningRate * (errorPct > 0.01 ? 1.2 : 0.8); // Adjust rate based on error size
    
    let updatedWeights = [...weights];
    let updatedBias = bias;
    
    // Reinforcement learning: only update weights if prediction was incorrect
    if (errorPct > 0.005) {
      // Update weights based on error and features
      updatedWeights = weights.map((w, i) => {
        const sign = actual > prediction ? 1 : -1;
        return w + sign * learningRate * features[i] * errorPct;
      });
      
      // Update bias
      updatedBias = bias + learningRate * (actual > prediction ? 1 : -1) * errorPct;
    }
    
    // Update the network
    setTraderNetworks(prev => {
      const newNetworks = {...prev};
      if (isHod) {
        newNetworks[traderId].hodWeights = updatedWeights;
        newNetworks[traderId].biasHod = updatedBias;
      } else {
        newNetworks[traderId].lodWeights = updatedWeights;
        newNetworks[traderId].biasLod = updatedBias;
      }
      return newNetworks;
    });
    
    return errorPct <= 0.01; // Success if error is less than 1%
  };
  
  // Evaluate prediction accuracy
  const evaluatePrediction = (traderId, prediction, actualData) => {
    if (!prediction) return null;
    
    const actualHod = Math.max(...actualData.map(d => d.high));
    const actualLod = Math.min(...actualData.map(d => d.low));
    
    // Calculate accuracy as percentage error
    const hodError = Math.abs((prediction.hodPrediction - actualHod) / actualHod);
    const lodError = Math.abs((prediction.lodPrediction - actualLod) / actualLod);
    
    // Success if error is less than thresholds (1% for now)
    const hodSuccess = updateNetworkWeights(traderId, prediction.features, actualHod, prediction.hodPrediction, true);
    const lodSuccess = updateNetworkWeights(traderId, prediction.features, actualLod, prediction.lodPrediction, false);
    
    // Calculate accuracy scores (100% - error%)
    const hodAccuracy = Math.max(0, 100 - (hodError * 100));
    const lodAccuracy = Math.max(0, 100 - (lodError * 100));
    
    // Return evaluation results
    return {
      hodSuccess,
      lodSuccess,
      hodAccuracy,
      lodAccuracy,
      actualHod,
      actualLod,
      hodError,
      lodError
    };
  };
  
  // Update trader statistics
  const updateTraderStats = (trader, evaluation) => {
    if (!evaluation) return trader;
    
    const { hodSuccess, lodSuccess, hodAccuracy, lodAccuracy } = evaluation;
    
    // Calculate success rate for this prediction
    const overallSuccess = (hodSuccess && lodSuccess) || 
                          (hodSuccess && lodAccuracy > 95) || 
                          (lodSuccess && hodAccuracy > 95);
    
    // Update consecutive win/loss streaks
    let consecutiveWins = trader.consecutiveWins;
    let consecutiveLosses = trader.consecutiveLosses;
    
    if (overallSuccess) {
      consecutiveWins += 1;
      consecutiveLosses = 0;
    } else {
      consecutiveWins = 0;
      consecutiveLosses += 1;
    }
    
    // Update accuracy metrics with exponential moving average
    const alpha = 0.05; // Weight for newest data
    const newHodAccuracy = alpha * hodAccuracy + (1 - alpha) * trader.accuracy.hod;
    const newLodAccuracy = alpha * lodAccuracy + (1 - alpha) * trader.accuracy.lod;
    
    // Update trader object
    return {
      ...trader,
      wins: overallSuccess ? trader.wins + 1 : trader.wins,
      accuracy: {
        hod: newHodAccuracy,
        lod: newLodAccuracy
      },
      consecutiveWins,
      consecutiveLosses
    };
  };

  // Run simulation
  useEffect(() => {
    if (!running || day >= data.length - 5) { // Need at least 5 days for prediction eval
      if (day >= data.length - 5 && !roundEnded) {
        setRoundEnded(true);
        // Determine the winner
        const winner = [...traders].sort((a, b) => 
          b.accuracy.hod + b.accuracy.lod - (a.accuracy.hod + a.accuracy.lod)
        )[0];
        
        // Log end of round
        addSystemMessage(`${marketLabel} prediction training complete. Best performer: ${winner.name}`, "success");
      }
      return;
    }
    
    const timer = setTimeout(() => {
      const windowData = data.slice(0, day + 1);
      const currentCandle = data[day];
      const futureData = data.slice(day + 1, day + 6); // Next 5 days for evaluation
      setCurrentPrice(currentCandle.close);
      
      // Update each trader
      const updatedTraders = [...traders].map(trader => {
        // Make predictions for next 5 days
        const prediction = makePredictions(trader, windowData, 5);
        
        // Store prediction
        const updatedTrader = {
          ...trader,
          predictions: [...trader.predictions, {
            day,
            prediction
          }]
        };
        
        // Evaluate previous predictions (from 5 days ago)
        if (day >= 5) {
          const prevPredIndex = updatedTrader.predictions.findIndex(p => p.day === day - 5);
          
          if (prevPredIndex >= 0) {
            const prevPrediction = updatedTrader.predictions[prevPredIndex].prediction;
            const actualDataForEval = data.slice(day - 5 + 1, day + 1); // 5 days after prediction
            
            // Evaluate prediction
            const evaluation = evaluatePrediction(trader.id, prevPrediction, actualDataForEval);
            
            if (evaluation) {
              // Update trader stats
              return updateTraderStats(updatedTrader, evaluation);
            }
          }
        }
        
        return updatedTrader;
      });
      
      // Update stats for display
      const newStats = {};
      updatedTraders.forEach(trader => {
        const winRatio = trader.wins / Math.max(1, trader.predictions.length - 5);
        const totalAccuracy = (trader.accuracy.hod + trader.accuracy.lod) / 2;
        
        newStats[trader.id] = {
          hodAccuracy: trader.accuracy.hod.toFixed(2),
          lodAccuracy: trader.accuracy.lod.toFixed(2),
          totalAccuracy: totalAccuracy.toFixed(2),
          winRate: (winRatio * 100).toFixed(2),
          drawdown: (trader.consecutiveLosses * 5).toFixed(2) // Simple drawdown estimation
        };
      });
      
      setPredictionStats(newStats);
      setTraders(updatedTraders);
      setDay(prev => prev + 1);
    }, speed);
    
    return () => clearTimeout(timer);
  }, [running, day, data, speed, traders, traderNetworks]);

  // Scroll message log to bottom when new messages arrive
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = 0;
    }
  }, [systemMessages]);

  // Reset simulation
  const resetSimulation = () => {
    setDay(0);
    setRoundEnded(false);
    setTraders(prevTraders => prevTraders.map(trader => ({
      ...trader,
      predictions: [],
      wins: 0,
      consecutiveWins: 0,
      consecutiveLosses: 0
    })));
    setRunning(false);
  };

  // Set selected AI for production use
  const selectAI = (traderId) => {
    const trader = traders.find(t => t.id === traderId);
    setSelectedAI(traderId);
    addSystemMessage(`Selected ${trader.name} as primary prediction AI`, "success");
  };

  // Render candlestick chart
  const renderChart = () => {
    const chartWidth = 900;  // Increased from 700
    const chartHeight = 500; // Increased from 300
    const padding = 50;      // Increased from 40
    const visibleCandles = 100; // Increased from 70
    
    // Calculate start index to show a moving window of the chart
    const startIdx = Math.max(0, day - visibleCandles + 5);
    const endIdx = day + 1;
    const displayData = data.slice(startIdx, endIdx);
    
    if (displayData.length === 0) return <div>Loading chart...</div>;
    
    const maxPrice = Math.max(...displayData.map(d => d.high)) * 1.02;
    const minPrice = Math.min(...displayData.map(d => d.low)) * 0.98;
    const priceRange = maxPrice - minPrice;
    
    const scaleY = (price) => chartHeight - padding - ((price - minPrice) / priceRange) * (chartHeight - 2 * padding);
    const candleWidth = Math.min(8, (chartWidth - 2 * padding) / Math.max(20, displayData.length));
    const scaleX = (i) => padding + i * candleWidth;

    // Get predictions for visualization
    const activePredictions = traders
      .filter(t => showAll || t.id === activeTrader || t.id === selectedAI)
      .map(trader => {
        // Find the most recent prediction
        const latestPrediction = trader.predictions.length > 0 
          ? trader.predictions[trader.predictions.length - 1].prediction 
          : null;
          
        return {
          traderId: trader.id,
          color: trader.color,
          prediction: latestPrediction
        };
      })
      .filter(p => p.prediction !== null);

    return (
      <svg width={chartWidth} height={chartHeight} className="bg-gray-100 rounded-lg">
        {/* Price axis */}
        <line 
          x1={padding} y1={padding} 
          x2={padding} y2={chartHeight - padding} 
          stroke="#666" 
          strokeWidth="1"
        />
        {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
          const price = minPrice + priceRange * ratio;
          const y = scaleY(price);
          return (
            <g key={ratio}>
              <line 
                x1={padding - 5} y1={y} 
                x2={chartWidth - padding} y2={y} 
                stroke="#666" 
                strokeWidth="0.5" 
                strokeDasharray="2,2"
              />
              <text x={padding - 8} y={y + 4} textAnchor="end" fontSize="10">
                {price.toFixed(2)}
              </text>
            </g>
          );
        })}
        
        {/* Time axis */}
        <line 
          x1={padding} y1={chartHeight - padding} 
          x2={chartWidth - padding} y2={chartHeight - padding} 
          stroke="#666" 
          strokeWidth="1"
        />
        
        {/* Prediction lines (HOD/LOD) */}
        {activePredictions.map((pred, i) => {
          const x1 = scaleX(displayData.length - 1);
          const x2 = scaleX(displayData.length - 1 + 5); // Extending 5 days ahead
          
          return (
            <g key={`prediction-${pred.traderId}`}>
              {/* HOD Prediction */}
              <line 
                x1={x1} 
                y1={scaleY(pred.prediction.hodPrediction)} 
                x2={x2} 
                y2={scaleY(pred.prediction.hodPrediction)} 
                stroke={pred.color} 
                strokeWidth="1.5" 
                strokeDasharray="5,3"
              />
              <circle 
                cx={x2} 
                cy={scaleY(pred.prediction.hodPrediction)} 
                r="3" 
                fill={pred.color}
              />
              <text 
                x={x2 + 5} 
                y={scaleY(pred.prediction.hodPrediction)} 
                fontSize="10" 
                fill={pred.color}
              >
                HOD: {pred.prediction.hodPrediction.toFixed(2)}
              </text>
              
              {/* LOD Prediction */}
              <line 
                x1={x1} 
                y1={scaleY(pred.prediction.lodPrediction)} 
                x2={x2} 
                y2={scaleY(pred.prediction.lodPrediction)} 
                stroke={pred.color} 
                strokeWidth="1.5" 
                strokeDasharray="5,3"
              />
              <circle 
                cx={x2} 
                cy={scaleY(pred.prediction.lodPrediction)} 
                r="3" 
                fill={pred.color}
              />
              <text 
                x={x2 + 5} 
                y={scaleY(pred.prediction.lodPrediction)} 
                fontSize="10" 
                fill={pred.color}
              >
                LOD: {pred.prediction.lodPrediction.toFixed(2)}
              </text>
            </g>
          );
        })}
        
        {/* Candlesticks */}
        {displayData.map((candle, i) => {
          const x = scaleX(i);
          const open = scaleY(candle.open);
          const close = scaleY(candle.close);
          const high = scaleY(candle.high);
          const low = scaleY(candle.low);
          const color = candle.close >= candle.open ? "green" : "red";
          
          return (
            <g key={i}>
              {/* Wick */}
              <line 
                x1={x + candleWidth/2} y1={high} 
                x2={x + candleWidth/2} y2={low} 
                stroke={color} 
                strokeWidth="1"
              />
              {/* Body */}
              <rect 
                x={x} 
                y={Math.min(open, close)} 
                width={candleWidth} 
                height={Math.abs(close - open) || 1} 
                fill={color} 
                stroke={color}
              />
            </g>
          );
        })}
        
        {/* Indicator for current day */}
        <line 
          x1={scaleX(displayData.length - 1)} 
          y1={padding} 
          x2={scaleX(displayData.length - 1)} 
          y2={chartHeight - padding} 
          stroke="#6366f1" 
          strokeWidth="1" 
          strokeDasharray="4,4"
        />
      </svg>
    );
  };
  
  // Calculate trader statistics
  const calculateTraderStats = (trader) => {
    return {
      totalPredictions: trader.predictions.length,
      evaluatedPredictions: Math.max(0, trader.predictions.length - 5),
      successRate: predictionStats[trader.id].winRate,
      hodAccuracy: predictionStats[trader.id].hodAccuracy,
      lodAccuracy: predictionStats[trader.id].lodAccuracy,
      totalAccuracy: predictionStats[trader.id].totalAccuracy,
      drawdown: predictionStats[trader.id].drawdown
    };
  };
  
  // Legend item component
  const LegendItem = ({ color, label }) => (
    <div className="flex items-center mr-4">
      <div className="w-3 h-3 mr-1" style={{ backgroundColor: color }}></div>
      <span className="text-xs">{label}</span>
    </div>
  );
  
  return (
    <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-lg">
      <h2 className="text-xl font-bold mb-2">Advanced HOD/LOD Prediction Backtesting</h2>
      
      {roundEnded && (
        <div className="w-full bg-yellow-100 border-l-4 border-yellow-500 p-4 mb-4">
          <div className="flex justify-between items-center">
            <div>
              <p className="font-bold">Backtesting Round Complete!</p>
              <p>
                Best Performer: {traders.sort((a, b) => 
                  parseFloat(predictionStats[b.id].totalAccuracy) - parseFloat(predictionStats[a.id].totalAccuracy)
                )[0].name} with 
                {' ' + traders.sort((a, b) => 
                  parseFloat(predictionStats[b.id].totalAccuracy) - parseFloat(predictionStats[a.id].totalAccuracy)
                )[0].wins} wins
              </p>
            </div>
            <div className="flex space-x-2">
              <button 
                onClick={regenerateData}
                className="px-4 py-2 bg-blue-600 text-white rounded"
              >
                New Market
              </button>
              {!autoTraining && (
                <button 
                  onClick={startAutoTraining}
                  className="px-4 py-2 bg-purple-600 text-white rounded"
                >
                  Start Auto-Training
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      
      <div className="w-full grid grid-cols-3 gap-4 mb-4">
        <div>
          <div className="flex space-x-2">
            <select 
              value={patternType} 
              onChange={(e) => setPatternType(e.target.value)}
              className="p-2 border rounded"
              disabled={autoTraining}
            >
              <option value="all">All Market Types</option>
              <option value="mixed">Mixed Market</option>
              <option value="uptrend">Strong Uptrend</option>
              <option value="downtrend">Strong Downtrend</option>
              <option value="range">Range-Bound</option>
              <option value="choppy">Choppy Market</option>
            </select>
            <button 
              onClick={regenerateData} 
              className="px-4 py-2 bg-blue-500 text-white rounded"
              disabled={autoTraining}
            >
              New Market
            </button>
          </div>
        </div>
        
        <div className="flex flex-col items-center justify-center">
          <div className="text-lg font-semibold">{marketLabel}</div>
          <div className="text-sm">
            Day: {day}/{data.length}
            <div className="h-2 w-32 bg-gray-300 rounded mt-1">
              <div 
                className="h-full bg-blue-600 rounded" 
                style={{width: `${(day / data.length) * 100}%`}}
              ></div>
            </div>
          </div>
        </div>
        
        <div className="flex justify-end">
          {autoTraining ? (
            <div className="px-3 py-1 bg-purple-600 text-white rounded-full text-xs animate-pulse flex items-center">
              Auto-Training in Progress...
            </div>
          ) : (
            <button onClick={startAutoTraining} className="px-4 py-2 bg-purple-600 text-white rounded">
              Start Auto-Training
            </button>
          )}
        </div>
      </div>
      
      {/* Trader Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4 w-full">
        {traders.map(trader => {
          const isActive = trader.id === activeTrader;
          const isSelected = trader.id === selectedAI;
          const stats = calculateTraderStats(trader);
          
          return (
            <div 
              key={trader.id}
              className={`p-3 rounded shadow cursor-pointer transition-all ${
                isActive ? 'bg-gray-800 text-white scale-105' : 
                isSelected ? 'bg-blue-700 text-white' : 'bg-white hover:bg-gray-100'
              }`}
              onClick={() => setActiveTrader(trader.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div 
                    className="w-3 h-3 rounded-full mr-2" 
                    style={{backgroundColor: trader.color}}
                  ></div>
                  <h3 className="font-bold">{trader.name}</h3>
                </div>
                <div className="flex space-x-2">
                  <div className="text-xs px-2 py-1 rounded bg-gray-200 text-gray-800">
                    Wins: {trader.wins}
                  </div>
                  {!isSelected && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        selectAI(trader.id);
                      }}
                      className="text-xs px-2 py-1 rounded bg-blue-500 text-white"
                    >
                      Select
                    </button>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                <div>
                  <div className={isActive || isSelected ? 'text-gray-300' : 'text-gray-500'}>HOD Accuracy</div>
                  <div>{stats.hodAccuracy}%</div>
                </div>
                <div>
                  <div className={isActive || isSelected ? 'text-gray-300' : 'text-gray-500'}>LOD Accuracy</div>
                  <div>{stats.lodAccuracy}%</div>
                </div>
                <div>
                  <div className={isActive || isSelected ? 'text-gray-300' : 'text-gray-500'}>Win Rate</div>
                  <div>{stats.successRate}%</div>
                </div>
                <div>
                  <div className={isActive || isSelected ? 'text-gray-300' : 'text-gray-500'}>Max Drawdown</div>
                  <div>{stats.drawdown}%</div>
                </div>
              </div>
              <div className="mt-2 text-xs opacity-70">
                Strategy: {trader.strategy === 'trend' ? 'Trend Following' : 
                          trader.strategy === 'reversal' ? 'Mean Reversal' : 
                          'Volatility Breakout'}
              </div>
            </div>
          );
        })}
      </div>
      
      {renderChart()}
      
      <div className="flex flex-wrap mt-2 justify-center">
        {traders.map(trader => (
          <LegendItem key={trader.id} color={trader.color} label={`${trader.name} Predictions`} />
        ))}
      </div>
      
      <div className="flex mt-4 justify-between items-center w-full">
        <div className="flex space-x-2">
          <button 
            onClick={() => setRunning(!running)} 
            className={`px-4 py-2 rounded ${running ? 'bg-red-500' : 'bg-green-500'} text-white`}
            disabled={autoTraining && !running}
          >
            {running ? 'Pause' : 'Start'}
          </button>
          <button 
            onClick={resetSimulation} 
            className="px-4 py-2 bg-gray-500 text-white rounded"
            disabled={autoTraining}
          >
            Reset
          </button>
          <div className="flex items-center ml-4">
            <input 
              type="checkbox" 
              id="showAll" 
              checked={showAll} 
              onChange={() => setShowAll(!showAll)}
              className="mr-2" 
            />
            <label htmlFor="showAll" className="text-sm">Show All Traders</label>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-sm">Speed:</span>
          <select 
            value={speed} 
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="p-2 border rounded"
            disabled={autoTraining}
          >
            <option value={1000}>Slow</option>
            <option value={500}>Medium</option>
            <option value={200}>Fast</option>
            <option value={50}>Ultra Fast</option>
            <option value={10}>Maximum</option>
          </select>
        </div>
      </div>
      
      <div className="w-full grid grid-cols-2 gap-4 mt-4">
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="font-bold mb-2">Prediction Log: {traders.find(t => t.id === activeTrader)?.name}</h3>
          <div className="h-40 overflow-y-auto bg-white p-2 rounded">
            {traders.find(t => t.id === activeTrader)?.predictions.length === 0 ? (
              <div className="text-gray-500 text-sm">No predictions yet. Start the simulation to see prediction activity.</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left">Day</th>
                    <th className="text-left">HOD Pred.</th>
                    <th className="text-left">LOD Pred.</th>
                    <th className="text-left">Actual HOD</th>
                    <th className="text-left">Actual LOD</th>
                    <th className="text-right">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {traders.find(t => t.id === activeTrader)?.predictions
                    .slice()
                    .reverse()
                    .filter(pred => pred.prediction)
                    .map((pred, i) => {
                      // Find actual values if available (5 days after prediction)
                      const actualData = pred.day + 5 < day ? 
                        data.slice(pred.day + 1, pred.day + 6) : null;
                      
                      const actualHod = actualData ? 
                        Math.max(...actualData.map(d => d.high)).toFixed(2) : "-";
                      
                      const actualLod = actualData ? 
                        Math.min(...actualData.map(d => d.low)).toFixed(2) : "-";
                      
                      // Determine if prediction was successful
                      const evaluated = actualData !== null;
                      
                      return (
                        <tr key={i} className="border-b">
                          <td>{pred.day}</td>
                          <td>{pred.prediction.hodPrediction.toFixed(2)}</td>
                          <td>{pred.prediction.lodPrediction.toFixed(2)}</td>
                          <td>{actualHod}</td>
                          <td>{actualLod}</td>
                          <td className="text-right">
                            {evaluated ? (
                              <span className={
                                parseFloat(actualHod) <= pred.prediction.hodPrediction * 1.01 && 
                                parseFloat(actualHod) >= pred.prediction.hodPrediction * 0.99 && 
                                parseFloat(actualLod) <= pred.prediction.lodPrediction * 1.01 && 
                                parseFloat(actualLod) >= pred.prediction.lodPrediction * 0.99 
                                  ? 'text-green-500' : 'text-red-500'
                              }>
                                {parseFloat(actualHod) <= pred.prediction.hodPrediction * 1.01 && 
                                 parseFloat(actualHod) >= pred.prediction.hodPrediction * 0.99 && 
                                 parseFloat(actualLod) <= pred.prediction.lodPrediction * 1.01 && 
                                 parseFloat(actualLod) >= pred.prediction.lodPrediction * 0.99 
                                  ? '✓' : '✗'}
                              </span>
                            ) : '—'}
                          </td>
                        </tr>
                      );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
        
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="font-bold mb-2">Learning Progress & System Messages</h3>
          <div className="h-40 overflow-y-auto bg-white p-2 rounded" ref={messageListRef}>
            {systemMessages.length === 0 ? (
              <div className="text-gray-500 text-sm">No system messages yet.</div>
            ) : (
              <div className="space-y-2">
                {systemMessages.map(msg => (
                  <div 
                    key={msg.id} 
                    className={`text-sm p-2 rounded ${
                      msg.type === 'success' ? 'bg-green-100 text-green-800' :
                      msg.type === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-blue-100 text-blue-800'
                    }`}
                  >
                    <div className="flex justify-between">
                      <span>{msg.text}</span>
                      <span className="text-xs opacity-60">{msg.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-600 w-full">
        <p>
          <span className="font-bold">HOD/LOD Prediction:</span> Each AI predicts High of Day (HOD) and Low of Day (LOD) for the coming market days using machine learning.
        </p>
        <p>
          <span className="font-bold">Reinforcement Learning:</span> AIs learn from successful and failed predictions, improving accuracy over time.
        </p>
        <p>
          <span className="font-bold">Strategy Selection:</span> After training, select the best performing AI to use for real market predictions.
        </p>
      </div>
    </div>
  );
};

export default EnhancedBacktesting;
