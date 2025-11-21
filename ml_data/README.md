# MAC Metrics ML Dataset

## Overview
This directory contains processed MAC layer metrics from srsRAN collected via FlexRIC xApp, structured for machine learning applications.

## Data Collection
- **Source**: srsRAN eNodeB MAC layer metrics via E2 interface
- **Collection Method**: FlexRIC xApp with 10ms reporting interval
- **Format**: 2D matrix (samples × features)

## Dataset Structure

### Input Features (X)
| Column | Description | Unit |
|--------|-------------|------|
| rnti | Radio Network Temporary Identifier | - |
| phr | Power Headroom Report | dB |
| dl_tbs | Downlink Transport Block Size | bytes |
| ul_tbs | Uplink Transport Block Size | bytes |
| dl_aggr_prb | Downlink Aggregated PRBs | PRBs |
| wb_cqi | Wideband Channel Quality Indicator | 0-15 |
| pusch_snr | Physical Uplink Shared Channel SNR | dB |
| pucch_snr | Physical Uplink Control Channel SNR | dB |
| ul_rssi | Uplink Received Signal Strength Indicator | dBfs |
| dl_bler | Downlink Block Error Rate | 0.0-1.0 |
| ul_bler | Uplink Block Error Rate | 0.0-1.0 |
| dl_mcs | Downlink Modulation and Coding Scheme | 0-28 |
| ul_mcs | Uplink Modulation and Coding Scheme | 0-28 |

### Target Variable (Y)
- **Y = Future RSSI**: ul_rssi shifted by +1 timestep
- **Prediction Task**: Given current MAC metrics at time `t`, predict RSSI at time `t+1`
- **Use Case**: Proactive radio resource management, predictive QoS

## File Formats

### 1. CSV Format (`mac_ml_dataset.csv`)
- **Structure**: Combined X and Y in single file
- **Columns**: All features + `target_future_rssi`
- **Usage**: Easy to load with pandas

```python
import pandas as pd
df = pd.read_csv('mac_ml_dataset.csv')
X = df.drop('target_future_rssi', axis=1)
Y = df['target_future_rssi']
```

### 2. NumPy Arrays
- **X_features.npy**: Input features matrix (samples × 13)
- **Y_target.npy**: Target vector (samples,)
- **Usage**: Direct loading for NumPy/TensorFlow/PyTorch

```python
import numpy as np
X = np.load('X_features.npy')
Y = np.load('Y_target.npy')
```

### 3. Metadata
- **feature_names.txt**: Ordered list of feature column names

## Data Preprocessing Notes

1. **Time Shift**: Y[t] = RSSI[t+1]
   - Each sample X[t] predicts the NEXT timestep's RSSI
   - Last sample is dropped (no future label available)

2. **Missing Values**: 
   - Last row removed due to undefined future RSSI
   - All other samples complete (no NaN values expected)

3. **Temporal Ordering**:
   - Samples are time-ordered
   - Consider temporal split for train/test (not random split)

## Machine Learning Workflow

### Automated MLP Training (Built into xApp)

The xApp now includes an **automated MLP training pipeline** that executes after data collection:

**Pipeline Steps:**
1. ✅ **Drop Constant Features**: Remove features with zero variance
2. ✅ **Separate RSSI**: Keep `ul_rssi` unscaled (used as both feature and related to target)
3. ✅ **Scale Features**: Apply StandardScaler to all features except RSSI
4. ✅ **Recombine**: Add unscaled RSSI back to feature matrix
5. ✅ **Train/Test Split**: 90% training, 10% testing
6. ✅ **Train MLP**: 2-layer neural network (64→32 neurons)
7. ✅ **Evaluate**: Calculate MSE, RMSE, MAE, R² on test set
8. ✅ **Save Model**: Export trained model, scaler, and configuration

**Saved Artifacts:**
- `mlp_model.pkl` - Trained MLP regressor
- `scaler.pkl` - Fitted StandardScaler
- `model_config.txt` - Complete model documentation

**Model Architecture:**
```
Input Layer (12-13 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Output Layer (1 neuron) → Future RSSI
```

### Manual Training (Alternative Methods)

#### 1. Load Data
```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('X_features.npy')
Y = np.load('Y_target.npy')
```

#### 2. Split Dataset
```python
# Option A: Random split (for i.i.d. assumption)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Option B: Temporal split (for time-series)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]
```

#### 3. Feature Scaling (Optional)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 4. Train Model
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
```

#### 5. Evaluate
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.4f}")
print(f"R²: {r2_score(Y_test, Y_pred):.4f}")
```

## Example Models

### Linear Regression (Baseline)
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
```

### Random Forest (Ensemble)
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, Y_train)
```

### Neural Network (Deep Learning)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)
```

### LSTM (Time-Series)
```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

# Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

model = tf.keras.Sequential([
    LSTM(50, activation='relu', input_shape=(1, 13)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, Y_train, epochs=50, batch_size=32)
```

## Performance Metrics

### Regression Metrics
- **MSE** (Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (R-squared): Proportion of variance explained (0-1, higher better)
- **RMSE** (Root MSE): Error in same units as target

### Time-Series Metrics
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy**: % of correct trend predictions

## Data Collection Configuration

### eNodeB Configuration
- **Network**: 4G LTE (srsRAN)
- **Bandwidth**: 25 PRBs (5 MHz)
- **Channel**: AWGN (signal_power=35, snr=15)
- **RF**: tx_gain=60, rx_gain=40

### FlexRIC Configuration
- **Reporting Interval**: 10ms (ric.Interval_ms_10)
- **Protocol**: E2 interface
- **Service Model**: MAC SM v2

## Notes

1. **RSSI Range**: Typically -100 to -40 dBfs
2. **SNR Range**: Typically 0 to 40 dB
3. **CQI Range**: 0 (worst) to 15 (best)
4. **MCS Range**: 0 (QPSK 1/8) to 28 (64QAM 7/8)

## Using Trained Model for Inference

After training, you can use the saved model to predict future RSSI:

```python
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('/home/fahad/srsRAN_4g/ml_data/mlp_model.pkl', 'rb') as f:
    mlp = pickle.load(f)

with open('/home/fahad/srsRAN_4g/ml_data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new MAC metrics (example)
new_sample = {
    'rnti': 70,
    'phr': 40,
    'dl_tbs': 2000,
    'ul_tbs': 1500,
    'dl_aggr_prb': 2,
    'wb_cqi': 12,
    'pusch_snr': 22.5,
    'pucch_snr': 8.8,
    'ul_rssi': -45.2,  # Keep unscaled
    'dl_bler': 0.0,
    'ul_bler': 0.0,
    'dl_mcs': 20,
    'ul_mcs': 16
}

# Convert to DataFrame
df = pd.DataFrame([new_sample])

# Separate RSSI (keep unscaled)
rssi = df['ul_rssi'].values
X_without_rssi = df.drop(columns=['ul_rssi'])

# Scale features (except RSSI)
X_scaled = scaler.transform(X_without_rssi)

# Recombine with unscaled RSSI
X_final = np.column_stack([X_scaled, rssi])

# Predict future RSSI
future_rssi = mlp.predict(X_final)
print(f"Current RSSI: {rssi[0]:.2f} dBfs")
print(f"Predicted Future RSSI: {future_rssi[0]:.2f} dBfs")
```

## Troubleshooting

### Issue: Not enough samples
**Solution**: Increase `runtime` parameter in xApp (default: 5 seconds)

### Issue: All BLER values are 0
**Reason**: Good channel quality with HARQ recovering errors
**Solution**: Reduce SNR or increase traffic load

### Issue: Constant RSSI values
**Reason**: Static channel conditions
**Solution**: Add mobility or fading channel model

### Issue: MLP training fails or poor performance
**Reason**: Too few samples or no variance in features
**Solution**: 
- Collect data for longer duration (increase `runtime`)
- Ensure UE is transmitting data (run traffic generator)
- Check if features have sufficient variation

## Citation

If using this dataset, please reference:
- **srsRAN**: Open-source 4G/5G software radio suite
- **FlexRIC**: Open RAN E2 Agent and Near-RT RIC
- **Data Source**: MAC layer metrics via E2 SM interface
