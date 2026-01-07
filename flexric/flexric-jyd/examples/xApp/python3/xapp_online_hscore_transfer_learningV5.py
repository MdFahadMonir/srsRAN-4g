#!/usr/bin/env python3
"""
Online H-Score xApp
Combines:
- Data collection and threading from xapp_online_learning.py
- H-Score model loading and sliding window approach from xapp_load_hscore.py
"""

import xapp_sdk as ric
import time
import os
import queue
import threading
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

####################
#### CONFIGURATION
####################
RUNTIME = 120  # Total runtime in seconds (allow multiple training cycles)
MAC_INTERVAL = ric.Interval_ms_10  # 10ms interval
RLC_INTERVAL = ric.Interval_ms_10

# H-Score Model Configuration (must match training)
SEQ_LEN = 10
SCALER_TYPE = "minmax"
OUTPUT_DIM = 8
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.157
SPARSITY = 0.138
LEAKY = 0.797
TARGET_KPI_INDEX = 7  # ul_rssi is at index 7 in FEATURE_NAMES

# Training configuration
TRAINING_INTERVAL = 30  # Retrain h-net every 30 seconds (faster adaptation)
MIN_SAMPLES_FOR_TRAINING = 400  # Minimum samples before first training
BUFFER_SIZE = 3000  # Rolling buffer size (3000 samples = ~30 seconds of data @ 10ms interval)

# Batch and optimization
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.005
HNET_PATIENCE = 10

# Paths
ML_DATA_DIR = "/home/fahad/srsRAN_4g/ml_data"
HSCORE_MODEL_PATH = os.path.join(ML_DATA_DIR, "hscore_model.pt")
OUTPUT_DIR = ML_DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns (same as training)
FEATURE_NAMES = [
    "phr", "dl_tbs", "ul_tbs", "dl_aggr_prb",
    "wb_cqi", "pusch_snr", "pucch_snr", "ul_rssi",
    "dl_bler", "ul_bler", "dl_mcs", "ul_mcs",
]

# Device
SEED = 2
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cpu")  # Using CPU as per xapp_load_hscore
print(f"Using device: {device}")

####################
#### SHARED DATA STRUCTURES
####################

# Thread-safe queues
kpi_queue = queue.Queue()  # Raw KPI samples from E2 Agent
prediction_queue = queue.Queue()  # Samples ready for prediction (with future label)

# Rolling buffer for training (thread-safe with lock)
training_buffer_lock = threading.Lock()
training_buffer = deque(maxlen=BUFFER_SIZE)  # Automatically removes old samples

# Model versioning (thread-safe with lock)
model_lock = threading.Lock()
frozen_model = None  # Pre-trained ESN (frozen)
current_hnet = None  # Fine-tuned h-net
current_scaler_x = None
current_scaler_y = None
hnet_version = 0
last_training_max_timestamp = 0  # Track max timestamp used in last training

# Thread control
shutdown_event = threading.Event()
stats_lock = threading.Lock()

# Prediction error tracking
prediction_errors = deque(maxlen=1000)  # For trainer (cleared after each training)
prediction_errors_lock = threading.Lock()

# All prediction errors for final MSE calculation (never cleared)
all_prediction_errors = []  # Stores all errors throughout runtime
all_predictions = []  # Stores all predicted values
all_ground_truth = []  # Stores all actual values
all_prediction_errors_lock = threading.Lock()

# Live prediction MSE tracking with EMA
live_mse_ema = 0.0  # Exponential moving average of live prediction MSE
live_mse_ema_alpha = 0.2  # Weight for new errors (80% previous, 20% new)
live_mse_lock = threading.Lock()

# Statistics
stats = {
    'total_kpis_received': 0,
    'total_predictions': 0,
    'total_trainings': 0,
    'last_training_time': None,
    'last_prediction_time': None,
    'last_training_mse': None,
    'last_training_rmse': None,
    'last_training_r2': None,
    'last_training_corr': None,
    'last_validation_mse': None,  # MSE on hold-out validation set
    'last_previous_model_live_mse': None,  # Live MSE of previous model version during inter-training interval (Test thread predictions)
    'mse_history': []
}

####################
#### H-SCORE MODEL CLASSES
####################

class ESN(nn.Module):
    def __init__(self, input_dim, window_len, reservoir_size, spectral_radius, sparsity, leaky, output_dim):
        super(ESN, self).__init__()
        self.window_len = window_len
        self.reservoir_size = reservoir_size
        self.leaky = leaky
        self.input_dim = input_dim
        
        input_weights = torch.empty(input_dim, reservoir_size)
        nn.init.xavier_uniform_(input_weights)
        self.register_buffer("input_weights", input_weights)
        
        reservoir_weights = torch.empty(reservoir_size, reservoir_size)
        nn.init.xavier_uniform_(reservoir_weights)
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        reservoir_weights[mask] = 0
        current_radius = torch.max(torch.abs(torch.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= spectral_radius / current_radius
        self.register_buffer("reservoir_weights", reservoir_weights)

        bias = torch.zeros(reservoir_size)
        self.register_buffer("bias", bias)
        
        self.readout = nn.Linear(reservoir_size*window_len + input_dim * window_len, output_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_skip = x.reshape(batch_size, -1)
        reservoir_state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        states_stack = []
        
        for t in range(seq_len-1, -1, -1):
            u_t = x[:, t, :]
            pre_act = torch.matmul(u_t, self.input_weights) + torch.matmul(reservoir_state, self.reservoir_weights.T) + self.bias
            reservoir_state = (1 - self.leaky) * reservoir_state + self.leaky * torch.tanh(pre_act)
            states_stack.append(reservoir_state)
        
        states_stack = list(reversed(states_stack))
        states_stack = torch.cat(states_stack, dim=1)
        combined = torch.cat([states_stack, x_skip], dim=1)
        output = self.readout(combined)
        return output


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) > 2:
            batch, _, _ = x.shape
            x = x.reshape(batch, -1)
        return self.model(x)


class fg_nn(nn.Module):
    def __init__(self, fnet, gnet):
        super().__init__()
        self.fnet = fnet
        self.gnet = gnet

    def forward(self, x, y):
        return self.fnet(x), self.gnet(y)

    def get_f(self, x):
        return self.fnet(x)

    def get_g(self, y):
        return self.gnet(y)


####################
#### HELPER FUNCTIONS
####################

def load_hscore_model():
    """Load pre-trained frozen H-Score model (ESN + g-net)"""
    print(f"\n[Init] Loading frozen H-Score model from {HSCORE_MODEL_PATH}...")
    
    F = len(FEATURE_NAMES)
    
    fnet = ESN(
        input_dim=F,
        window_len=SEQ_LEN,
        reservoir_size=RESERVOIR_SIZE,
        spectral_radius=SPECTRAL_RADIUS,
        sparsity=SPARSITY,
        leaky=LEAKY,
        output_dim=OUTPUT_DIM
    ).to(device)
    
    gnet = MLP([F, 32, 16, OUTPUT_DIM]).to(device)
    
    model = fg_nn(fnet, gnet).to(device)
    
    if not os.path.exists(HSCORE_MODEL_PATH):
        raise FileNotFoundError(f"H-Score model not found at {HSCORE_MODEL_PATH}")
    
    model.load_state_dict(torch.load(HSCORE_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Frozen in eval mode
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"[Init] ✓ H-Score model loaded and frozen")
    return model


def initialize_hnet():
    """Initialize trainable h-net (MLP predictor)"""
    hnet = MLP([OUTPUT_DIM, 16, 32, 1], nn.ReLU()).to(device)
    print(f"[Init] ✓ h-net initialized: {OUTPUT_DIM} → 16 → 32 → 1")
    return hnet


def make_sliding_windows(dataset, seq_len):
    """
    Create sliding windows with newest → oldest ordering.
    TRUE one-step-ahead prediction structure.
    
    Y[i] = dataset[i] (newest sample, the target)
    X[i] = dataset[i+1:i+1+seq_len] (next seq_len older samples, input)
    
    Example (seq_len=10):
        dataset[0] = t100 (newest) → Y[0] (target)
        dataset[1:11] = [t99, t98, ..., t90] → X[0] (input window)
    
    Returns: X (N, seq_len, F), Y (N, F)
    """
    T, F = dataset.shape
    N = T - seq_len
    
    if N <= 0:
        return None, None
    
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    Y = np.zeros((N, F), dtype=np.float32)
    
    for i in range(N):
        # Y is the target (one step ahead, more recent)
        Y[i] = dataset[i, :]
        # X is the input window (seq_len samples starting from i+1, older than Y)
        X[i] = dataset[i + 1 : i + 1 + seq_len, :]
    
    return X, Y


def scale_data(X, Y, scaler_x=None, scaler_y=None, fit_on_train_only=False, train_size=0.8):
    """Scale X and Y using MinMaxScaler. If scalers provided, use them; otherwise fit new ones.
    
    Args:
        fit_on_train_only: If True, fit scalers only on training portion (temporal split)
        train_size: Fraction of data to use for fitting (when fit_on_train_only=True)
    """
    seq_len = X.shape[1]
    input_dim = X.shape[2]
    N = X.shape[0]  # Number of windows

    if scaler_x is None:
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        # Flatten X to scale per feature
        Xf = X.reshape(-1, input_dim)  # Shape: (N*seq_len, input_dim)
        
        if fit_on_train_only:
            # Fit only on training portion (temporal split: older data)
            # Use first 80% of windows, which corresponds to first 80% * seq_len rows in flattened array
            train_windows = int(N * train_size)
            train_samples = train_windows * seq_len
            Xf_train = Xf[:train_samples]
            scaler_x.fit(Xf_train)
            Xf = scaler_x.transform(Xf)
        else:
            Xf = scaler_x.fit_transform(Xf)
        X = Xf.reshape(-1, seq_len, input_dim)
    else:
        Xf = X.reshape(-1, input_dim)
        Xf = scaler_x.transform(Xf)
        X = Xf.reshape(-1, seq_len, input_dim)

    if scaler_y is None:
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        if fit_on_train_only:
            # Fit only on training portion (temporal split: older data)
            train_samples = int(len(Y) * train_size)
            Y_train_portion = Y[:train_samples]
            scaler_y.fit(Y_train_portion)
            Y = scaler_y.transform(Y)
        else:
            Y = scaler_y.fit_transform(Y)
    else:
        Y = scaler_y.transform(Y)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        scaler_x, scaler_y
    )


def inverse_scale_data(predictions, ground_truth, scaler, column_idx=None):
    """Inverse transform predictions and ground truth."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    if scaler is not None:
        if column_idx is not None and predictions.shape[1] == 1 and hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 1:
            pred_dummy = np.zeros((predictions.shape[0], scaler.n_features_in_))
            truth_dummy = np.zeros((ground_truth.shape[0], scaler.n_features_in_))

            pred_dummy[:, column_idx] = predictions.reshape(-1)
            if ground_truth.shape[1] > 1:
                truth_dummy = ground_truth
            else:
                truth_dummy[:, column_idx] = ground_truth.reshape(-1)

            predictions = scaler.inverse_transform(pred_dummy)[:, column_idx].reshape(-1, 1)
            ground_truth = scaler.inverse_transform(truth_dummy)[:, column_idx].reshape(-1, 1)
        else:
            predictions = scaler.inverse_transform(predictions)
            ground_truth = scaler.inverse_transform(ground_truth)

    return predictions, ground_truth


def calc_metric(ground_truth, predictions):
    """Calculate MSE, RMSE, R2, and correlation metrics."""
    mse = mean_squared_error(ground_truth[:, 0], predictions[:, 0])
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth[:, 0], predictions[:, 0])
    
    # Check if either array has zero variance
    if np.std(ground_truth[:, 0]) == 0 or np.std(predictions[:, 0]) == 0:
        print(f"  ⚠ Warning: Zero variance detected!")
        print(f"    Ground truth std: {np.std(ground_truth[:, 0]):.6f}")
        print(f"    Predictions std:  {np.std(predictions[:, 0]):.6f}")
        print(f"    Predictions unique values: {np.unique(predictions[:, 0])[:10]}")
        corr = 0.0  # or np.nan
    else:
        corr = np.corrcoef(ground_truth[:, 0], predictions[:, 0])[0, 1]
    
    return mse, rmse, r2, corr


####################
#### MAC CALLBACK
####################

class MACCallback(ric.mac_cb):
    def __init__(self):
        ric.mac_cb.__init__(self)
        self.sample_count = 0  # For debugging first few samples
    
    def handle(self, ind):
        if len(ind.ue_stats) > 0:
            stats_data = ind.ue_stats[0]
            
            # ul_rssi is in dBm (MinMaxScaler will handle normalization)
            ul_rssi = stats_data.ul_rssi
            
            # Debug logging for first 5 samples to verify values
            if self.sample_count < 5:
                print(f"[MAC Debug] Sample {self.sample_count+1}: "
                      f"ul_rssi={ul_rssi:.2f} dBm")
                self.sample_count += 1
            
            kpi_sample = {
                'timestamp': ind.tstamp / 1.0,
                'rnti': stats_data.rnti,
                'phr': stats_data.phr,
                'dl_tbs': stats_data.dl_aggr_tbs,
                'ul_tbs': stats_data.ul_aggr_tbs,
                'dl_aggr_prb': stats_data.dl_aggr_prb,
                'wb_cqi': stats_data.wb_cqi,
                'pusch_snr': stats_data.pusch_snr,
                'pucch_snr': stats_data.pucch_snr,
                'ul_rssi': ul_rssi,  #(scaled in dBm)
                'dl_bler': stats_data.dl_bler,
                'ul_bler': stats_data.ul_bler,
                'dl_mcs': stats_data.dl_mcs1,
                'ul_mcs': stats_data.ul_mcs1
            }
            
            kpi_queue.put(kpi_sample)


####################
#### RLC CALLBACK
####################

class RLCCallback(ric.rlc_cb):
    def __init__(self):
        ric.rlc_cb.__init__(self)
    
    def handle(self, ind):
        # RLC metrics for additional context (optional)
        pass


####################
#### H-NET TRAINING
####################

def train_hnet(model, hnet, d_des, dl_train, dl_test, epochs, lr, criterion_h, pi):
    """Train h-net while keeping model (ESN) frozen."""
    model.eval()
    h_opt = optim.Adam(hnet.parameters(), lr=lr)

    min_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        hnet.train()
        h_train_loss = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb[:, [pi]]
            h_opt.zero_grad()
            with torch.no_grad():
                f_out = model.get_f(xb)
            f_out = f_out[:, :d_des]
            h_out = hnet(f_out)
            loss = criterion_h(h_out, yb)
            loss.backward()
            h_opt.step()
            h_train_loss += loss.item()
        h_train_loss /= len(dl_train)

        hnet.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb[:, [pi]]
                f_out = model.get_f(xb)
                f_out = f_out[:, :d_des]
                h_out = hnet(f_out)
                loss = criterion_h(h_out, yb)
                test_loss += loss.item()
        test_loss /= len(dl_test)

        if test_loss < min_val_loss:
            min_val_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HNET_PATIENCE:
                break

    return h_train_loss, test_loss


def predict_nestedh(model, hnet, dl_test, d_des, pi):
    """Make predictions using frozen model + h-net."""
    model.eval()
    hnet.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            yb = yb[:, pi].reshape(-1, 1)
            f_out = model.get_f(xb)
            f_out = f_out[:, :d_des]
            h_out = hnet(f_out)
            predictions.append(h_out.cpu().numpy())
            ground_truth.append(yb.cpu().numpy())

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    return predictions, ground_truth


####################
#### THREAD 1: KPI COLLECTOR
####################

def kpi_collector_thread():
    """
    Receives KPI samples, stores in training buffer.
    Follows xapp_online_learning.py style.
    """
    print(f"[KPI Collector] Thread started")
    
    while not shutdown_event.is_set():
        try:
            kpi_sample = kpi_queue.get(timeout=0.1)
            
            with stats_lock:
                stats['total_kpis_received'] += 1
            
            # Add to training buffer (raw samples)
            with training_buffer_lock:
                training_buffer.append(kpi_sample)
        
        except queue.Empty:
            continue
    
    print(f"[KPI Collector] Thread stopped")


####################
#### THREAD 2: TRAINER
####################

def trainer_thread():
    """
    Periodically retrains h-net using the latest training buffer.
    Uses sliding window approach from xapp_load_hscore.py.
    """
    global current_hnet, current_scaler_x, current_scaler_y, hnet_version, last_training_max_timestamp
    
    print(f"[Trainer] Thread started")
    print(f"[Trainer] First training: as soon as buffer fills to {BUFFER_SIZE}")
    print(f"[Trainer] Subsequent trainings: every {TRAINING_INTERVAL} seconds")
    
    last_train_time = time.time()
    first_training_done = False  # Track if first training completed
    
    while not shutdown_event.is_set():
        current_time = time.time()
        
        # Check if it's time to retrain
        time_since_last_train = current_time - last_train_time
        
        # First training: trigger as soon as buffer is full
        # Subsequent trainings: use time interval
        should_train = False
        if not first_training_done:
            # First training: only check buffer size
            should_train = True  # Will check buffer size below
            trigger_reason = "buffer full"
        else:
            # Subsequent trainings: check time interval
            should_train = (time_since_last_train >= TRAINING_INTERVAL)
            trigger_reason = f"interval ({time_since_last_train:.1f}s >= {TRAINING_INTERVAL}s)"
        
        if should_train:
            # Get training data from buffer
            with training_buffer_lock:
                buffer_size = len(training_buffer)
                # First training: require full buffer (1200 samples)
                # Subsequent trainings: minimum 400 samples (but buffer usually has 1200)
                min_required = BUFFER_SIZE if not first_training_done else MIN_SAMPLES_FOR_TRAINING
                
                if buffer_size >= min_required:
                    training_data = list(training_buffer)
                else:
                    training_data = None
            
            if training_data:
                samples_count = len(training_data)
                print(f"\n[Trainer] Trigger: {trigger_reason} | Buffer: {buffer_size} samples")
                print(f"[Trainer] Starting training #{hnet_version + 1} with {samples_count} samples...")
                train_start = time.time()
                
                try:
                    # Convert to DataFrame
                    df = pd.DataFrame(training_data)
                    print(f"[Trainer] DEBUG: Created DataFrame with {len(df)} rows")
                    df = df.dropna()
                    print(f"[Trainer] DEBUG: After dropna: {len(df)} rows")
                    
                    # Track max timestamp for this training batch
                    max_timestamp_in_batch = df['timestamp'].max()
                    
                    # Sort by timestamp (oldest → newest), then reverse to (newest → oldest)
                    # This matches xapp_load_hscore.py exactly
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Extract features and reverse (newest → oldest)
                    print(f"[Trainer] DEBUG: Extracting features: {FEATURE_NAMES}")
                    dataset = df[FEATURE_NAMES].values.astype(np.float32)
                    print(f"[Trainer] DEBUG: Dataset shape before reverse: {dataset.shape}")
                    dataset = dataset[::-1]  # Reverse to newest → oldest
                    T, F = dataset.shape
                    
                    print(f"[Trainer] Dataset: {T} samples, {F} features (reversed)")
                    
                    # Check if enough for sliding windows
                    if T <= SEQ_LEN:
                        print(f"[Trainer] ✗ Not enough samples (T={T}, need T > {SEQ_LEN})")
                        last_train_time = current_time
                        continue
                    
                    # Create sliding windows
                    X, Y = make_sliding_windows(dataset, SEQ_LEN)
                    N = X.shape[0]
                    print(f"[Trainer] Created {N} windows (seq_len={SEQ_LEN})")
                    
                    # CORRECTED TEMPORAL SPLIT (newest → oldest ordering)
                    # After reversing: index 0 = NEWEST, index N-1 = OLDEST
                    # Validate on newest 25% (low indices), train on remaining 75%
                    val_size = int(0.25 * N)     # Validation on newest 25%
                    train_size = N - val_size    # Training on remaining 75%
                    
                    print(f"[Trainer] Temporal split (using all {N} windows):")
                    print(f"  Validate: {val_size} windows (indices 0-{val_size-1}, NEWEST data)")
                    print(f"  Train: {train_size} windows (indices {val_size}-{N-1}, older but recent data)")
                    
                    # Ensure validation set has enough samples for at least 1 batch
                    # With drop_last=True, we need at least BATCH_SIZE samples
                    min_val_samples = BATCH_SIZE
                    if val_size < min_val_samples:
                        print(f"[Trainer] ⚠ Warning: Validation set too small ({val_size} < {min_val_samples})")
                        # Adjust to ensure minimum validation size
                        if N >= min_val_samples + 100:  # Need buffer for training too
                            val_size = min_val_samples
                            train_size = N - val_size
                            print(f"[Trainer] Adjusted - Validate: {val_size}, Train: {train_size}")
                        else:
                            print(f"[Trainer] ✗ Not enough windows ({N}) for batch size {BATCH_SIZE}, skipping training")
                            last_train_time = current_time
                            continue
                    
                    # CRITICAL FIX: Use GLOBAL SCALERS fitted on first training batch
                    # This ensures consistent scaling bounds across all training cycles
                    # Prevents catastrophic failures from distribution shift
                    X_val_portion = X[:val_size]           # Newest data for validation
                    Y_val_portion = Y[:val_size]
                    X_train_portion = X[val_size:]         # Older (but recent) data for training
                    Y_train_portion = Y[val_size:]
                    
                    # Use GLOBAL scalers (fitted once, reused forever)
                    with model_lock:
                        if current_scaler_x is None:
                            # FIRST TRAINING: Fit new scalers on ALL data
                            print(f"[Trainer] FIRST TRAINING: Fitting NEW global scalers on all {N} windows")
                            X_all = np.concatenate([X_train_portion, X_val_portion], axis=0)
                            Y_all = np.concatenate([Y_train_portion, Y_val_portion], axis=0)
                            
                            X_scaled, Y_scaled, scaler_x, scaler_y = scale_data(
                                X_all, Y_all,
                                scaler_x=None, 
                                scaler_y=None,
                                fit_on_train_only=False
                            )
                        else:
                            # SUBSEQUENT TRAININGS: Reuse existing global scalers
                            print(f"[Trainer] REUSING global scalers from training #1")
                            scaler_x = current_scaler_x
                            scaler_y = current_scaler_y
                            
                            # Transform training and validation separately
                            X_train_scaled, Y_train_scaled, _, _ = scale_data(
                                X_train_portion, Y_train_portion,
                                scaler_x=scaler_x, scaler_y=scaler_y
                            )
                            X_val_scaled, Y_val_scaled, _, _ = scale_data(
                                X_val_portion, Y_val_portion,
                                scaler_x=scaler_x, scaler_y=scaler_y
                            )
                            
                            # Combine scaled data
                            X_scaled = torch.cat([X_train_scaled, X_val_scaled], dim=0)
                            Y_scaled = torch.cat([Y_train_scaled, Y_val_scaled], dim=0)
                    
                    # Split scaled data back into train/validation
                    X_train = X_scaled[:train_size]
                    X_test = X_scaled[train_size:]
                    Y_train = Y_scaled[:train_size]
                    Y_test = Y_scaled[train_size:]
                    
                    X_train = X_train.to(device)
                    X_test = X_test.to(device)
                    Y_train = Y_train.to(device)
                    Y_test = Y_test.to(device)
                    
                    dl_train = DataLoader(
                        TensorDataset(X_train, Y_train),
                        batch_size=BATCH_SIZE, shuffle=False, drop_last=True
                    )

                    dl_test = DataLoader(
                        TensorDataset(X_test, Y_test),
                        batch_size=BATCH_SIZE, shuffle=False, drop_last=True
                    )
                    
                    print(f"[Trainer] Train: {len(X_train)}, Test: {len(X_test)}")
                    
                    # Create new h-net with transfer learning (warm start)
                    hnet = MLP([OUTPUT_DIM, 16, 32, 1], nn.ReLU()).to(device)
                    
                    # Load previous h-net weights if available (transfer learning / warm start)
                    with model_lock:
                        if current_hnet is not None and hnet_version > 0:
                            print(f"[Trainer] Warm start: Loading weights from previous h-net version {hnet_version}")
                            hnet.load_state_dict(current_hnet.state_dict())
                        else:
                            print(f"[Trainer] Fresh start: Initializing h-net with random weights")
                    
                    criterion = nn.MSELoss()
                    
                    # Train h-net
                    train_loss, test_loss = train_hnet(
                        frozen_model, hnet, OUTPUT_DIM, dl_train, dl_test, 
                        EPOCHS, LR, criterion, TARGET_KPI_INDEX
                    )
                    
                    # Calculate training metrics (inverse scaled to original dBm)
                    p_train, gt_train = predict_nestedh(frozen_model, hnet, dl_train, OUTPUT_DIM, TARGET_KPI_INDEX)
                    p_train, gt_train = inverse_scale_data(p_train, gt_train, scaler_y, TARGET_KPI_INDEX)
                    
                    print(f"[Trainer] Calculating training metrics...")
                    training_mse, training_rmse, training_r2, training_corr = calc_metric(gt_train, p_train)
                    print(f"[Trainer] Training Metrics: MSE={training_mse:.6f}, RMSE={training_rmse:.6f}, R2={training_r2:.6f}, Corr={training_corr:.6f}")
                    
                    # Calculate VALIDATION MSE on hold-out test set (newer data)
                    print(f"[Trainer] Calculating validation metrics on hold-out set...")
                    p_val, gt_val = predict_nestedh(frozen_model, hnet, dl_test, OUTPUT_DIM, TARGET_KPI_INDEX)
                    p_val, gt_val = inverse_scale_data(p_val, gt_val, scaler_y, TARGET_KPI_INDEX)
                    
                    validation_mse, validation_rmse, validation_r2, validation_corr = calc_metric(gt_val, p_val)
                    print(f"[Trainer] Validation Metrics: MSE={validation_mse:.6f}, RMSE={validation_rmse:.6f}, R2={validation_r2:.6f}, Corr={validation_corr:.6f}")
                    
                    train_time = time.time() - train_start
                    
                    # Get PREVIOUS MODEL LIVE MSE: Performance of old model version on live predictions during inter-training interval
                    # This measures how well the previous model performed in production before being replaced
                    with prediction_errors_lock:
                        if len(prediction_errors) > 0:
                            recent_errors = list(prediction_errors)
                            previous_model_live_mse = np.mean([e**2 for e in recent_errors])
                            num_previous_model_samples = len(recent_errors)
                        else:
                            previous_model_live_mse = None
                            num_previous_model_samples = 0
                        
                        prediction_errors.clear()  # Clear for next training cycle
                    
                    # Update global model (thread-safe)
                    with model_lock:
                        current_hnet = hnet
                        current_scaler_x = scaler_x
                        current_scaler_y = scaler_y
                        hnet_version += 1
                        last_training_max_timestamp = max_timestamp_in_batch  # Update timestamp boundary
                        
                        # Update stats
                        with stats_lock:
                            stats['total_trainings'] += 1
                            stats['last_training_time'] = datetime.now().strftime('%H:%M:%S')
                            stats['last_training_mse'] = training_mse
                            stats['last_training_rmse'] = training_rmse
                            stats['last_training_r2'] = training_r2
                            stats['last_training_corr'] = training_corr
                            stats['last_validation_mse'] = validation_mse
                            stats['last_previous_model_live_mse'] = previous_model_live_mse
                            stats['mse_history'].append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'version': hnet_version,
                                'training_mse': training_mse,
                                'training_rmse': training_rmse,
                                'training_r2': training_r2,
                                'training_corr': training_corr,
                                'validation_mse': validation_mse,
                                'validation_rmse': validation_rmse,
                                'validation_r2': validation_r2,
                                'validation_corr': validation_corr,
                                'previous_model_live_mse': previous_model_live_mse,
                                'num_previous_model_samples': num_previous_model_samples
                            })
                    
                    prev_model_mse_str = f"{previous_model_live_mse:.6f} (n={num_previous_model_samples})" if previous_model_live_mse else "N/A"
                    print(f"[Trainer] ✓ Training complete in {train_time:.2f}s | Version: {hnet_version}")
                    print(f"          Training: MSE={training_mse:.6f}, RMSE={training_rmse:.6f}, R2={training_r2:.6f}, Corr={training_corr:.6f}")
                    print(f"          Validation: MSE={validation_mse:.6f}, RMSE={validation_rmse:.6f}, R2={validation_r2:.6f}, Corr={validation_corr:.6f}")
                    print(f"          Previous Model Live MSE: {prev_model_mse_str}")
                    
                    # Save h-net
                    hnet_path = os.path.join(OUTPUT_DIR, f"hnet_v{hnet_version}.pt")
                    torch.save(hnet.state_dict(), hnet_path)
                    
                    # Mark first training as complete
                    first_training_done = True
                    
                except Exception as e:
                    print(f"[Trainer] ✗ Training failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Update last_train_time after successful training
                last_train_time = current_time
            else:
                # Buffer not full yet - DO NOT update last_train_time
                # This ensures the 45-second interval is preserved for subsequent trainings
                if not first_training_done:
                    # First training: wait for buffer to fill
                    pass  # Keep waiting silently
                else:
                    # Subsequent trainings: shouldn't happen (buffer should always have data)
                    # If this happens, we keep checking until buffer is full again
                    print(f"[Trainer] ⚠ Insufficient samples ({buffer_size}/{MIN_SAMPLES_FOR_TRAINING}), waiting...")
                # DO NOT reset last_train_time here - let the interval continue
        
        time.sleep(1)
    
    print(f"[Trainer] Thread stopped")


####################
#### THREAD 3: TEST/PREDICTION
####################

def test_thread():
    """
    Makes predictions on incoming data using frozen ESN (loaded trained h_score) + current h-net.
    Reads from training_buffer instead of kpi_queue to ensure all data is seen.

    Note: These samples are NEW/UNSEEN by the model - they haven't been
    used for training yet. We're testing on the most recent live data.
    """
    print(f"[Test] Thread started")
    
    local_hnet = None
    local_scaler_x = None
    local_scaler_y = None
    local_version = 0
    local_training_max_timestamp = 0  # Track training data boundary for current model
    
    # Track last processed timestamp to avoid duplicate predictions
    last_processed_timestamp = 0
    
    # Buffer to collect samples for windowing
    sample_buffer = deque(maxlen=SEQ_LEN + 100)
    
    while not shutdown_event.is_set():
        try:
            # Check if there's a newer model version
            with model_lock:
                if hnet_version > local_version:
                    local_hnet = current_hnet
                    local_scaler_x = current_scaler_x
                    local_scaler_y = current_scaler_y
                    local_version = hnet_version
                    local_training_max_timestamp = last_training_max_timestamp  # Get training boundary
                    print(f"[Test] Loaded h-net version {local_version} (trained on data up to t={local_training_max_timestamp:.0f})")
            
            # Read from training buffer (snapshot to avoid holding lock too long)
            with training_buffer_lock:
                if len(training_buffer) > 0:
                    # Get recent samples that we haven't processed yet
                    buffer_snapshot = list(training_buffer)
                else:
                    buffer_snapshot = []
            
            # Add new samples to sample_buffer
            # CRITICAL: Only use samples NEWER than training data to avoid data leakage
            for sample in buffer_snapshot:
                timestamp = sample.get('timestamp', 0)
                # Must satisfy TWO conditions:
                # 1. Not predicted on yet (timestamp > last_processed_timestamp)
                # 2. Not used in training current model (timestamp > local_training_max_timestamp)
                if timestamp > last_processed_timestamp and timestamp > local_training_max_timestamp:
                    sample_buffer.append(sample)
                    last_processed_timestamp = timestamp
            
            # Make prediction if model is available and we have enough samples
            if local_hnet is not None and len(sample_buffer) >= SEQ_LEN + 1:
                try:
                    # Convert buffer to array
                    df_buffer = pd.DataFrame(list(sample_buffer))
                    df_buffer = df_buffer.dropna()
                    
                    if len(df_buffer) < SEQ_LEN + 1:
                        continue
                    
                    # Sort by timestamp (oldest → newest), then reverse to (newest → oldest)
                    # This matches xapp_load_hscore.py exactly
                    df_buffer = df_buffer.sort_values('timestamp').reset_index(drop=True)
                    
                    # Reverse (newest → oldest)
                    dataset = df_buffer[FEATURE_NAMES].values.astype(np.float32)
                    dataset = dataset[::-1]
                    
                    # True one-step-ahead: Y is newest (target), X is older samples (input)
                    Y_actual_unscaled = dataset[0, TARGET_KPI_INDEX]  # Target: most recent sample (t100)
                    X_sample = dataset[1:SEQ_LEN+1, :].reshape(1, SEQ_LEN, -1)  # Input: samples [t99 to t90]
                    
                    # Scale X
                    seq_len, input_dim = X_sample.shape[1], X_sample.shape[2]
                    Xf = X_sample.reshape(-1, input_dim)
                    Xf = local_scaler_x.transform(Xf)
                    X_scaled = Xf.reshape(1, seq_len, input_dim)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                    
                    # Predict (output is SCALED)
                    with torch.no_grad():
                        f_out = frozen_model.get_f(X_tensor)
                        f_out = f_out[:, :OUTPUT_DIM]
                        h_out = local_hnet(f_out)  # Shape: (1, 1), SCALED prediction
                    
                    # FIX: Inverse scale ONLY the prediction (not Y_actual, which is already unscaled)
                    # Create dummy array with all features for proper inverse transform
                    Y_pred_scaled = h_out.cpu().numpy()  # (1, 1)
                    dummy = np.zeros((1, len(FEATURE_NAMES)))
                    dummy[:, TARGET_KPI_INDEX] = Y_pred_scaled[0, 0]
                    
                    # Inverse transform to get unscaled prediction (in original dBm)
                    Y_pred_unscaled_full = local_scaler_y.inverse_transform(dummy)
                    Y_pred_unscaled = Y_pred_unscaled_full[0, TARGET_KPI_INDEX]
                    
                    # Calculate error (both values in original dBm scale)
                    error = Y_actual_unscaled - Y_pred_unscaled
                    squared_error = error ** 2
                    
                    # Store prediction error for next training cycle
                    with prediction_errors_lock:
                        prediction_errors.append(error)
                    
                    # Store ALL prediction errors, predictions, and ground truth for final metrics
                    global all_prediction_errors, all_predictions, all_ground_truth
                    with all_prediction_errors_lock:
                        all_prediction_errors.append(error)
                        all_predictions.append(Y_pred_unscaled)
                        all_ground_truth.append(Y_actual_unscaled)
                    
                    # Update live/current MSE with Exponential Moving Average
                    global live_mse_ema
                    with live_mse_lock:
                        if live_mse_ema == 0.0:
                            live_mse_ema = squared_error  # Initialize with first error
                        else:
                            live_mse_ema = (1 - live_mse_ema_alpha) * live_mse_ema + live_mse_ema_alpha * squared_error
                    
                    # Update stats
                    with stats_lock:
                        stats['total_predictions'] += 1
                        stats['last_prediction_time'] = datetime.now().strftime('%H:%M:%S')
                    
                    # Print every 100th predictions
                    if stats['total_predictions'] % 100 == 0:
                        with live_mse_lock:
                            current_live_mse = live_mse_ema
                        print(f"[Test] Prediction #{stats['total_predictions']} | "
                              f"Actual: {Y_actual_unscaled:.2f} dBm | Predicted: {Y_pred_unscaled:.2f} dBm | "
                              f"Error: {error:.2f} dB | Live MSE (EMA): {current_live_mse:.6f} | H-net v{local_version}")
                
                except Exception as e:
                    print(f"[Test] ✗ Prediction failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.01)  # Check buffer every 10ms
        
        except Exception as e:
            print(f"[Test] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    print(f"[Test] Thread stopped")


####################
#### MONITORING THREAD
####################

def monitoring_thread():
    """
    Prints statistics every 10 seconds
    """
    print(f"[Monitor] Thread started")
    
    while not shutdown_event.is_set():
        time.sleep(10)
        
        # Get current live MSE (EMA)
        with live_mse_lock:
            current_live_mse_ema = live_mse_ema
        
        # Get number of predictions since last training
        with prediction_errors_lock:
            num_predictions_since_train = len(prediction_errors)
        
        with stats_lock:
            print(f"\n{'='*80}")
            print(f"[Monitor] Statistics:")
            print(f"  KPIs Received:        {stats['total_kpis_received']}")
            print(f"  Predictions Made:     {stats['total_predictions']}")
            print(f"  Trainings Completed:  {stats['total_trainings']}")
            print(f"  Last Training:        {stats['last_training_time'] or 'Not yet'}")
            print(f"  Last Prediction:      {stats['last_prediction_time'] or 'Not yet'}")
            print(f"  Training Buffer Size: {len(training_buffer)}/{BUFFER_SIZE}")
            print(f"  Current H-net Version: {hnet_version}")
            print(f"\n  Performance Metrics:")
            train_mse_str = f"{stats['last_training_mse']:.6f}" if stats['last_training_mse'] is not None else 'N/A'
            train_rmse_str = f"{stats['last_training_rmse']:.6f}" if stats['last_training_rmse'] is not None else 'N/A'
            train_r2_str = f"{stats['last_training_r2']:.6f}" if stats['last_training_r2'] is not None else 'N/A'
            train_corr_str = f"{stats['last_training_corr']:.6f}" if stats['last_training_corr'] is not None else 'N/A'
            
            val_mse_str = f"{stats['last_validation_mse']:.6f}" if stats['last_validation_mse'] is not None else 'N/A'
            
            if current_live_mse_ema > 0:
                live_mse_str = f"{current_live_mse_ema:.6f} (EMA, n={num_predictions_since_train} since retrain)"
            else:
                live_mse_str = 'N/A (no predictions yet)'
            
            print(f"    Training MSE:       {train_mse_str}")
            print(f"    Training RMSE:      {train_rmse_str}")
            print(f"    Training R2:        {train_r2_str}")
            print(f"    Training Corr:      {train_corr_str}")
            print(f"    Validation MSE:     {val_mse_str}  (hold-out set, newer data)")
            print(f"    Current Prediction MSE: {live_mse_str}")
            
            # Show MSE history
            if len(stats['mse_history']) > 0:
                print(f"\n  Training History (last {min(5, len(stats['mse_history']))} trainings):")
                print(f"    {'Time':<10} {'Ver':<5} {'Train MSE':<10} {'Train R2':<10} {'Val MSE':<10} {'Val R2':<10} {'Prev Model Live MSE (n=samples)':<30}")
                print(f"    {'-'*95}")
                for record in stats['mse_history'][-5:]:
                    val_mse = record.get('validation_mse', None)
                    val_r2 = record.get('validation_r2', None)
                    prev_model_live_mse = record.get('previous_model_live_mse', None)
                    num_prev_model = record.get('num_previous_model_samples', 0)
                    
                    val_mse_str = f"{val_mse:.4f}" if val_mse is not None else "N/A"
                    val_r2_str = f"{val_r2:.4f}" if val_r2 is not None else "N/A"
                    prev_model_mse_str = f"{prev_model_live_mse:.4f} (n={num_prev_model})" if prev_model_live_mse is not None else "N/A"
                    
                    print(f"    {record['time']:<10} {record['version']:<5} "
                          f"{record['training_mse']:<10.4f} {record['training_r2']:<10.4f} "
                          f"{val_mse_str:<10} {val_r2_str:<10} {prev_model_mse_str:<30}")
            
            print(f"{'='*80}\n")
    
    print(f"[Monitor] Thread stopped")


####################
#### MAIN
####################

def main():
    global frozen_model
    
    print(f"\n{'='*80}")
    print(f"ONLINE H-SCORE xApp - Real-Time H-net Fine-Tuning")
    print(f"{'='*80}")
    print(f"Runtime: {RUNTIME} seconds")
    print(f"Training Interval: {TRAINING_INTERVAL} seconds")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Target KPI: {FEATURE_NAMES[TARGET_KPI_INDEX]}")
    print(f"{'='*80}\n")

    # Load pre-trained H-Score model (frozen ESN)
    try:
        frozen_model = load_hscore_model()
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        return
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    ####################
    ####  GENERAL 
    ####################
    
    # Initialize FlexRIC
    ric.init()
    
    conn = ric.conn_e2_nodes()
    assert(len(conn) > 0)
    print(f"Connected to {len(conn)} E2 Node(s)")

    ####################
    #### MAC INDICATION
    ####################
    
    # Start MAC reporting
    mac_hndlr = []
    for i in range(len(conn)):
        mac_cb = MACCallback()
        hndlr = ric.report_mac_sm(conn[i].id, MAC_INTERVAL, mac_cb)
        mac_hndlr.append(hndlr)
    print(f"✓ Started MAC metrics reporting")

    ####################
    #### RLC INDICATION
    ####################
    
    # Start RLC reporting
    rlc_hndlr = []
    for i in range(len(conn)):
        rlc_cb = RLCCallback()
        hndlr = ric.report_rlc_sm(conn[i].id, RLC_INTERVAL, rlc_cb)
        rlc_hndlr.append(hndlr)
    print(f"✓ Started RLC metrics reporting")

    ####################
    #### START THREADS
    ####################
    
    # Start threads
    print(f"\nStarting worker threads...")
    
    collector_t = threading.Thread(target=kpi_collector_thread, daemon=True)
    trainer_t = threading.Thread(target=trainer_thread, daemon=True)
    test_t = threading.Thread(target=test_thread, daemon=True)
    monitor_t = threading.Thread(target=monitoring_thread, daemon=True)
    
    collector_t.start()
    trainer_t.start()
    test_t.start()
    monitor_t.start()
    
    print(f"✓ All threads started\n")
    print(f"Running for {RUNTIME} seconds...\n")
    
    # Run for specified duration
    start_time = time.time()
    try:
        while time.time() - start_time < RUNTIME:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                print(f"[Main] Running... {elapsed}/{RUNTIME} seconds")
    except KeyboardInterrupt:
        print(f"\n[Main] Interrupted by user")
    
    # Shutdown
    print(f"\n[Main] Shutting down...")
    shutdown_event.set()
    
    # Wait for threads to finish
    collector_t.join(timeout=2)
    trainer_t.join(timeout=2)
    test_t.join(timeout=2)
    monitor_t.join(timeout=2)
    
    # Stop reporting
    for hndlr in mac_hndlr:
        ric.rm_report_mac_sm(hndlr)
    for hndlr in rlc_hndlr:
        ric.rm_report_rlc_sm(hndlr)
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    
    # Get final LIVE MSE (EMA)
    with live_mse_lock:
        final_live_mse_ema = live_mse_ema
    
    # Get number of predictions since last training
    with prediction_errors_lock:
        final_test_samples = len(prediction_errors)
    
    # Calculate final overall metrics from all real-time predictions
    with all_prediction_errors_lock:
        if len(all_prediction_errors) > 0:
            overall_mse = np.mean([e**2 for e in all_prediction_errors])
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean([abs(e) for e in all_prediction_errors])
            total_realtime_predictions = len(all_prediction_errors)
            
            # Calculate R2 and Correlation
            y_true = np.array(all_ground_truth)
            y_pred = np.array(all_predictions)
            overall_r2 = r2_score(y_true, y_pred)
            
            # Check for zero variance before calculating correlation
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                overall_corr = 0.0
            else:
                overall_corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            overall_mse = None
            overall_rmse = None
            overall_mae = None
            overall_r2 = None
            overall_corr = None
            total_realtime_predictions = 0
    
    with stats_lock:
        print(f"Total KPIs Received:       {stats['total_kpis_received']}")
        print(f"Total Predictions Made:    {stats['total_predictions']}")
        print(f"Total Trainings:           {stats['total_trainings']}")
        print(f"Final H-net Version:       {hnet_version}")
        print(f"Training Buffer Size:      {len(training_buffer)}/{BUFFER_SIZE}")
        print(f"\nFinal Performance:")
        if stats['last_training_mse']:
            print(f"  Training MSE (v{hnet_version}):      {stats['last_training_mse']:.6f}")
            print(f"  Training RMSE (v{hnet_version}):     {stats['last_training_rmse']:.6f}")
            print(f"  Training R2 (v{hnet_version}):       {stats['last_training_r2']:.6f}")
            print(f"  Training Corr (v{hnet_version}):     {stats['last_training_corr']:.6f}")
        if stats['last_validation_mse']:
            print(f"  Validation MSE (v{hnet_version}):    {stats['last_validation_mse']:.6f} (hold-out set)")
        if final_live_mse_ema > 0:
            print(f"  Current MSE (v{hnet_version} EMA):   {final_live_mse_ema:.6f} (n={final_test_samples} since retrain)")
        
        # Show full metrics history
        if len(stats['mse_history']) > 0:
            print(f"\nMetrics Evolution During Training:")
            print(f"  {'Time':<10} {'Ver':<5} {'Train MSE':<10} {'Train R2':<10} {'Val MSE':<10} {'Val R2':<10} {'Prev Model Live MSE (n=samples)':<30}")
            print(f"  {'-'*95}")
            for record in stats['mse_history']:
                val_mse = record.get('validation_mse', None)
                val_r2 = record.get('validation_r2', None)
                prev_model_live_mse = record.get('previous_model_live_mse', None)
                num_prev_model = record.get('num_previous_model_samples', 0)
                
                val_mse_str = f"{val_mse:.4f}" if val_mse is not None else "N/A"
                val_r2_str = f"{val_r2:.4f}" if val_r2 is not None else "N/A"
                prev_model_mse_str = f"{prev_model_live_mse:.4f} (n={num_prev_model})" if prev_model_live_mse is not None else "N/A"
                
                print(f"  {record['time']:<10} {record['version']:<5} "
                      f"{record['training_mse']:<10.4f} {record['training_r2']:<10.4f} "
                      f"{val_mse_str:<10} {val_r2_str:<10} {prev_model_mse_str:<30}")
    print(f"{'='*80}")
    
    # Print average real-time metrics after metrics evolution table
    if overall_mse is not None:
        print(f"\n  *** AVERAGE REAL-TIME MSE (all {total_realtime_predictions} predictions): {overall_mse:.6f} ***")
        print(f"  *** AVERAGE REAL-TIME RMSE: {overall_rmse:.6f} ***")
        print(f"  *** AVERAGE REAL-TIME MAE: {overall_mae:.6f} ***")
        print(f"  *** AVERAGE REAL-TIME R2: {overall_r2:.6f} ***")
        print(f"  *** AVERAGE REAL-TIME CORRELATION: {overall_corr:.6f} ***")
    print(f"{'='*80}\n")
    
    # Save final training buffer
    if len(training_buffer) > 0:
        print(f"Saving final training buffer ({len(training_buffer)} samples)...")
        df_final = pd.DataFrame(list(training_buffer))
        csv_path = os.path.join(OUTPUT_DIR, "online_hscore_final_buffer.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"✓ Saved to: {csv_path}")
    
    print(f"\n✓ xApp terminated successfully\n")


if __name__ == "__main__":
    main()
