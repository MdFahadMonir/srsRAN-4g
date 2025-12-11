#!/usr/bin/env python3
"""
H-Score Offline Training xApp
Collects KPIs for RUN_TIME seconds, then performs offline H-Score training.
"""

import xapp_sdk as ric
import time
import os
import json
import queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, max_error


# ==========================================================
#                   FIXED CONFIGURATION
# ==========================================================
RUN_TIME        = 120          # seconds
SEQ_LEN         = 10
SCALER_TYPE     = "minmax"     # one of: "none", "standard", "minmax"
BATCH_SIZE      = 128
EPOCHS          = 30 
LR              = 1e-3
HS_TRAIN_FRAC   = 0.6          # H-score train fraction
PS_RESERVE_FRAC = 0.2          # reserved for future prediction split (create only)
RESERVOIR_SIZE  = 32
SPECTRAL_RADIUS = 0.9
SPARSITY        = 0.8
LEAKY           = 0.2
OUTPUT_DIM      = 8            # Output dimension for f-net and g-net
ML_OUT_DIR      = "/home/fahad/srsRAN_4g/ml_data"
SEED            = 2
DEVICE_MODE     = "auto"        # "auto" -> cuda if available else cpu
FEATURE_NAMES   = [
    "rnti","phr","dl_tbs","ul_tbs","dl_aggr_prb",
    "wb_cqi","pusch_snr","pucch_snr","ul_rssi",
    "dl_bler","ul_bler","dl_mcs","ul_mcs",
]

MAC_INTERVAL = ric.Interval_ms_10
RLC_INTERVAL = ric.Interval_ms_10

# ==========================================================
#                   SEED INITIALIZATION
# ==========================================================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device


#device = torch.device("cpu") 


if DEVICE_MODE == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif DEVICE_MODE == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print(f"Using device: {device}")

# Ensure output directory exists
os.makedirs(ML_OUT_DIR, exist_ok=True)

# ==========================================================
#                     DATA COLLECTION QUEUES
# ==========================================================
mac_queue = queue.Queue()
rlc_queue = queue.Queue()

# ==========================================================
#                     H-SCORE CORE MODULES
# ==========================================================

class ESN(nn.Module):
    def __init__(self, input_dim, window_len, reservoir_size, spectral_radius, sparsity, leaky, output_dim):
        super(ESN, self).__init__()
        self.window_len = window_len
        self.reservoir_size = reservoir_size
        self.leaky = leaky  # Fixed leaky parameter
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
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        x_skip = x.reshape(batch_size, -1)  # Store original input for skip connection

        reservoir_state = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        states_stack = []
        
        for t in range(seq_len-1, -1, -1):
            u_t = x[:, t, :]  # Current input at time t
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


# ==========================================================
#                   TRAINING FUNCTION
# ==========================================================
def train_hnet(model, hnet, d_des, dl_train, dl_test, epochs, lr, weight_decay, criterion_h, pi, print_every=5):
    model.eval()
    h_opt = optim.Adam(hnet.parameters(), lr=lr)#, weight_decay=weight_decay)
    h_train_losses = []
    test_losses = []

    min_val_loss = float('inf')
    epochs_no_improve = 0
    hnet_patience = 10 # Patience for hnet training

    print(f"\nStarting h-net training for {epochs} epochs...")
    print(f"  Target KPI: {FEATURE_NAMES[pi]} (index {pi})")
    print(f"  Optimizer: Adam (lr={lr})")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        hnet.train()
        h_train_loss = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb[:, [pi]]
            h_opt.zero_grad()
            with torch.no_grad():
                f_out = model.get_f(xb)
            f_out = f_out[:,:d_des]
            h_out = hnet(f_out)
            loss = criterion_h(h_out, yb)
            loss.backward()
            h_opt.step()
            h_train_loss += loss.item()
        h_train_loss /= len(dl_train)
        h_train_losses.append(h_train_loss)

        hnet.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb[:, [pi]]
                f_out = model.get_f(xb)
                f_out = f_out[:,:d_des]
                h_out = hnet(f_out)
                loss = criterion_h(h_out, yb)
                test_loss += loss.item()
        test_loss /= len(dl_test)
        test_losses.append(test_loss)

        if epoch % print_every == 0 or epoch == epochs:
            print(f"[Epoch {epoch:02d}/{epochs}] Train Loss: {h_train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < min_val_loss:
            min_val_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hnet_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {hnet_patience} epochs)")
                break
    
    print("-" * 80)
    print("✓ h-net training complete")
    
    return h_train_losses, test_losses

def predict_nestedh(model, hnet, dl_test, d_des=None, pi=None):
    model.eval()
    hnet.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            yb = yb[:, pi].reshape(-1, 1)
            f_out = model.get_f(xb)
            f_out = f_out[:,:d_des]
            h_out = hnet(f_out)
            predictions.append(h_out.cpu().numpy())
            ground_truth.append(yb.cpu().numpy())

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    return predictions, ground_truth


# ==========================================================
#                     MAC CALLBACK
# ==========================================================
class MACCallback(ric.mac_cb):
    def __init__(self):
        ric.mac_cb.__init__(self)
    
    def handle(self, ind):
        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            stats = ind.ue_stats[0]
            
            mac_data = {
                'timestamp': t_mac,
                'rnti': stats.rnti,
                'phr': stats.phr,
                'dl_tbs': stats.dl_aggr_tbs,
                'ul_tbs': stats.ul_aggr_tbs,
                'dl_aggr_prb': stats.dl_aggr_prb,
                'wb_cqi': stats.wb_cqi,
                'pusch_snr': stats.pusch_snr,
                'pucch_snr': stats.pucch_snr,
                'ul_rssi': stats.ul_rssi,
                'dl_bler': stats.dl_bler,
                'ul_bler': stats.ul_bler,
                'dl_mcs': stats.dl_mcs1,
                'ul_mcs': stats.ul_mcs1
            }
            mac_queue.put(mac_data)


# ==========================================================
#                     RLC CALLBACK
# ==========================================================
class RLCCallback(ric.rlc_cb):
    def __init__(self):
        ric.rlc_cb.__init__(self)
    
    def handle(self, ind):
        if len(ind.rb_stats) > 0:
            stats = ind.rb_stats[0]
            rlc_data = {
                'timestamp': ind.tstamp / 1.0,
                'rnti': stats.rnti,
                'rbid': stats.rbid,
                'mode': stats.mode
            }
            rlc_queue.put(rlc_data)


# ==========================================================
#                SLIDING WINDOW CREATION
# ==========================================================
def make_sliding_windows(dataset, seq_len):
    """
    Given dataset shape (T, F) with newest → oldest ordering, return:
        X: (N, seq_len, F) - each window contains seq_len consecutive timesteps
        Y: (N, F) - target is the first row of the next window, i.e. Y[i] = X[i+1, 0, :]
    where N = T - seq_len
    
    Note: For the last window (i = N-1), Y is the row immediately after its window ends.
    """
    T, F = dataset.shape
    N = T - seq_len

    if N <= 0:
        raise ValueError(f"Not enough samples: T={T}, seq_len={seq_len}, need T > seq_len")

    X = np.zeros((N, seq_len, F), dtype=np.float32)
    Y = np.zeros((N, F), dtype=np.float32)

    for i in range(N):
        # X[i] contains rows [i, i+1, ..., i+seq_len-1]
        X[i] = dataset[i : i + seq_len, :]
        
        # Y[i] is the first row of the next window (i.e., X[i+1, 0, :])
        # which is dataset[i+1] when i < N-1, or dataset[i+seq_len] when i == N-1
        if i < N - 1:
            Y[i] = dataset[i + 1, :]  # First row of next window
        else:
            Y[i] = dataset[i + seq_len, :]  # Row after last window ends

    return X, Y


# ==========================================================
#                   DATA SCALING
# ==========================================================
def scale_data(X, Y, scaler_type="minmax"):
    seq_len = X.shape[1]
    input_dim = X.shape[2]

    if scaler_type == "standard":
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
    elif scaler_type == "minmax":
        x_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), None, None

    # flatten X to scale per feature
    Xf = X.reshape(-1, input_dim)
    Xf = x_scaler.fit_transform(Xf)
    X = Xf.reshape(-1, seq_len, input_dim)

    Y = y_scaler.fit_transform(Y)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        x_scaler, y_scaler
    )



def calc_metric(ground_truth, predictions):
    mse = mean_squared_error(ground_truth[:, 0], predictions[:, 0])
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth[:, 0], predictions[:, 0])
    print(f"  MSE: {mse:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")
    # Check if either array has zero variance
    if np.std(ground_truth[:, 0]) == 0 or np.std(predictions[:, 0]) == 0:
        print(f"⚠ Warning: Zero variance detected!")
        print(f"  Ground truth std: {np.std(ground_truth[:, 0]):.6f}")
        print(f"  Predictions std:  {np.std(predictions[:, 0]):.6f}")
        print(f"  Predictions unique values: {np.unique(predictions[:, 0])[:10]}")
        corr = 0.0  # or np.nan
    else:
        corr = np.corrcoef(ground_truth[:, 0], predictions[:, 0])[0, 1]
        print(f"  Calculated correlation: {corr:.6f}")
    
    return mse, corr

def to_dataloader(x_train, y_train, x_test, y_test, batch, drop_last=False):
    # train_idx, test_idx, _, _ = train_test_split(range(len(x)), range(len(y)), test_size=split)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=drop_last, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, drop_last=drop_last, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def hscore_pred_split(x, y, hs, ps, batch, drop_last=False):
    x_train_hs, x_test_hs, y_train_hs, y_test_hs = train_test_split(x, y, train_size=hs) # hscore data split
    dl_train_hs, dl_test_hs = to_dataloader(x_train_hs, y_train_hs, x_test_hs, y_test_hs, batch, drop_last)

    x_train_ps, x_test_ps, y_train_ps, y_test_ps = train_test_split(x, y, test_size=ps) # prediction data split
    dl_train_ps, dl_test_ps = to_dataloader(x_train_ps, y_train_ps, x_test_ps, y_test_ps, batch, drop_last)

    return dl_train_hs, dl_test_hs, dl_train_ps, dl_test_ps

def inverse_scale_data(predictions, ground_truth, scaler, column_idx=None):
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

# ==========================================================
#                        MAIN
# ==========================================================
def main():
    print("\n" + "="*80)
    print("H-SCORE OFFLINE TRAINING xApp")
    print("="*80)
    print(f"Configuration:")
    print(f"  Runtime:          {RUN_TIME}s")
    print(f"  Sequence Length:  {SEQ_LEN}")
    print(f"  Scaler:           {SCALER_TYPE}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Learning Rate:    {LR}")
    print(f"  H-Train Fraction: {HS_TRAIN_FRAC}")
    print(f"  PS Reserve:       {PS_RESERVE_FRAC}")
    print(f"  Device:           {device}")
    print(f"  Output Dir:       {ML_OUT_DIR}")
    print("="*80 + "\n")

    # ==========================================================
    #              PART 1: KPI COLLECTION
    # ==========================================================
    print("[Part 1] Starting KPI collection...")
    
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No E2 nodes connected"
    print(f"✓ Connected to {len(conn)} E2 Node(s)")

    mac_hndlr = []
    rlc_hndlr = []

    try:
        # Register MAC callbacks
        for i in range(len(conn)):
            mac_cb = MACCallback()
            hndlr = ric.report_mac_sm(conn[i].id, MAC_INTERVAL, mac_cb)
            mac_hndlr.append(hndlr)
        print(f"✓ Started MAC metrics reporting")

        # Register RLC callbacks
        for i in range(len(conn)):
            rlc_cb = RLCCallback()
            hndlr = ric.report_rlc_sm(conn[i].id, RLC_INTERVAL, rlc_cb)
            rlc_hndlr.append(hndlr)
        print(f"✓ Started RLC metrics reporting")

        # Collect for RUN_TIME seconds
        print(f"\nCollecting KPIs for {RUN_TIME} seconds...")
        start_time = time.time()
        elapsed = 0
        while elapsed < RUN_TIME:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                print(f"  Progress: {elapsed}/{RUN_TIME} seconds ({mac_queue.qsize()} MAC samples)")

        print(f"\n✓ Collection complete: {mac_queue.qsize()} MAC samples collected")

    finally:
        # Stop reporting
        print("\nStopping metric reporting...")
        for hndlr in mac_hndlr:
            ric.rm_report_mac_sm(hndlr)
        for hndlr in rlc_hndlr:
            ric.rm_report_rlc_sm(hndlr)
        print("✓ All reports stopped")

    # Extract MAC data to DataFrame
    print("\nBuilding dataset from collected KPIs...")
    mac_data_list = []
    while not mac_queue.empty():
        mac_data_list.append(mac_queue.get())

    if len(mac_data_list) == 0:
        print("✗ ERROR: No MAC data collected. Exiting.")
        return

    df_mac = pd.DataFrame(mac_data_list)
    df_mac = df_mac.sort_values('timestamp').reset_index(drop=True)
    print(f"  Initial samples: {len(df_mac)}")

    # Drop NaNs
    df_mac = df_mac.dropna()
    print(f"  After dropping NaNs: {len(df_mac)}")

    # Check if we have enough samples
    T = len(df_mac)
    if T <= SEQ_LEN:
        print(f"\n✗ ERROR: Not enough samples after cleaning (T={T}, need T > {SEQ_LEN}). Exiting.")
        return

    # Extract feature matrix
    dataset = df_mac[FEATURE_NAMES].values.astype(np.float32)
    # Reverse so row 0 is newest sample (newest → oldest ordering)
    dataset = dataset[::-1]
    T, F = dataset.shape
    print(f"  Dataset shape: ({T}, {F}) [reversed: newest → oldest]")

    # Save Part 1 outputs
    csv_path = os.path.join(ML_OUT_DIR, "mac_ml_dataset.csv")
    npy_path = os.path.join(ML_OUT_DIR, "dataset.npy")
    names_path = os.path.join(ML_OUT_DIR, "feature_names.txt")

    df_mac.to_csv(csv_path, index=False)
    np.save(npy_path, dataset)
    with open(names_path, 'w') as f:
        for name in FEATURE_NAMES:
            f.write(name + '\n')

    print(f"\n✓ Saved Part 1 outputs:")
    print(f"  CSV:      {csv_path}")
    print(f"  NumPy:    {npy_path}")
    print(f"  Features: {names_path}")

    # ==========================================================
    #           PART 2: CREATE SLIDING WINDOWS
    # ==========================================================
    print(f"\n{'='*80}")
    print("[Part 2] Creating sliding windows (X, Y)...")
    print(f"  Sequence length: {SEQ_LEN}")

    X, Y = make_sliding_windows(dataset, SEQ_LEN)
    N = X.shape[0]
    print(f"  X shape: {X.shape} (N={N}, seq_len={SEQ_LEN}, F={F})")
    print(f"  Y shape: {Y.shape} (N={N}, F={F})")

    # Preview first 3 windows
    print(f"\n  Preview of sliding windows (first 3, newest → oldest):")
    for t in range(min(3, N)):
        if t < N - 1:
            print(f"    Window {t}: rows [{t}:{t+SEQ_LEN-1}] → target = first row of next window (row {t+1})")
        else:
            print(f"    Window {t}: rows [{t}:{t+SEQ_LEN-1}] → target = row after window ends (row {t+SEQ_LEN})")

    # Save X, Y
    x_path = os.path.join(ML_OUT_DIR, f"X_seq{SEQ_LEN}.npy")
    y_path = os.path.join(ML_OUT_DIR, f"Y_seq{SEQ_LEN}.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"\n✓ Saved Part 2 outputs:")
    print(f"  X: {x_path}")
    print(f"  Y: {y_path}")

    # ==========================================================
    #           PART 3: H-SCORE TRAINING
    # ==========================================================
    print(f"\n{'='*80}")
    print("[Part 3] H-Score Training...")

    # Scale data
    print(f"\nScaling data with {SCALER_TYPE} scaler...")
    X_scaled, Y_scaled, x_scaler, y_scaler = scale_data(X, Y, SCALER_TYPE)
    X_scaled = X_scaled.to(device)
    Y_scaled = Y_scaled.to(device)
    print(f"✓ Data scaled and moved to {device}")

    # Split data
    print(f"\nSplitting data:")
    print(f"  H-Score train fraction: {HS_TRAIN_FRAC}")
    print(f"  Prediction reserve:     {PS_RESERVE_FRAC}")

    # H-Score split
    X_train_hs, X_test_hs, Y_train_hs, Y_test_hs = train_test_split(
        X_scaled, Y_scaled, train_size=HS_TRAIN_FRAC, random_state=SEED, shuffle=True
    )

    # Prediction split (reserved, not used now)
    X_train_ps, X_test_ps, Y_train_ps, Y_test_ps = train_test_split(
        X_scaled, Y_scaled, test_size=PS_RESERVE_FRAC, random_state=SEED, shuffle=True
    )

    print(f"  H-Score train: {len(X_train_hs)} samples")
    print(f"  H-Score test:  {len(X_test_hs)} samples")
    print(f"  Prediction reserve: {len(X_test_ps)} samples (saved for future use)")

    # DataLoaders
    dl_train_hs = DataLoader(
        TensorDataset(X_train_hs, Y_train_hs),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    dl_test_hs = DataLoader(
        TensorDataset(X_test_hs, Y_test_hs),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )

    print(f"\n✓ DataLoaders created (batch_size={BATCH_SIZE}, drop_last=True)")

    # Build model
    esn_model_full_path = os.path.join(ML_OUT_DIR, "hscore_model.pt")

    # print(f'Loading H-Score ESN from {esn_model_full_path}')
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

    # Wrap in fg_nn
    model = fg_nn(fnet, gnet).to(device)
    criterion = nn.MSELoss()
    activation = nn.ReLU()
    drop_last = True
    if not os.path.exists(esn_model_full_path):
        print(f"✗ ERROR: Model file not found at {esn_model_full_path}")
        return
    # Load trained weights (only the model's weights, not optimizer)
    model.load_state_dict(torch.load(esn_model_full_path, map_location=device, weights_only=True))
    model.eval()
    pi=8  # ul_rssi is at index 8 in FEATURE_NAMES
    test_times=5
    results = {
        'MSE': {'H-score ESN': {f'dim {i}': {name: 0.0 for name in FEATURE_NAMES} for i in range(1, OUTPUT_DIM + 1)}},
        'Correlation': {'H-score ESN': {f'dim {i}': {name: 0.0 for name in FEATURE_NAMES} for i in range(1, OUTPUT_DIM + 1)}}
    }
    _, _, dl_train_ps_e, dl_test_ps_e = hscore_pred_split(X_scaled, Y_scaled, hs=HS_TRAIN_FRAC, ps=PS_RESERVE_FRAC,batch=BATCH_SIZE, drop_last=drop_last)
    hnet = MLP([OUTPUT_DIM, 16, 32, 1], activation).to(device)
    _, _ = train_hnet(model, hnet, OUTPUT_DIM, dl_train_ps_e, dl_test_ps_e, epochs=EPOCHS, lr=LR, weight_decay=0, criterion_h=criterion, pi=pi)
    #p_e, gt_e = predict_nestedh(model, hnet, dl_test_ps_e, 1, pi)
    p_e, gt_e = predict_nestedh(model, hnet, dl_test_ps_e, d_des=OUTPUT_DIM, pi=pi)
    p_e, gt_e = inverse_scale_data(p_e, gt_e, y_scaler, pi)





    metrics = calc_metric(gt_e, p_e)
    results['MSE']['H-score ESN'][f'dim {OUTPUT_DIM}'][FEATURE_NAMES[pi]] += metrics[0] / test_times
    results['Correlation']['H-score ESN'][f'dim {OUTPUT_DIM}'][FEATURE_NAMES[pi]] += metrics[1] / test_times




if __name__ == "__main__":
    main()
