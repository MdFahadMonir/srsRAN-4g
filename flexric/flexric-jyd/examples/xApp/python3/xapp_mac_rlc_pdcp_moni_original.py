import xapp_sdk as ric
import time
import os
import pdb
import queue
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

####################
#### DATA COLLECTION QUEUES
####################
mac_queue = queue.Queue()
rlc_queue = queue.Queue()
pdcp_queue = queue.Queue()

####################
#### MAC INDICATION CALLBACK
####################
####################
sleep_value = 1  # Short delay between registrations

# Using the standard 10ms interval that we know exists in the SDK
MAC_INTERVAL = ric.Interval_ms_10
RLC_INTERVAL = ric.Interval_ms_10
PDCP_INTERVAL = ric.Interval_ms_10

#  MACCallback class is defined and derived from C++ class mac_cb
class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_mac_ind_msg_t
        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            t_diff = t_now - t_mac
            
            stats = ind.ue_stats[0]
            
            # Store data in queue
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
            
            # Print for real-time monitoring
            print(f' RNTI: {stats.rnti} (0x{stats.rnti:x}) PHR: {stats.phr} DL TBS: {stats.dl_aggr_tbs} UL TBS: {stats.ul_aggr_tbs} DL_PRB: {stats.dl_aggr_prb}')
            print(f' RF Metrics: WB_CQI: {stats.wb_cqi} PUSCH_SNR: {stats.pusch_snr:.1f} dB PUCCH_SNR: {stats.pucch_snr:.1f} dB UL_RSSI: {stats.ul_rssi:.2f} dBfs')
            print(f' Link Quality: DL_BLER: {stats.dl_bler:.6f} UL_BLER: {stats.ul_bler:.6f} DL_MCS: {stats.dl_mcs1} UL_MCS: {stats.ul_mcs1}')

####################
#### RLC INDICATION CALLBACK
####################

class RLCCallback(ric.rlc_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.rlc_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_rlc_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_rlc_ind_msg_t 
        if len(ind.rb_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_rlc = ind.tstamp / 1.0
            t_diff = t_now - t_rlc
            stats = ind.rb_stats[0]

            # Map RLC modes to strings
            rlc_modes = {0: "AM", 1: "UM", 2: "TM"}
            mode_str = rlc_modes.get(stats.mode, "Unknown")
            
            # Store data in queue
            rlc_data = {
                'timestamp': t_rlc,
                'rnti': stats.rnti,
                'rbid': stats.rbid,
                'mode': stats.mode,
                'mode_str': mode_str,
                'txpdu_pkts': stats.txpdu_pkts,
                'txpdu_bytes': stats.txpdu_bytes,
                'rxpdu_pkts': stats.rxpdu_pkts,
                'rxpdu_bytes': stats.rxpdu_bytes,
                'txsdu_pkts': stats.txsdu_pkts,
                'txsdu_bytes': stats.txsdu_bytes,
                'rxsdu_pkts': stats.rxsdu_pkts,
                'rxsdu_bytes': stats.rxsdu_bytes,
                'txbuf_occ_bytes': stats.txbuf_occ_bytes,
                'txbuf_occ_pkts': stats.txbuf_occ_pkts,
                'rxbuf_occ_bytes': stats.rxbuf_occ_bytes,
                'rxbuf_occ_pkts': stats.rxbuf_occ_pkts
            }
            rlc_queue.put(rlc_data)

            # Print for real-time monitoring
            print(f' RLC RNTI: {stats.rnti} (0x{stats.rnti:x}) RB ID: {stats.rbid} Mode: {mode_str}')
            print(f' PDU Stats: TX: {stats.txpdu_bytes/1000:.1f}kB ({stats.txpdu_pkts} pkts) RX: {stats.rxpdu_bytes/1000:.1f}kB ({stats.rxpdu_pkts} pkts)')
            print(f' SDU Stats: TX: {stats.txsdu_bytes/1000:.1f}kB ({stats.txsdu_pkts} pkts) RX: {stats.rxsdu_bytes/1000:.1f}kB ({stats.rxsdu_pkts} pkts)')

####################
#### PDCP INDICATION CALLBACK
####################

class PDCPCallback(ric.pdcp_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.pdcp_cb.__init__(self)
   # Override C++ method: virtual void handle(swig_pdcp_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_pdcp_ind_msg_t
        if len(ind.rb_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_pdcp = ind.tstamp / 1.0
            t_diff = t_now - t_pdcp
            stats = ind.rb_stats[0]
            
            # Store data in queue
            pdcp_data = {
                'timestamp': t_pdcp,
                'rnti': stats.rnti,
                'rbid': stats.rbid
            }
            pdcp_queue.put(pdcp_data)
            
            # Print for real-time monitoring
            print(f' PDCP RNTI: {stats.rnti} (0x{stats.rnti:x}) RB ID: {stats.rbid}')


####################
####  GENERAL 
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

####################
#### MAC INDICATION
####################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, MAC_INTERVAL, mac_cb)
    mac_hndlr.append(hndlr)     
    time.sleep(sleep_value)
    print(f"Started MAC metrics reporting with {MAC_INTERVAL}ms interval")

####################
#### RLC INDICATION
####################

rlc_hndlr = []
for i in range(0, len(conn)):
    rlc_cb = RLCCallback()
    hndlr = ric.report_rlc_sm(conn[i].id, RLC_INTERVAL, rlc_cb)
    rlc_hndlr.append(hndlr) 
    time.sleep(sleep_value)
    print(f"Started RLC metrics reporting with {RLC_INTERVAL}ms interval")

# ####################
# #### PDCP INDICATION
# ####################

# pdcp_hndlr = []
# for i in range(0, len(conn)):
#     pdcp_cb = PDCPCallback()
#     hndlr = ric.report_pdcp_sm(conn[i].id, PDCP_INTERVAL, pdcp_cb)
#     pdcp_hndlr.append(hndlr) 
#     print(f"Started PDCP metrics reporting with {PDCP_INTERVAL}ms interval")
#     time.sleep(sleep_value)

# Run for a fixed duration
runtime = 5  # seconds - change this value to adjust test duration
print(f"xApp will run for {runtime} seconds...")

# Simple sleep for the specified duration
for i in range(runtime):
    time.sleep(1)
    if (i + 1) % 10 == 0:
        print(f"Running... {i + 1}/{runtime} seconds")

### End - Stop reporting

for i in range(0, len(mac_hndlr)):
    ric.rm_report_mac_sm(mac_hndlr[i])

for i in range(0, len(rlc_hndlr)):
    ric.rm_report_rlc_sm(rlc_hndlr[i])

# for i in range(0, len(pdcp_hndlr)):
#     ric.rm_report_pdcp_sm(pdcp_hndlr[i])

print(f"\nTest finished after {runtime} seconds")

####################
#### PROCESS MAC DATA INTO 2D MATRIX FOR ML
####################

print(f"\n{'='*80}")
print(f"PROCESSING MAC DATA FOR MACHINE LEARNING")
print(f"{'='*80}")

print(f"\nMAC Queue Size: {mac_queue.qsize()} samples")
print(f"RLC Queue Size: {rlc_queue.qsize()} samples")
print(f"PDCP Queue Size: {pdcp_queue.qsize()} samples")

# Extract all MAC samples from queue
mac_samples = []
while not mac_queue.empty():
    mac_samples.append(mac_queue.get())

if len(mac_samples) == 0:
    print("\nNo MAC data collected. Exiting...")
else:
    print(f"\n✓ Collected {len(mac_samples)} MAC samples")
    
    # Step 1: Convert to DataFrame (2D structure)
    df_mac = pd.DataFrame(mac_samples)
    
    print(f"\n{'='*80}")
    print(f"STEP 1: MAC DATA AS 2D MATRIX")
    print(f"{'='*80}")
    print(f"Shape: {df_mac.shape} (samples × features)")
    print(f"Columns: {list(df_mac.columns)}")
    print(f"\nFirst 5 samples:")
    print(df_mac.head())
    print(f"\nLast 5 samples:")
    print(df_mac.tail())
    
    # Step 2: Select features for ML (exclude timestamp as feature, but keep for reference)
    feature_columns = ['rnti', 'phr', 'dl_tbs', 'ul_tbs', 'dl_aggr_prb', 
                       'wb_cqi', 'pusch_snr', 'pucch_snr', 'ul_rssi', 
                       'dl_bler', 'ul_bler', 'dl_mcs', 'ul_mcs']
    
    # Verify all columns exist
    available_features = [col for col in feature_columns if col in df_mac.columns]
    print(f"\n{'='*80}")
    print(f"STEP 2: FEATURE SELECTION")
    print(f"{'='*80}")
    print(f"Selected features: {available_features}")
    
    X_full = df_mac[available_features].copy()
    print(f"X matrix shape: {X_full.shape}")
    
    # Step 3: Create target Y = RSSI shifted by +1 timestep (future prediction)
    print(f"\n{'='*80}")
    print(f"STEP 3: CREATE TARGET Y (FUTURE RSSI PREDICTION)")
    print(f"{'='*80}")
    
    if 'ul_rssi' in df_mac.columns:
        # Y[t] = RSSI[t+1] (shift RSSI backward by -1, so current X predicts future Y)
        Y_full = df_mac['ul_rssi'].shift(-1).copy()
        
        print(f"Target: ul_rssi (shifted by +1 timestep)")
        print(f"Y vector shape: {Y_full.shape}")
        print(f"\nExample: X[t] → Y[t] mapping:")
        print(f"{'Sample':<8} {'X[t] RSSI':<15} {'Y[t] (Future RSSI)':<20}")
        print(f"{'-'*45}")
        for i in range(min(5, len(df_mac))):
            x_rssi = df_mac['ul_rssi'].iloc[i]
            y_rssi = Y_full.iloc[i]
            print(f"{i:<8} {x_rssi:<15.2f} {y_rssi if pd.notna(y_rssi) else 'NaN (no future)':<20}")
        
        # Step 4: Handle the last sample (no future label available)
        print(f"\n{'='*80}")
        print(f"STEP 4: HANDLE MISSING FUTURE LABEL")
        print(f"{'='*80}")
        print(f"Last sample has no future RSSI (Y is NaN)")
        print(f"Solution: Drop last sample from both X and Y")
        
        # Remove last row where Y is NaN
        valid_indices = Y_full.notna()
        X = X_full[valid_indices].copy()
        Y = Y_full[valid_indices].copy()
        
        print(f"\nFinal shapes after dropping last sample:")
        print(f"X shape: {X.shape} (samples × features)")
        print(f"Y shape: {Y.shape} (samples,)")
        print(f"Dropped samples: {(~valid_indices).sum()}")
        
        # Step 5: Summary statistics
        print(f"\n{'='*80}")
        print(f"STEP 5: DATASET STATISTICS")
        print(f"{'='*80}")
        print(f"\nFeature statistics (X):")
        print(X.describe())
        
        print(f"\nTarget statistics (Y - Future RSSI):")
        print(Y.describe())
        
        # Step 6: Convert to numpy arrays for ML frameworks
        print(f"\n{'='*80}")
        print(f"STEP 6: CONVERT TO NUMPY ARRAYS")
        print(f"{'='*80}")
        
        X_array = X.to_numpy()
        Y_array = Y.to_numpy()
        
        print(f"X_array shape: {X_array.shape} dtype: {X_array.dtype}")
        print(f"Y_array shape: {Y_array.shape} dtype: {Y_array.dtype}")
        
        # Step 7: Save to files for later use
        print(f"\n{'='*80}")
        print(f"STEP 7: SAVE PROCESSED DATA")
        print(f"{'='*80}")
        
        output_dir = "/home/fahad/srsRAN_4g/ml_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        df_combined = X.copy()
        df_combined['target_future_rssi'] = Y
        csv_path = os.path.join(output_dir, "mac_ml_dataset.csv")
        df_combined.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Save as numpy arrays
        X_path = os.path.join(output_dir, "X_features.npy")
        Y_path = os.path.join(output_dir, "Y_target.npy")
        np.save(X_path, X_array)
        np.save(Y_path, Y_array)
        print(f"✓ Saved X array: {X_path}")
        print(f"✓ Saved Y array: {Y_path}")
        
        # Save feature names
        feature_names_path = os.path.join(output_dir, "feature_names.txt")
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(available_features))
        print(f"✓ Saved feature names: {feature_names_path}")
        
        # Step 8: Example ML usage
        print(f"\n{'='*80}")
        print(f"STEP 8: EXAMPLE - HOW TO USE THIS DATA FOR ML")
        print(f"{'='*80}")
        
        print(f"\n{'='*80}")
        print(f"✓ MAC DATA PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Ready for ML training with {X.shape[0]} samples and {X.shape[1]} features")
        
        ####################
        #### MLP TRAINING PIPELINE
        ####################
        
        print(f"\n{'='*80}")
        print(f"STEP 9: MLP TRAINING PIPELINE")
        print(f"{'='*80}")
        
        # Step 9.1: Drop constant features
        print(f"\n--- 9.1: Drop Constant Features ---")
        print(f"Original shape: {X.shape}")
        
        # Calculate variance for each feature
        feature_variance = X.var()
        print(f"\nFeature variances:")
        for i, col in enumerate(available_features):
            print(f"  {col:<20} variance: {feature_variance.iloc[i]:.6f}")
        
        # Identify constant features (variance = 0 or very close to 0)
        constant_threshold = 1e-10
        constant_features = feature_variance[feature_variance < constant_threshold].index.tolist()
        
        if len(constant_features) > 0:
            print(f"\n⚠ Constant features detected (variance < {constant_threshold}): {constant_features}")
            X_filtered = X.drop(columns=constant_features)
            remaining_features = [f for f in available_features if f not in constant_features]
            print(f"✓ Dropped {len(constant_features)} constant feature(s)")
        else:
            print(f"✓ No constant features detected")
            X_filtered = X.copy()
            remaining_features = available_features.copy()
        
        print(f"Filtered shape: {X_filtered.shape}")
        print(f"Remaining features: {remaining_features}")
        
        # Step 9.2: Scale ALL features including RSSI
        print(f"\n--- 9.2: Scale ALL Features with StandardScaler (including RSSI) ---")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        print(f"✓ Scaled {X_scaled.shape[1]} features (including ul_rssi)")
        print(f"\nScaler statistics:")
        for i, feature in enumerate(remaining_features):
            print(f"  {feature:<20} mean: {scaler.mean_[i]:>10.4f}  std: {scaler.scale_[i]:>10.4f}")
        
        final_features = remaining_features
        print(f"\nFinal feature matrix shape: {X_scaled.shape}")
        print(f"Final features: {final_features}")
        
        # Step 9.3: Split into train/test (90% / 10%)
        print(f"\n--- 9.3: Train/Test Split (90% / 10%) ---")
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y.values, test_size=0.1, random_state=42
        )
        
        print(f"Training set:   X={X_train.shape}  Y={Y_train.shape}")
        print(f"Test set:       X={X_test.shape}  Y={Y_test.shape}")
        print(f"Train samples: {len(X_train)} ({len(X_train)/len(X_scaled)*100:.1f}%)")
        print(f"Test samples:  {len(X_test)} ({len(X_test)/len(X_scaled)*100:.1f}%)")
        
        # Step 9.4: Train MLP
        print(f"\n--- 9.4: Train MLP Regressor ---")
        print(f"Model architecture: Input({X_train.shape[1]}) → Hidden(64) → Hidden(32) → Output(1)")
        print(f"Activation: ReLU, Solver: adam, Max iterations: 500")
        
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )
        
        print(f"\nTraining MLP...")
        train_start = time.time()
        mlp.fit(X_train, Y_train)
        train_time = time.time() - train_start
        
        print(f"✓ Training completed in {train_time:.2f} seconds")
        print(f"  Iterations: {mlp.n_iter_}")
        print(f"  Training loss: {mlp.loss_:.6f}")
        
        # Step 9.5: Evaluate on test set
        print(f"\n--- 9.5: Evaluate on Test Set ---")
        
        Y_test_pred = mlp.predict(X_test)
        
        # Calculate MSE only
        test_mse = mean_squared_error(Y_test, Y_test_pred)
        
        print(f"\nMSE: {test_mse:.6f}")
        
        # Step 9.6: Show prediction examples
        print(f"\n--- 9.6: Prediction Examples (Test Set) ---")
        print(f"{'Sample':<8} {'Actual RSSI':<15} {'Predicted RSSI':<18} {'Error':<12}")
        print(f"{'-'*55}")
        
        num_examples = min(10, len(Y_test))
        for i in range(num_examples):
            actual = Y_test[i]
            predicted = Y_test_pred[i]
            error = actual - predicted
            print(f"{i:<8} {actual:<15.4f} {predicted:<18.4f} {error:<12.4f}")
        
        # Step 9.7: Save model and scaler
        print(f"\n--- 9.7: Save Trained Model ---")
        
        import pickle
        
        model_path = os.path.join(output_dir, "mlp_model.pkl")
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(mlp, f)
        print(f"✓ Saved MLP model: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # Save feature configuration
        config_path = os.path.join(output_dir, "model_config.txt")
        with open(config_path, 'w') as f:
            f.write(f"Model Configuration\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Input Features ({len(final_features)}):\n")
            for i, feat in enumerate(final_features):
                f.write(f"  {i+1}. {feat}\n")
            f.write(f"\nTarget: Future RSSI (ul_rssi at t+1)\n")
            f.write(f"\nModel Architecture:\n")
            f.write(f"  Input layer: {X_train.shape[1]} neurons\n")
            f.write(f"  Hidden layer 1: 64 neurons (ReLU)\n")
            f.write(f"  Hidden layer 2: 32 neurons (ReLU)\n")
            f.write(f"  Output layer: 1 neuron (linear)\n")
            f.write(f"\nTraining Configuration:\n")
            f.write(f"  Solver: adam\n")
            f.write(f"  Max iterations: 500\n")
            f.write(f"  Early stopping: Yes\n")
            f.write(f"  Train/Test split: 90/10\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  Test MSE: {test_mse:.6f}\n")
        print(f"✓ Saved model config: {config_path}")
        
        print(f"\n{'='*80}")
        print(f"✓ MLP TRAINING PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Model Performance: MSE = {test_mse:.6f}")
        print(f"\nModel ready for inference!")
        
    else:
        print("ERROR: 'ul_rssi' column not found in MAC data!")

print(f"\n{'='*80}")
