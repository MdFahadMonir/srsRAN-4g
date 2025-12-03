import xapp_sdk as ric
import time
import os
import queue
import threading
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from collections import deque
from datetime import datetime

####################
#### CONFIGURATION
####################
RUNTIME = 150  # Total runtime in seconds (allow multiple training cycles)
MAC_INTERVAL = ric.Interval_ms_10  # 10ms interval
RLC_INTERVAL = ric.Interval_ms_10

# Training configuration
TRAINING_INTERVAL = 30  # Retrain every 30 seconds (allow time for training to complete)
MIN_SAMPLES_FOR_TRAINING = 600  # Wait for sufficient data before first training
BUFFER_SIZE = 800  # Rolling buffer size (sweet spot: stable learning + good adaptation)

# Output directory
OUTPUT_DIR = "/home/fahad/srsRAN_4g/ml_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns - REMOVED dl_tbs, ul_tbs (extreme distribution shifts between trainings)
# These throughput values change dramatically (80k→75k) causing model instability
# Keeping only RSSI-relevant, stable features
FEATURE_COLUMNS = ['dl_aggr_prb', 'wb_cqi', 'pusch_snr', 'pucch_snr', 'ul_rssi',
                   'dl_bler', 'ul_bler', 'dl_mcs', 'ul_mcs']

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
current_model = None
current_scaler = None
model_version = 0
feature_names = None

# Thread control
shutdown_event = threading.Event()
stats_lock = threading.Lock()

# Prediction error tracking (for real-time test MSE)
prediction_errors = deque(maxlen=1000)  # Keep last 1000 prediction errors
prediction_errors_lock = threading.Lock()

# Statistics
stats = {
    'total_kpis_received': 0,
    'total_predictions': 0,
    'total_trainings': 0,
    'last_training_time': None,
    'last_prediction_time': None,
    'last_training_mse': None,  # MSE on training data
    'last_test_mse': None,      # MSE on real-time predictions
    'mse_history': []           # Track MSE evolution: [(time, training_mse, test_mse), ...]
}

####################
#### MAC CALLBACK
####################

class MACCallback(ric.mac_cb):
    def __init__(self):
        ric.mac_cb.__init__(self)
    
    def handle(self, ind):
        if len(ind.ue_stats) > 0:
            stats = ind.ue_stats[0]
            
            # Create KPI sample
            kpi_sample = {
                'timestamp': ind.tstamp / 1.0,
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
            
            # Send to KPI collector thread
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
#### THREAD 1: KPI COLLECTOR
####################

def kpi_collector_thread():
    """
    Receives KPI samples, aligns labels (future RSSI), 
    sends to prediction queue, and stores in training buffer.
    """
    print(f"[KPI Collector] Thread started")
    
    samples_buffer = []  # Temporary buffer to align labels
    
    while not shutdown_event.is_set():
        try:
            # Get new KPI sample (timeout to check shutdown_event)
            kpi_sample = kpi_queue.get(timeout=0.1) #Avoide blocking indefinitely so thread can check for shutdown
            
            with stats_lock:
                stats['total_kpis_received'] += 1
            
            samples_buffer.append(kpi_sample)
            
            # If we have at least 2 samples, we can create X[t] -> Y[t+1] pair
            if len(samples_buffer) >= 2:
                # Take the second-to-last sample as X
                X_sample = samples_buffer[-2]
                # Take the last sample's RSSI as Y (future label)
                Y_label = samples_buffer[-1]['ul_rssi']
                
                # Create labeled sample
                labeled_sample = X_sample.copy()
                labeled_sample['target_future_rssi'] = Y_label
                
                # Send to prediction queue (for immediate testing)
                prediction_queue.put(labeled_sample)
                
                # Add to training buffer
                with training_buffer_lock:
                    training_buffer.append(labeled_sample)
                
                # Keep only recent samples in buffer (last 100) to avoid memory issues
                if len(samples_buffer) > 100:
                    samples_buffer = samples_buffer[-100:]
        
        except queue.Empty:
            continue
    
    print(f"[KPI Collector] Thread stopped")

####################
#### THREAD 2: TRAINER
####################

def trainer_thread():
    """
    Periodically retrains the model using the latest training buffer.
    Updates model version for the test thread to reload.
    """
    global current_model, current_scaler, model_version, feature_names
    
    print(f"[Trainer] Thread started")
    print(f"[Trainer] Will retrain every {TRAINING_INTERVAL} seconds")
    
    last_train_time = time.time()
    
    while not shutdown_event.is_set():
        current_time = time.time()
        
        # Check if it's time to retrain
        if current_time - last_train_time >= TRAINING_INTERVAL:
            # Get training data from buffer
            with training_buffer_lock:
                buffer_size = len(training_buffer)
                if buffer_size >= MIN_SAMPLES_FOR_TRAINING:
                    # Copy buffer to avoid holding lock during training
                    training_data = list(training_buffer)
                else:
                    training_data = None
            
            if training_data:
                samples_count = len(training_data)
                print(f"\n[Trainer] Starting training with {samples_count} samples...")
                train_start = time.time()
                
                try:
                    # Convert to DataFrame
                    df = pd.DataFrame(training_data)
                    
                    # Separate features and target
                    X = df[FEATURE_COLUMNS].copy()
                    Y = df['target_future_rssi'].copy()
                    
                    # DIAGNOSTIC: Print training data statistics
                    print(f"\n[Trainer] === TRAINING DATA STATISTICS ===")
                    print(f"[Trainer] Samples: {len(X)}")
                    print(f"[Trainer] Target (target_future_rssi):")
                    print(f"  Mean: {Y.mean():.2f} dB, Std: {Y.std():.2f} dB")
                    print(f"  Min: {Y.min():.2f} dB, Max: {Y.max():.2f} dB")
                    print(f"[Trainer] Features (Top 4):")
                    for feat in FEATURE_COLUMNS[:4]:
                        print(f"  {feat}: Mean={X[feat].mean():.2f}, Std={X[feat].std():.2f}, Min={X[feat].min():.2f}, Max={X[feat].max():.2f}")
                    
                    # Drop constant features
                    feature_variance = X.var()
                    constant_threshold = 1e-10
                    constant_features = feature_variance[feature_variance < constant_threshold].index.tolist()
                    
                    if len(constant_features) > 0:
                        X = X.drop(columns=constant_features)
                        remaining_features = [f for f in FEATURE_COLUMNS if f not in constant_features]
                    else:
                        remaining_features = FEATURE_COLUMNS.copy()
                    
                    # ROBUST SCALING: Handles outliers and distribution shifts
                    # Uses median and IQR instead of mean/std or min/max
                    # Critical for online learning where data distribution changes over time
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # DIAGNOSTIC: Print scaler parameters
                    print(f"[Trainer] Scaling: RobustScaler (median/IQR) for ALL features")
                    print(f"[Trainer] Feature statistics (first 4):")
                    for i, feat in enumerate(remaining_features[:4]):
                        print(f"  {feat}: median={scaler.center_[i]:.2f}, scale={scaler.scale_[i]:.2f}")
                    
                    # Train MLP for online learning
                    # With only 4-5 features, use simpler network and higher iterations
                    mlp = MLPRegressor(
                        hidden_layer_sizes=(16, 8),  # REDUCED: Simpler for limited features (was 32,16)
                        activation='relu',
                        solver='adam',
                        max_iter=500,  # INCREASED: Allow full convergence
                        random_state=42,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,  # INCREASED: Faster convergence (was 0.0005)
                        early_stopping=True,
                        validation_fraction=0.15,
                        verbose=False,
                        alpha=0.001,  # REDUCED regularization: Less overfitting risk with simple network (was 0.01)
                        tol=1e-4,  # Stricter tolerance for better convergence
                        n_iter_no_change=20,  # More patience
                        warm_start=False,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-8
                    )
                    
                    mlp.fit(X_scaled, Y.values)
                    
                    # Calculate training MSE
                    Y_pred = mlp.predict(X_scaled)
                    training_mse = mean_squared_error(Y.values, Y_pred)
                    
                    train_time = time.time() - train_start
                    
                    # Calculate current prediction MSE (from recent predictions WITH OLD MODEL)
                    with prediction_errors_lock:
                        if len(prediction_errors) > 0:
                            recent_errors = list(prediction_errors)
                            test_mse = np.mean([e**2 for e in recent_errors])  # MSE from squared errors
                            num_test_samples = len(recent_errors)
                        else:
                            test_mse = None
                            num_test_samples = 0
                        
                        # CRITICAL: Clear prediction errors after training
                        # This ensures test MSE only reflects NEW model performance, not old predictions
                        prediction_errors.clear()
                    
                    # Update global model (thread-safe)
                    with model_lock:
                        current_model = mlp
                        current_scaler = scaler  # MinMaxScaler
                        feature_names = remaining_features
                        model_version += 1
                        
                        # Update stats
                        with stats_lock:
                            stats['total_trainings'] += 1
                            stats['last_training_time'] = datetime.now().strftime('%H:%M:%S')
                            stats['last_training_mse'] = training_mse
                            stats['last_test_mse'] = test_mse
                            # Store history
                            stats['mse_history'].append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'version': model_version,
                                'training_mse': training_mse,
                                'test_mse': test_mse,
                                'num_test_samples': num_test_samples
                            })
                    
                    test_mse_str = f"{test_mse:.6f} (n={num_test_samples})" if test_mse else "N/A (no predictions yet)"
                    print(f"[Trainer] ✓ Training complete in {train_time:.2f}s | Version: {model_version}")
                    print(f"          Training MSE (NEW Model): {training_mse:.6f} | Test MSE (OLD Snap [Right Before New Model Swap]): {test_mse_str}")
                    
                    # Save model to disk
                    model_path = os.path.join(OUTPUT_DIR, f"mlp_model_v{model_version}.pkl")
                    scaler_path = os.path.join(OUTPUT_DIR, f"scaler_v{model_version}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(mlp, f)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)  # Save MinMaxScaler
                    
                except Exception as e:
                    print(f"[Trainer] ✗ Training failed: {e}")
                
                last_train_time = current_time
            else:
                print(f"[Trainer] Waiting for more samples... ({buffer_size}/{MIN_SAMPLES_FOR_TRAINING})")
                last_train_time = current_time  # Reset timer
        
        # Sleep briefly to avoid busy waiting
        time.sleep(1)
    
    print(f"[Trainer] Thread stopped")

####################
#### THREAD 3: TEST/PREDICTION
####################

def test_thread():
    """
    For each new KPI sample, loads the latest model and makes a prediction.
    Does NOT perform training - only inference.
    """
    print(f"[Test] Thread started")
    
    local_model = None
    local_scaler = None
    local_feature_names = None
    local_model_version = 0
    
    while not shutdown_event.is_set():
        try:
            # Get labeled sample (timeout to check shutdown_event)
            sample = prediction_queue.get(timeout=0.1) #Avoide blocking indefinitely so thread can check for shutdown
            
            # Check if there's a newer model version
            with model_lock:
                if model_version > local_model_version:
                    # Reload model
                    local_model = current_model
                    local_scaler = current_scaler
                    local_feature_names = feature_names
                    local_model_version = model_version
                    print(f"[Test] Loaded model version {local_model_version}")
            
            # Make prediction if model is available
            if local_model is not None:
                try:
                    # ROBUST SCALING: Apply same median/IQR scaling as training
                    X_sample = {k: [sample[k]] for k in local_feature_names}
                    X_df = pd.DataFrame(X_sample, columns=local_feature_names)
                    X_scaled = local_scaler.transform(X_df)
                    
                    # Predict
                    Y_pred = local_model.predict(X_scaled)[0]
                    Y_actual = sample['target_future_rssi']
                    error = Y_actual - Y_pred
                    
                    # DIAGNOSTIC: Print first 5 test samples to compare with training data
                    with prediction_errors_lock:
                        num_predictions = len(prediction_errors)
                        if num_predictions < 5:
                            print(f"\n[Test] === TEST SAMPLE #{num_predictions+1} ===")
                            print(f"[Test] Features (RAW):")
                            for feat in local_feature_names[:4]:  # Show top 4
                                print(f"  {feat}: {sample[feat]:.2f}")
                            print(f"[Test] Features (SCALED):")
                            for i, feat in enumerate(local_feature_names[:4]):
                                print(f"  {feat}: {X_scaled[0][i]:.4f}")
                            print(f"[Test] Target (actual): {Y_actual:.2f} dB")
                            print(f"[Test] Prediction: {Y_pred:.2f} dB")
                            print(f"[Test] Error: {error:.2f} dB")
                    
                    # Store prediction error for test MSE calculation
                    with prediction_errors_lock:
                        prediction_errors.append(error)
                    
                    # Update stats
                    with stats_lock:
                        stats['total_predictions'] += 1
                        stats['last_prediction_time'] = datetime.now().strftime('%H:%M:%S')
                    
                    # Print prediction (every 100 predictions to avoid spam)
                    if stats['total_predictions'] % 100 == 0:
                        # Calculate rolling test MSE (most recent ≤1000 predictions)
                        with prediction_errors_lock:
                            if len(prediction_errors) > 0:
                                rolling_mse = np.mean([e**2 for e in prediction_errors])
                                print(f"[Test] Prediction #{stats['total_predictions']} | "
                                      f"Actual: {Y_actual:.2f} | Predicted: {Y_pred:.2f} | Error: {error:.2f} | "
                                      f"Rolling MSE: {rolling_mse:.6f} | Model v{local_model_version}")
                
                except Exception as e:
                    print(f"[Test] ✗ Prediction failed: {e}")
            else:
                # No model yet - skip prediction
                pass
        
        except queue.Empty:
            continue
    
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
        
        # Calculate CURRENT test MSE (not stale value from last training)
        with prediction_errors_lock:
            if len(prediction_errors) > 0:
                current_test_mse = np.mean([e**2 for e in prediction_errors])
                current_test_samples = len(prediction_errors)
            else:
                current_test_mse = None
                current_test_samples = 0
        
        with stats_lock:
            print(f"\n{'='*80}")
            print(f"[Monitor] Statistics:")
            print(f"  KPIs Received:        {stats['total_kpis_received']}")
            print(f"  Predictions Made:     {stats['total_predictions']}")
            print(f"  Trainings Completed:  {stats['total_trainings']}")
            print(f"  Last Training:        {stats['last_training_time'] or 'Not yet'}")
            print(f"  Last Prediction:      {stats['last_prediction_time'] or 'Not yet'}")
            print(f"  Training Buffer Size: {len(training_buffer)}/{BUFFER_SIZE}")
            print(f"  Current Model Version: {model_version}")
            print(f"\n  Performance Metrics:")
            train_mse_str = f"{stats['last_training_mse']:.6f}" if stats['last_training_mse'] is not None else 'N/A'
            
            # Show CURRENT test MSE (live), not the stale value from last training
            if current_test_mse is not None:
                test_mse_str = f"{current_test_mse:.6f} (n={current_test_samples}, LIVE)"
            else:
                test_mse_str = 'N/A'
            
            print(f"    Training MSE (Last): {train_mse_str}")
            print(f"    Test MSE (Current):  {test_mse_str}")
            
            # Show MSE evolution if history exists
            if len(stats['mse_history']) > 0:
                print(f"\n  MSE History (last {min(5, len(stats['mse_history']))} trainings):")
                print(f"    {'Time':<10} {'Ver':<5} {'Train MSE':<12} {'Test MSE':<20}")
                print(f"    {'-'*50}")
                for record in stats['mse_history'][-5:]:
                    if record['test_mse']:
                        test_mse_str = f"{record['test_mse']:.4f} (n={record.get('num_test_samples', '?')})"
                    else:
                        test_mse_str = "N/A"
                    print(f"    {record['time']:<10} {record['version']:<5} {record['training_mse']:<12.6f} {test_mse_str:<20}")
            
            print(f"{'='*80}\n")
    
    print(f"[Monitor] Thread stopped")

####################
#### MAIN
####################

def main():
    print(f"\n{'='*80}")
    print(f"ONLINE LEARNING xApp - Near Real-Time Training & Prediction")
    print(f"{'='*80}")
    print(f"Runtime: {RUNTIME} seconds")
    print(f"Training Interval: {TRAINING_INTERVAL} seconds")
    print(f"KPI Collection: Every 10ms")  # MAC_INTERVAL = Interval_ms_10
    print(f"Min Samples Before Training: {MIN_SAMPLES_FOR_TRAINING}")
    print(f"{'='*80}\n")


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
    
    # Calculate final LIVE test MSE
    with prediction_errors_lock:
        if len(prediction_errors) > 0:
            final_live_test_mse = np.mean([e**2 for e in prediction_errors])
            final_test_samples = len(prediction_errors)
        else:
            final_live_test_mse = None
            final_test_samples = 0
    
    with stats_lock:
        print(f"Total KPIs Received:       {stats['total_kpis_received']}")
        print(f"Total Predictions Made:    {stats['total_predictions']}")
        print(f"Total Trainings:           {stats['total_trainings']}")
        print(f"Final Model Version:       {model_version}")
        print(f"Training Buffer Size:      {len(training_buffer)}/{BUFFER_SIZE}")
        print(f"\nFinal Performance:")
        if stats['last_training_mse']:
            print(f"  Training MSE (v{model_version}):     {stats['last_training_mse']:.6f}")
        if final_live_test_mse:
            print(f"  Test MSE (v{model_version} LIVE):    {final_live_test_mse:.6f} (n={final_test_samples})")
            print(f"  → Current model's real-time prediction error")
        
        # Show full MSE history
        if len(stats['mse_history']) > 0:
            print(f"\nMSE Evolution During Training:")
            print(f"  (Note: Test MSE reflects previous model's performance before retraining)")
            print(f"  {'Time':<10} {'Ver':<5} {'Train MSE':<12} {'Test MSE (prev)':<20} {'Improvement':<12}")
            print(f"  {'-'*65}")
            prev_test_mse = None
            for record in stats['mse_history']:
                if record['test_mse']:
                    test_mse_str = f"{record['test_mse']:.4f} (n={record.get('num_test_samples', '?')})"
                else:
                    test_mse_str = "N/A"
                if prev_test_mse and record['test_mse']:
                    improvement = prev_test_mse - record['test_mse']
                    improvement_str = f"{improvement:+.4f}"
                else:
                    improvement_str = "—"
                print(f"  {record['time']:<10} {record['version']:<5} {record['training_mse']:<12.6f} "
                      f"{test_mse_str:<20} {improvement_str:<12}")
                if record['test_mse']:
                    prev_test_mse = record['test_mse']
    print(f"{'='*80}\n")
    
    # Save final training buffer
    if len(training_buffer) > 0:
        print(f"Saving final training buffer ({len(training_buffer)} samples)...")
        df_final = pd.DataFrame(list(training_buffer))
        csv_path = os.path.join(OUTPUT_DIR, "online_learning_final_buffer.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"✓ Saved to: {csv_path}")
    
    print(f"\n✓ xApp terminated successfully\n")

if __name__ == "__main__":
    main()
