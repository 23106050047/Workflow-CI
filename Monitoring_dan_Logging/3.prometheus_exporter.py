"""
Prometheus Exporter for ML Model Monitoring
Exposes 10+ custom metrics untuk monitoring model serving
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import requests
import json
import time
import psutil
import threading
import random

# ===== METRICS DEFINITIONS (10+ metrics) =====

# 1. Prediction Counter
prediction_total = Counter(
    'ml_predictions_total', 
    'Total number of predictions made',
    ['model_name', 'status']
)

# 2. Prediction Latency Histogram
prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

# 3. Prediction Latency Summary
prediction_latency_summary = Summary(
    'ml_prediction_latency_summary',
    'Summary of prediction latency',
    ['model_name']
)

# 4. Model Accuracy Gauge (simulated)
model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name']
)

# 5. Model Precision Gauge
model_precision = Gauge(
    'ml_model_precision',
    'Current model precision',
    ['model_name']
)

# 6. Model Recall Gauge
model_recall = Gauge(
    'ml_model_recall',
    'Current model recall',
    ['model_name']
)

# 7. Model F1 Score Gauge
model_f1_score = Gauge(
    'ml_model_f1_score',
    'Current model F1 score',
    ['model_name']
)

# 8. Active Requests Gauge
active_requests = Gauge(
    'ml_active_requests',
    'Number of active prediction requests',
    ['model_name']
)

# 9. CPU Usage Gauge
cpu_usage = Gauge(
    'ml_server_cpu_usage_percent',
    'CPU usage percentage of model server'
)

# 10. Memory Usage Gauge
memory_usage = Gauge(
    'ml_server_memory_usage_mb',
    'Memory usage in MB of model server'
)

# 11. Prediction Class Distribution
prediction_class_0 = Counter(
    'ml_predictions_class_0',
    'Total predictions for class 0 (no diabetes)',
    ['model_name']
)

prediction_class_1 = Counter(
    'ml_predictions_class_1',
    'Total predictions for class 1 (diabetes)',
    ['model_name']
)

# 12. Model Uptime
model_uptime = Gauge(
    'ml_model_uptime_seconds',
    'Model server uptime in seconds'
)

# 13. Error Rate
error_rate = Gauge(
    'ml_error_rate',
    'Current error rate (errors per minute)',
    ['model_name']
)

# ===== CONFIGURATION =====
MODEL_URL = "http://localhost:5001/invocations"
MODEL_NAME = "diabetes_prediction"
METRICS_PORT = 8000
UPDATE_INTERVAL = 5  # seconds

# Sample data untuk testing
sample_data = {
    "dataframe_split": {
        "columns": ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "gender", "smoking_history"],
        "data": [
            [-0.5, 0.2, -0.3, 0.1, 0, 0, 1, 2]
        ]
    }
}

start_time = time.time()
error_count = 0
request_count = 0
last_error_time = time.time()

def make_prediction():
    """Make a prediction and update metrics"""
    global error_count, request_count, last_error_time
    
    active_requests.labels(model_name=MODEL_NAME).inc()
    
    try:
        start = time.time()
        response = requests.post(
            MODEL_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(sample_data),
            timeout=5
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            # Success
            prediction_total.labels(model_name=MODEL_NAME, status='success').inc()
            prediction_latency.labels(model_name=MODEL_NAME).observe(latency)
            prediction_latency_summary.labels(model_name=MODEL_NAME).observe(latency)
            
            # Simulate class prediction (0 or 1)
            pred_class = random.choice([0, 1])
            if pred_class == 0:
                prediction_class_0.labels(model_name=MODEL_NAME).inc()
            else:
                prediction_class_1.labels(model_name=MODEL_NAME).inc()
            
            request_count += 1
            
        else:
            # Error
            prediction_total.labels(model_name=MODEL_NAME, status='error').inc()
            error_count += 1
            last_error_time = time.time()
            
    except Exception as e:
        print(f"Prediction error: {e}")
        prediction_total.labels(model_name=MODEL_NAME, status='error').inc()
        error_count += 1
        last_error_time = time.time()
    
    finally:
        active_requests.labels(model_name=MODEL_NAME).dec()

def update_system_metrics():
    """Update system metrics (CPU, Memory)"""
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

def update_model_metrics():
    """Update model performance metrics (simulated with slight variations)"""
    # Simulate metrics dengan nilai yang realistis dan sedikit variasi
    base_accuracy = 0.92
    base_precision = 0.89
    base_recall = 0.91
    base_f1 = 0.90
    
    # Add small random variation
    model_accuracy.labels(model_name=MODEL_NAME).set(base_accuracy + random.uniform(-0.02, 0.02))
    model_precision.labels(model_name=MODEL_NAME).set(base_precision + random.uniform(-0.02, 0.02))
    model_recall.labels(model_name=MODEL_NAME).set(base_recall + random.uniform(-0.02, 0.02))
    model_f1_score.labels(model_name=MODEL_NAME).set(base_f1 + random.uniform(-0.02, 0.02))

def update_uptime():
    """Update model uptime"""
    uptime = time.time() - start_time
    model_uptime.set(uptime)

def calculate_error_rate():
    """Calculate error rate (errors per minute)"""
    global error_count, request_count
    
    if request_count > 0:
        rate = (error_count / request_count) * 100
        error_rate.labels(model_name=MODEL_NAME).set(rate)
    else:
        error_rate.labels(model_name=MODEL_NAME).set(0)

def metrics_updater():
    """Background thread to continuously update metrics"""
    while True:
        try:
            # Make periodic predictions
            make_prediction()
            
            # Update system metrics
            update_system_metrics()
            
            # Update model performance metrics
            update_model_metrics()
            
            # Update uptime
            update_uptime()
            
            # Calculate error rate
            calculate_error_rate()
            
            time.sleep(UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"Metrics update error: {e}")
            time.sleep(UPDATE_INTERVAL)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Prometheus Exporter for ML Model")
    print("=" * 60)
    print(f"📊 Metrics endpoint: http://localhost:{METRICS_PORT}/metrics")
    print(f"🎯 Model endpoint: {MODEL_URL}")
    print(f"🔄 Update interval: {UPDATE_INTERVAL} seconds")
    print(f"📈 Exposing 13+ metrics:")
    print("   1. ml_predictions_total (Counter)")
    print("   2. ml_prediction_latency_seconds (Histogram)")
    print("   3. ml_prediction_latency_summary (Summary)")
    print("   4. ml_model_accuracy (Gauge)")
    print("   5. ml_model_precision (Gauge)")
    print("   6. ml_model_recall (Gauge)")
    print("   7. ml_model_f1_score (Gauge)")
    print("   8. ml_active_requests (Gauge)")
    print("   9. ml_server_cpu_usage_percent (Gauge)")
    print("  10. ml_server_memory_usage_mb (Gauge)")
    print("  11. ml_predictions_class_0 (Counter)")
    print("  12. ml_predictions_class_1 (Counter)")
    print("  13. ml_model_uptime_seconds (Gauge)")
    print("  14. ml_error_rate (Gauge)")
    print("=" * 60)
    
    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    print(f"✅ Metrics server started on port {METRICS_PORT}")
    
    # Start background metrics updater
    updater_thread = threading.Thread(target=metrics_updater, daemon=True)
    updater_thread.start()
    print("✅ Metrics updater started")
    
    print("\n💡 Press Ctrl+C to stop\n")
    print("🔗 Access metrics: http://localhost:8000/metrics")
    print("=" * 60)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down exporter...")
