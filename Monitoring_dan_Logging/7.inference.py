import requests
import json
import pandas as pd
import time

# Model server endpoint
MODEL_URL = "http://localhost:5001/invocations"

# Sample data untuk prediction (sesuaikan dengan jumlah features Anda)
# Contoh: 8 features sesuai dataset diabetes preprocessing
sample_data = {
    "dataframe_split": {
        "columns": ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "gender", "smoking_history"],
        "data": [
            [-0.5, 0.2, -0.3, 0.1, 0, 0, 1, 2],  # Sample 1
            [0.8, 1.5, 0.5, 1.2, 1, 0, 0, 1],    # Sample 2
            [-1.2, -0.8, -1.0, -0.5, 0, 1, 1, 0] # Sample 3
        ]
    }
}

def test_inference():
    """Test single prediction"""
    print("=" * 60)
    print("Testing Model Inference")
    print("=" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(
            MODEL_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(sample_data),
            timeout=10
        )
        latency = time.time() - start_time
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"✅ Prediction successful!")
            print(f"📊 Predictions: {predictions}")
            print(f"⏱️  Latency: {latency:.3f} seconds")
            return True, latency
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False, None

def load_test(n_requests=10):
    """Simple load test"""
    print(f"\n{'=' * 60}")
    print(f"Running Load Test ({n_requests} requests)")
    print("=" * 60)
    
    latencies = []
    successes = 0
    
    for i in range(n_requests):
        success, latency = test_inference()
        if success:
            successes += 1
            latencies.append(latency)
        time.sleep(0.5)  # Small delay between requests
    
    if latencies:
        print(f"\n{'=' * 60}")
        print("Load Test Results")
        print("=" * 60)
        print(f"Total Requests: {n_requests}")
        print(f"Successful: {successes}")
        print(f"Failed: {n_requests - successes}")
        print(f"Success Rate: {(successes/n_requests)*100:.1f}%")
        print(f"Avg Latency: {sum(latencies)/len(latencies):.3f}s")
        print(f"Min Latency: {min(latencies):.3f}s")
        print(f"Max Latency: {max(latencies):.3f}s")
        print("=" * 60)

if __name__ == "__main__":
    # Test single inference
    test_inference()
    
    # Optional: Load test
    # load_test(n_requests=20)
