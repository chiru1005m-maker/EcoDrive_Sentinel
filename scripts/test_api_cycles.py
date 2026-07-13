import requests
import time

base = {'battery_id': 'B0005', 'chemistry': 'LiFePO4', 'voltage': 3.2, 'current': -1.0, 'temperature': 25.0, 'timestamp': time.time()}

for cycles in [100, 1000, 2000, 3000, 4500, 5800]:
    payload = base.copy()
    payload['cycle_count'] = cycles
    resp = requests.post('http://localhost:8000/api/v1/diagnose', json=payload)
    print(f'Cycles: {cycles} -> RUL: {resp.json().get("rul_percent")}%')
