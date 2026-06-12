import asyncio
import csv
import io
import json
import logging
from datetime import datetime, timedelta
import httpx

# Configure logging for the console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EV-Simulator")

# Generate 60 rows of dummy data for the CSV string
def generate_csv_data():
    header = "timestamp,battery_id,chemistry,voltage_v,current_a,temperature_c,cycle_count,telemetry_event\n"
    rows = []
    start_time = datetime.now()
    for i in range(60):
        dt = (start_time + timedelta(seconds=i)).isoformat()
        event = "HIGHWAY_MERGE_SAG_ALERT" if i % 15 == 0 else "NORMAL_DRIVING"
        vol = 3.8 - (i * 0.005)
        cur = 15.0 + (i * 0.5) if event == "HIGHWAY_MERGE_SAG_ALERT" else 10.5
        temp = 25.0 + (i * 0.1)
        row = f"{dt},BATT-NMC-001,LiNiMnCoO2,{vol:.2f},{cur:.1f},{temp:.1f},152,{event}\n"
        rows.append(row)
    return header + "".join(rows)

CSV_DATA = generate_csv_data()
API_ENDPOINT = "http://localhost:8000/api/v1/diagnose"

async def stream_telemetry():
    logger.info("Initializing Real-Time EV Telemetry Simulator...")
    logger.info(f"Target API Endpoint: {API_ENDPOINT}")
    
    # Parse the CSV string
    f = io.StringIO(CSV_DATA)
    reader = csv.DictReader(f)
    
    # We use an AsyncClient to persist connection pooling over the loop
    async with httpx.AsyncClient() as client:
        for row in reader:
            timestamp = row['timestamp']
            event = row['telemetry_event']
            
            # Construct the Pydantic-compliant payload
            payload = {
                "battery_id": row['battery_id'],
                "timestamp": int(datetime.fromisoformat(row['timestamp']).timestamp()),
                "chemistry": row['chemistry'],
                "voltage": float(row['voltage_v']),
                "current": float(row['current_a']),
                "temperature": float(row['temperature_c']),
                "cycle_count": int(row['cycle_count'])
            }
            
            metrics_str = f"V:{payload['voltage']} A:{payload['current']} T:{payload['temperature']}C"
            logger.info(f"[{event}] {timestamp} -> Sending metrics: {metrics_str}")
            
            # Transmit to the FastAPI endpoint with error handling
            try:
                response = await client.post(API_ENDPOINT, json=payload, timeout=2.0)
                
                # Check if it was successful
                if response.status_code == 200:
                    resp_data = response.json()
                    # Attempt to extract predicted RUL/Status if present
                    rul = resp_data.get('rul', 'N/A')
                    status = resp_data.get('status', 'N/A')
                    logger.info(f"    [API OK {response.status_code}] Predicted RUL: {rul}, Status: {status}")
                else:
                    logger.warning(f"    [API WARN {response.status_code}] Response: {response.text}")
                    
            except httpx.RequestError as exc:
                logger.error(f"    [API ERROR] Failed to reach server. Error: {str(exc)}")
            
            # Pause for exactly 1.0 second to simulate 1Hz sensor broadcast
            await asyncio.sleep(1.0)
            
    logger.info("Telemetry stream completed. Reached end of 60-row dataset.")

if __name__ == "__main__":
    try:
        asyncio.run(stream_telemetry())
    except KeyboardInterrupt:
        logger.info("Telemetry stream interrupted by user.")
