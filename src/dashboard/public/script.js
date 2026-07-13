document.addEventListener('DOMContentLoaded', () => {
    // --- Tab Switching ---
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    // --- Chart.js Setup for Training Monitor ---
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    // Gradient for Train MSE
    let trainGradient = ctx.createLinearGradient(0, 0, 0, 400);
    trainGradient.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
    trainGradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    // Gradient for Val MSE
    let valGradient = ctx.createLinearGradient(0, 0, 0, 400);
    valGradient.addColorStop(0, 'rgba(192, 132, 252, 0.5)');
    valGradient.addColorStop(1, 'rgba(192, 132, 252, 0.0)');

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train MSE',
                    data: [],
                    borderColor: '#00f0ff',
                    backgroundColor: trainGradient,
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Val MSE',
                    data: [],
                    borderColor: '#c084fc',
                    backgroundColor: valGradient,
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#94a3b8',
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f0f4f8' } }
            }
        }
    });

    // --- Training Log Polling ---
    async function fetchTrainingLogs() {
        try {
            const res = await fetch('/api/logs');
            const result = await res.json();
            
            if (result.status === 'success' && result.data.length > 0) {
                const data = result.data;
                const latest = data[data.length - 1];
                
                // Update stats
                document.getElementById('stat-epoch').innerText = `${latest.epoch}/50`;
                document.getElementById('stat-train-mse').innerText = latest.train_mse.toFixed(5);
                document.getElementById('stat-val-mse').innerText = latest.val_mse.toFixed(5);
                
                // Update chart
                chart.data.labels = data.map(d => `Ep ${d.epoch}`);
                chart.data.datasets[0].data = data.map(d => d.train_mse);
                chart.data.datasets[1].data = data.map(d => d.val_mse);
                chart.update();
            }
        } catch (e) {
            console.error('Failed to fetch logs:', e);
        }
    }

    // Poll every 5 seconds
    setInterval(fetchTrainingLogs, 5000);
    fetchTrainingLogs(); // Initial fetch

    // --- Live Inference Form ---
    const form = document.getElementById('inference-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        document.getElementById('results-placeholder').classList.add('hidden');
        document.getElementById('results-display').classList.add('hidden');
        document.getElementById('loading-spinner').classList.remove('hidden');

        const payload = {
            battery_id: document.getElementById('batt-id').value,
            timestamp: Date.now() / 1000,
            voltage: parseFloat(document.getElementById('voltage').value),
            current: parseFloat(document.getElementById('current').value),
            temperature: parseFloat(document.getElementById('temp').value),
            cycle_count: parseInt(document.getElementById('cycle').value),
            chemistry: document.getElementById('chemistry').value
        };
        const nomCapValue = document.getElementById('nominal-capacity').value;
        if (nomCapValue && nomCapValue.trim() !== '') {
            payload.nominal_capacity = parseFloat(nomCapValue);
        }

        try {
            // Note: In production, this would call the actual API on port 8000
            // Since we might run the dashboard separately, we fetch from localhost:8000
            const response = await fetch('http://localhost:8000/api/v1/diagnose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error('API Error');
            
            const data = await response.json();
            
            document.getElementById('res-rul').innerText = data.rul_percent.toFixed(1) + '%';
            
            const statusEl = document.getElementById('res-status');
            statusEl.innerText = data.maintenance_status;
            statusEl.className = 'status-badge';
            if (data.maintenance_status.includes('NORMAL')) statusEl.classList.add('normal');
            else if (data.maintenance_status.includes('WARNING')) statusEl.classList.add('warning');
            else statusEl.classList.add('critical');
            
            document.getElementById('res-llm').innerText = data.llm_summary || "No Agentic summary provided.";

            document.getElementById('loading-spinner').classList.add('hidden');
            document.getElementById('results-display').classList.remove('hidden');

            // Add to History Table
            const historyBody = document.getElementById('history-body');
            const newRow = document.createElement('tr');
            newRow.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
            
            const timeStr = new Date().toLocaleTimeString();
            
            // Format status with color
            let statusColor = '#4ade80'; // normal (green)
            if (data.maintenance_status.includes('WARNING')) statusColor = '#fbbf24'; // yellow
            if (data.maintenance_status.includes('MAINTENANCE_REQUIRED') || data.maintenance_status.includes('CRITICAL')) {
                statusColor = '#ef4444'; // red
            }
            
            newRow.innerHTML = `
                <td style="padding: 10px; color: #94a3b8;">${timeStr}</td>
                <td style="padding: 10px; font-weight: 600;">${payload.battery_id}</td>
                <td style="padding: 10px; color: #94a3b8;">${payload.chemistry}</td>
                <td style="padding: 10px;">${payload.cycle_count}</td>
                <td style="padding: 10px; font-weight: 600; color: #38bdf8;">${data.rul_percent.toFixed(1)}%</td>
                <td style="padding: 10px; font-weight: 600; color: ${statusColor};">${data.maintenance_status}</td>
            `;
            
            // Prepend so newest is at the top
            historyBody.insertBefore(newRow, historyBody.firstChild);

        } catch (e) {
            console.error(e);
            alert('Error calling inference API. Make sure the API server is running on port 8000.');
            document.getElementById('loading-spinner').classList.add('hidden');
            document.getElementById('results-placeholder').classList.remove('hidden');
        }
    });

    // --- Sequence CSV Upload Form ---
    const seqForm = document.getElementById('sequence-form');
    seqForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fileInput = document.getElementById('csv-file');
        const file = fileInput.files[0];
        if (!file) return;

        document.getElementById('results-placeholder').classList.add('hidden');
        document.getElementById('results-display').classList.add('hidden');
        document.getElementById('loading-spinner').classList.remove('hidden');

        const reader = new FileReader();
        reader.onload = async (event) => {
            const csv = event.target.result;
            const lines = csv.split('\n').filter(line => line.trim().length > 0);
            
            // Assume format: Voltage, Current, Temperature, CycleCount
            // We will skip the header if the first line doesn't start with a number
            let startIndex = 0;
            if (lines.length > 0 && isNaN(parseFloat(lines[0].split(',')[0]))) {
                startIndex = 1;
            }

            const readings = [];
            const batteryId = document.getElementById('batt-id').value;
            const chemistry = document.getElementById('chemistry').value;
            const now = Date.now() / 1000;

            for (let i = startIndex; i < lines.length; i++) {
                const parts = lines[i].split(',');
                if (parts.length >= 4) {
                    readings.push({
                        battery_id: batteryId,
                        timestamp: now + i,
                        voltage: parseFloat(parts[0]),
                        current: parseFloat(parts[1]),
                        temperature: parseFloat(parts[2]),
                        cycle_count: parseInt(parts[3]),
                        chemistry: chemistry
                    });
                }
            }

            if (readings.length === 0) {
                alert("Failed to parse CSV or empty file.");
                document.getElementById('loading-spinner').classList.add('hidden');
                document.getElementById('results-placeholder').classList.remove('hidden');
                return;
            }

            const payload = { readings: readings };

            try {
                const response = await fetch('http://localhost:8000/api/v1/diagnose-sequence', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) throw new Error('API Error');
                
                const data = await response.json();
                
                document.getElementById('res-rul').innerText = data.rul_percent.toFixed(1) + '%';
                
                const statusEl = document.getElementById('res-status');
                statusEl.innerText = data.maintenance_status;
                statusEl.className = 'status-badge';
                if (data.maintenance_status.includes('NORMAL')) statusEl.classList.add('normal');
                else if (data.maintenance_status.includes('WARNING')) statusEl.classList.add('warning');
                else statusEl.classList.add('critical');
                
                document.getElementById('res-llm').innerText = data.llm_summary || "No Agentic summary provided.";

                document.getElementById('loading-spinner').classList.add('hidden');
                document.getElementById('results-display').classList.remove('hidden');

                // Add to History Table
                const historyBody = document.getElementById('history-body');
                const newRow = document.createElement('tr');
                newRow.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                
                const timeStr = new Date().toLocaleTimeString();
                
                let statusColor = '#4ade80';
                if (data.maintenance_status.includes('WARNING')) statusColor = '#fbbf24';
                if (data.maintenance_status.includes('MAINTENANCE_REQUIRED') || data.maintenance_status.includes('CRITICAL')) {
                    statusColor = '#ef4444';
                }
                
                newRow.innerHTML = `
                    <td style="padding: 10px; color: #94a3b8;">${timeStr} (CSV)</td>
                    <td style="padding: 10px; font-weight: 600;">${batteryId}</td>
                    <td style="padding: 10px; color: #94a3b8;">${chemistry}</td>
                    <td style="padding: 10px;">${readings[readings.length - 1].cycle_count}</td>
                    <td style="padding: 10px; font-weight: 600; color: #38bdf8;">${data.rul_percent.toFixed(1)}%</td>
                    <td style="padding: 10px; font-weight: 600; color: ${statusColor};">${data.maintenance_status}</td>
                `;
                
                historyBody.insertBefore(newRow, historyBody.firstChild);

            } catch (e) {
                console.error(e);
                alert('Error calling inference API for sequence data.');
                document.getElementById('loading-spinner').classList.add('hidden');
                document.getElementById('results-placeholder').classList.remove('hidden');
            }
        };

        reader.readAsText(file);
    });

});
