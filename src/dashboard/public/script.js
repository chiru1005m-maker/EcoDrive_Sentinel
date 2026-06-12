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

        } catch (e) {
            console.error(e);
            alert('Error calling inference API. Make sure the API server is running on port 8000.');
            document.getElementById('loading-spinner').classList.add('hidden');
            document.getElementById('results-placeholder').classList.remove('hidden');
        }
    });


});
