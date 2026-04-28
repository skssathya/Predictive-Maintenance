const form = document.getElementById('predict-form');
const resultCard = document.getElementById('result-card');
const alertBox = document.getElementById('alert');
const riskLevel = document.getElementById('risk-level');
const failureProbability = document.getElementById('failure-probability');
const recommendation = document.getElementById('recommendation');
const healthStatus = document.getElementById('health-status');
const modelList = document.getElementById('model-list');

// Configure API endpoint (change this to your backend URL)
const API_BASE_URL = 'http://localhost:5000'; // Change for production

async function updateHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const json = await response.json();
        healthStatus.textContent = json.status === 'healthy' ? 'Healthy' : 'Unhealthy';
    } catch (error) {
        healthStatus.textContent = 'Unavailable';
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        const json = await response.json();
        modelList.innerHTML = json.available_models
            .map((model) => `<li>${model.replace('_', ' ').toUpperCase()}</li>`)
            .join('');
    } catch (error) {
        modelList.innerHTML = '<li>Unable to load models.</li>';
    }
}

function showAlert(message, type = 'error') {
    alertBox.textContent = message;
    alertBox.className = `alert ${type === 'error' ? 'alert-error' : 'alert-success'}`;
    alertBox.classList.remove('hidden');
}

function hideAlert() {
    alertBox.classList.add('hidden');
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideAlert();

    const formData = new FormData(form);
    const body = {
        vibration: parseFloat(formData.get('vibration')),
        temperature: parseFloat(formData.get('temperature')),
        run_time: parseFloat(formData.get('run_time')),
        motor_current: parseFloat(formData.get('motor_current'))
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });

        const json = await response.json();

        if (!response.ok) {
            showAlert(json.error || 'Prediction failed.');
            resultCard.classList.add('hidden');
            return;
        }

        riskLevel.textContent = json.risk_level;
        failureProbability.textContent = `${(json.failure_probability * 100).toFixed(1)}%`;
        recommendation.textContent = json.recommendation;
        resultCard.classList.remove('hidden');
    } catch (error) {
        showAlert('Unable to reach the prediction service.');
        resultCard.classList.add('hidden');
    }
});

updateHealth();
loadModels();
