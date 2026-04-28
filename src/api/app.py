"""Main Flask API application for predictive maintenance"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('predictive_maintenance_api')

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Predictive Maintenance API is running'
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a prediction for equipment failure risk.
    """
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['vibration', 'temperature', 'run_time', 'motor_current']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields. Required: {required_fields}'
            }), 400
        
        logger.info(f"Prediction request received: {data}")
        
        # Placeholder response (integrate actual model here)
        response = {
            'prediction': 0,
            'failure_probability': 0.35,
            'risk_level': 'medium',
            'recommendation': 'Schedule maintenance within 2-3 weeks'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    List available models.
    """
    models = {
        'available_models': [
            'random_forest',
            'xgboost',
            'neural_network',
            'svm',
            'isolation_forest'
        ]
    }
    return jsonify(models), 200

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    """
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Predictive Maintenance API")
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
