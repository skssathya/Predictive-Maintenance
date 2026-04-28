"""Main Flask API application for predictive maintenance"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.utils.logger import setup_logger

logger = setup_logger('predictive_maintenance_api')

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)
CORS(app)

app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Predictive Maintenance API is running'
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON body'}), 400

        required_fields = ['vibration', 'temperature', 'run_time', 'motor_current']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields. Required: {required_fields}'
            }), 400

        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Field {field} must be a number'}), 400

        logger.info(f"Prediction request received: {data}")

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
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Predictive Maintenance API")
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
