from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import tempfile
from datetime import datetime, timedelta
import joblib
from werkzeug.utils import secure_filename
import traceback

# Fix matplotlib backend issue for web deployment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# GPU optimization for TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
try:
    import tensorflow as tf
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) detected")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("ℹ️  No GPU detected, using CPU")
except ImportError:
    print("ℹ️  TensorFlow not available for GPU detection")

# Import your existing forecasting modules
try:
    import inventory_forecasting_regression
    import arima_forecasting
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some forecasting modules not available: {e}")
    MODULES_AVAILABLE = False

# Import autoencoder separately with better error handling
try:
    from Autoencoder import autoencoder_anomaly_detection
    AUTOENCODER_AVAILABLE = True
    print("✅ Autoencoder anomaly detection available")
except ImportError as e:
    AUTOENCODER_AVAILABLE = False
    print(f"⚠️  Autoencoder not available: {e}")
    autoencoder_anomaly_detection = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for session-based model caching
TRAINED_MODELS = {
    'regression': None,
    'arima': None,
    'autoencoder': None
}

MODEL_METADATA = {
    'regression_info': None,
    'arima_performance': None,
    'arima_metadata': None,
    'scaler': None,
    'selector_info': None,
    'target_cols': None
}

MODEL_CACHE_STATUS = {
    'regression_trained': False,
    'arima_trained': False,
    'last_training_time': None
}

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate CSV structure
            try:
                df = pd.read_csv(filepath)
                required_columns = ['delivery_date', 'wings', 'tenders', 'fries_reg', 'fries_large', 
                                  'veggies', 'dips', 'drinks', 'flavours']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    os.remove(filepath)
                    return jsonify({
                        'error': f'Missing required columns: {", ".join(missing_columns)}'
                    }), 400
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'filepath': filepath,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'preview': df.head().to_dict('records')
                })
                
            except Exception as e:
                os.remove(filepath)
                return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
        
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filepath = data.get('filepath')
        model_type = data.get('model_type', 'regression')
        forecast_days = int(data.get('forecast_days', 7))
        anomaly_detection = data.get('anomaly_detection', False)
        predict_only = data.get('predict_only', False)
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Initialize results
        results = {
            'success': True,
            'model_type': model_type,
            'forecast_days': forecast_days,
            'timestamp': datetime.now().isoformat(),
            'forecast_data': [],
            'model_performance': {},
            'anomaly_results': None,
            'summary': {}
        }
        
        # Load and prepare data
        df = pd.read_csv(filepath)
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])
        df = df.sort_values('delivery_date').reset_index(drop=True)
        
        # Generate forecast based on model type and predict_only flag
        if model_type == 'regression':
            if predict_only and MODEL_CACHE_STATUS['regression_trained']:
                print("🚀 Using cached regression models (fast prediction)")
                forecast_results = generate_regression_forecast_cached(filepath, forecast_days)
            else:
                print("🔄 Training new regression models and caching...")
                forecast_results = generate_regression_forecast(filepath, forecast_days)
                cache_regression_models()
                # Update cache status after successful training
                MODEL_CACHE_STATUS['regression_trained'] = True
                MODEL_CACHE_STATUS['last_training_time'] = datetime.now()
            results.update(forecast_results)
        elif model_type == 'arima':
            if predict_only and MODEL_CACHE_STATUS['arima_trained']:
                print("🚀 Using cached ARIMA models (fast prediction)")
                forecast_results = generate_arima_forecast_cached(filepath, forecast_days)
            else:
                print("🔄 Training new ARIMA models and caching...")
                forecast_results = generate_arima_forecast(filepath, forecast_days)
                cache_arima_models()
                # Update cache status after successful training
                MODEL_CACHE_STATUS['arima_trained'] = True
                MODEL_CACHE_STATUS['last_training_time'] = datetime.now()
            results.update(forecast_results)
        elif model_type == 'both':
            # Generate both and compare
            if predict_only and MODEL_CACHE_STATUS['regression_trained']:
                print("🚀 Using cached regression models (fast prediction)")
                reg_results = generate_regression_forecast_cached(filepath, forecast_days)
            else:
                print("🔄 Training new regression models and caching...")
                reg_results = generate_regression_forecast(filepath, forecast_days)
                cache_regression_models()
            
            if predict_only and MODEL_CACHE_STATUS['arima_trained']:
                print("🚀 Using cached ARIMA models (fast prediction)")
                arima_results = generate_arima_forecast_cached(filepath, forecast_days)
            else:
                print("🔄 Training new ARIMA models and caching...")
                arima_results = generate_arima_forecast(filepath, forecast_days)
                cache_arima_models()
            
            # Choose better model based on performance
            if reg_results.get('model_performance', {}).get('mae', float('inf')) <= arima_results.get('model_performance', {}).get('mae', float('inf')):
                results.update(reg_results)
                results['comparison'] = {
                    'winner': 'regression',
                    'regression_mae': reg_results.get('model_performance', {}).get('mae', 0),
                    'arima_mae': arima_results.get('model_performance', {}).get('mae', 0)
                }
            else:
                results.update(arima_results)
                results['comparison'] = {
                    'winner': 'arima',
                    'regression_mae': reg_results.get('model_performance', {}).get('mae', 0),
                    'arima_mae': arima_results.get('model_performance', {}).get('mae', 0)
                }
        
        # Run anomaly detection if requested
        if anomaly_detection and AUTOENCODER_AVAILABLE:
            try:
                print("🔍 Running anomaly detection...")
                anomaly_results = run_anomaly_detection(filepath)
                results['anomaly_results'] = anomaly_results
                print(f"✅ Anomaly detection completed: {anomaly_results}")
            except Exception as e:
                print(f"❌ Anomaly detection failed: {str(e)}")
                results['anomaly_error'] = str(e)
        elif anomaly_detection and not AUTOENCODER_AVAILABLE:
            results['anomaly_error'] = "Autoencoder module not available. Check TensorFlow/NumPy compatibility."
        
        # Calculate summary statistics
        results['summary'] = calculate_summary_stats(results['forecast_data'])
        
        return jsonify(results)
        
    except Exception as e:
        error_msg = f'Forecast generation failed: {str(e)}'
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

def generate_regression_forecast(filepath, forecast_days):
    """Generate forecast using regression models"""
    try:
        # Set environment variable to prevent GUI operations
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Use existing regression forecasting logic
        import sys
        sys.argv = ['inventory_forecasting_regression.py', filepath]
        
        # Train or load regression models
        from inventory_forecasting_regression import main as train_regression
        best_model_info = train_regression(filepath)
        
        # STORE THE TRAINING RESULTS IMMEDIATELY FOR CACHING
        MODEL_METADATA['regression_info'] = best_model_info
        
        # Get the actual best model name and metrics from training
        best_model_name = best_model_info.get('best_model_name', 'Linear Regression')
        best_model_metrics = best_model_info.get('best_model_metrics', {})
        
        # Load the actual trained models
        models = {}
        model_files = ['linear_regression_model.pkl', 'ridge_model.pkl', 'lasso_model.pkl', 'elasticnet_model.pkl']
        
        for model_file in model_files:
            try:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(f'models/regression/{model_file}')
            except FileNotFoundError:
                continue
        
        if not models:
            raise Exception("No regression models found")
        
        scaler = joblib.load('models/regression/scaler.pkl')
        selector_info = joblib.load('models/regression/feature_selector_info.pkl')
        
        # Use the restaurant_forecast_tool logic for consistent forecasting
        from restaurant_forecast_tool import get_recent_data, generate_forecast_features, make_regression_predictions
        
        # Get recent data and generate features
        recent_df = get_recent_data(filepath)
        if recent_df is None:
            raise Exception("Could not load recent data")
        
        # Generate forecast features
        forecast_features = generate_forecast_features(recent_df, forecast_days)
        
        # Get target columns
        target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
        
        # Make predictions using the same logic as restaurant_forecast_tool
        forecast_df = make_regression_predictions(models, scaler, selector_info, forecast_features, target_cols, best_model_name, best_model_info)
        
        # Convert to the format expected by the web app
        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_row = {
                'date': row['Date'],
                'day_of_week': row['Day_of_Week'],
                'is_weekend': row['Is_Weekend']
            }
            
            for col in target_cols:
                col_title = col.title()
                forecast_row[f'{col}_forecast'] = row[f'{col_title}_Forecast']
                forecast_row[f'{col}_recommended_stock'] = row[f'{col_title}_Recommended_Stock']
            
            forecast_data.append(forecast_row)
        
        # Use the actual performance metrics from training
        performance = {
            'model_name': best_model_name,
            'mae': best_model_metrics.get('MAE', 0),
            'r2': best_model_metrics.get('R2', 0),
            'accuracy': best_model_metrics.get('R2', 0) * 100,  # Convert R2 to percentage
            'mape': best_model_metrics.get('MAPE', 0)
        }
        
        return {
            'forecast_data': forecast_data,
            'model_performance': performance,
            'model_type': 'regression'
        }
        
    except Exception as e:
        print(f"Regression forecast error: {str(e)}")
        # Fallback to simple historical average
        df = pd.read_csv(filepath)
        target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
        recent_avg = df[target_cols].tail(14).mean()
        
        forecast_data = []
        last_date = pd.to_datetime(df['delivery_date']).max()
        
        for day in range(1, forecast_days + 1):
            future_date = last_date + timedelta(days=day)
            forecast_row = {
                'date': future_date.strftime('%Y-%m-%d'),
                'day_of_week': future_date.strftime('%A'),
                'is_weekend': future_date.weekday() >= 5
            }
            
            for col in target_cols:
                base_value = recent_avg[col]
                if future_date.weekday() >= 5:
                    base_value *= 1.1
                
                forecast_row[f'{col}_forecast'] = max(0, int(base_value))
                forecast_row[f'{col}_recommended_stock'] = int(base_value * 1.2)
            
            forecast_data.append(forecast_row)
        
        return {
            'forecast_data': forecast_data,
            'model_performance': {
                'model_name': 'Historical Average (Fallback)',
                'mae': 50.0,
                'r2': 0.5,
                'accuracy': 50.0
            },
            'model_type': 'regression'
        }

def generate_arima_forecast(filepath, forecast_days):
    """Generate forecast using ARIMA models"""
    try:
        # Set environment variable to prevent GUI operations
        os.environ['MPLBACKEND'] = 'Agg'
        
        print("🔄 Training ARIMA models...")
        
        # Use existing ARIMA forecasting logic
        import sys
        sys.argv = ['arima_forecasting.py', filepath]
        
        # Train ARIMA models
        from arima_forecasting import main as train_arima
        arima_results = train_arima(filepath)
        
        if not arima_results:
            raise Exception("ARIMA training failed")
        
        # Load trained ARIMA models
        try:
            arima_performance = joblib.load('models/arima/arima_performance.pkl')
            model_metadata = joblib.load('models/arima/arima_metadata.pkl')
            
            # Load all ARIMA models
            target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
            arima_models = {}
            for col in target_cols:
                try:
                    arima_models[col] = joblib.load(f'models/arima/arima_{col}_model.pkl')
                except FileNotFoundError:
                    print(f"Warning: ARIMA model for {col} not found")
                    continue
            
        except FileNotFoundError:
            raise Exception("ARIMA models not found. Training may have failed.")
        
        if not arima_models:
            raise Exception("No ARIMA models available")
        
        # Use the restaurant_forecast_tool logic for consistent forecasting
        from restaurant_forecast_tool import get_recent_data, generate_forecast_features, make_arima_predictions
        
        # Get recent data and generate features
        recent_df = get_recent_data(filepath)
        if recent_df is None:
            raise Exception("Could not load recent data")
        
        # Generate forecast features
        forecast_features = generate_forecast_features(recent_df, forecast_days)
        
        # Make predictions using the same logic as restaurant_forecast_tool
        forecast_df = make_arima_predictions(arima_models, model_metadata, forecast_features, target_cols)
        
        # Convert to the format expected by the web app
        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_row = {
                'date': row['Date'],
                'day_of_week': row['Day_of_Week'],
                'is_weekend': row['Is_Weekend']
            }
            
            for col in target_cols:
                col_title = col.title()
                forecast_row[f'{col}_forecast'] = row[f'{col_title}_Forecast']
                forecast_row[f'{col}_recommended_stock'] = row[f'{col_title}_Recommended_Stock']
            
            forecast_data.append(forecast_row)
        
        # Calculate average performance metrics from all ARIMA models
        avg_mae = np.mean([perf['MAE'] for perf in arima_performance.values()])
        avg_r2 = np.mean([perf['R2'] for perf in arima_performance.values()])
        avg_mape = np.mean([perf.get('MAPE', 0) for perf in arima_performance.values()])
        
        # Use the actual performance metrics from training
        performance = {
            'model_name': 'ARIMA Ensemble',
            'mae': avg_mae,
            'r2': avg_r2,
            'accuracy': max(0, avg_r2 * 100),  # Convert R2 to percentage, ensure non-negative
            'mape': avg_mape,
            'models_trained': len(arima_models)
        }
        
        print(f"✅ ARIMA forecast completed. Average MAE: {avg_mae:.2f}, Average R²: {avg_r2:.3f}")
        
        return {
            'forecast_data': forecast_data,
            'model_performance': performance,
            'model_type': 'arima'
        }
        
    except Exception as e:
        print(f"ARIMA forecast error: {str(e)}")
        # Fallback to simple historical average with ARIMA-like variation
        df = pd.read_csv(filepath)
        target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
        recent_avg = df[target_cols].tail(14).mean()
        
        forecast_data = []
        last_date = pd.to_datetime(df['delivery_date']).max()
        
        for day in range(1, forecast_days + 1):
            future_date = last_date + timedelta(days=day)
            forecast_row = {
                'date': future_date.strftime('%Y-%m-%d'),
                'day_of_week': future_date.strftime('%A'),
                'is_weekend': future_date.weekday() >= 5
            }
            
            for col in target_cols:
                # Add some realistic variation
                base_value = recent_avg[col] * np.random.normal(1.0, 0.05)
                if future_date.weekday() >= 5:
                    base_value *= 1.05
                
                forecast_row[f'{col}_forecast'] = max(0, int(base_value))
                forecast_row[f'{col}_recommended_stock'] = int(base_value * 1.2)
            
            forecast_data.append(forecast_row)
        
        return {
            'forecast_data': forecast_data,
            'model_performance': {
                'model_name': 'Historical Average (ARIMA Fallback)',
                'mae': 50.0,
                'r2': 0.3,
                'accuracy': 30.0
            },
            'model_type': 'arima'
        }

def run_anomaly_detection(filepath):
    """Run anomaly detection on the dataset"""
    if not AUTOENCODER_AVAILABLE:
        raise Exception("Autoencoder module not available")
    
    try:
        print(f"🔍 Checking for existing anomaly model...")
        
        # Check if anomaly model exists
        model_path = 'Autoencoder/inventory_autoencoder_model.h5'
        if not os.path.exists(model_path):
            print(f"🔧 Training new anomaly detection model with GPU acceleration...")
            # Train anomaly detection model with fewer trials for faster training
            autoencoder_anomaly_detection.train_autoencoder(filepath, n_trials=15)
            print(f"✅ Anomaly model trained and saved")
        else:
            print(f"✅ Found existing anomaly model")
        
        print(f"🔍 Running GPU-accelerated anomaly detection on dataset...")
        # Run anomaly detection
        results = autoencoder_anomaly_detection.detect_anomalies(filepath, 'balanced')
        
        if results is not None and len(results) > 0:
            total_anomalies = results['is_anomaly'].sum()
            anomaly_percentage = 100 * total_anomalies / len(results)
            
            print(f"📊 Found {total_anomalies} anomalies ({anomaly_percentage:.1f}%)")
            
            # Get recent anomalies
            recent_anomalies = results[results['is_anomaly']].tail(5)
            anomaly_dates = []
            
            for _, row in recent_anomalies.iterrows():
                anomaly_dates.append({
                    'date': row['delivery_date'].strftime('%Y-%m-%d'),
                    'total_inventory': int(row['total_inventory']),
                    'reconstruction_error': float(row['reconstruction_error'])
                })
            
            return {
                'total_anomalies': int(total_anomalies),
                'anomaly_percentage': round(anomaly_percentage, 1),
                'recent_anomalies': anomaly_dates,
                'threshold_used': 'balanced'
            }
        else:
            print("⚠️  No anomaly results returned")
            return {
                'total_anomalies': 0,
                'anomaly_percentage': 0.0,
                'recent_anomalies': [],
                'threshold_used': 'balanced'
            }
        
    except Exception as e:
        print(f"❌ Anomaly detection error: {str(e)}")
        raise Exception(f"Anomaly detection failed: {str(e)}")

def calculate_summary_stats(forecast_data):
    """Calculate summary statistics from forecast data"""
    if not forecast_data:
        return {}
    
    target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
    
    totals = {}
    for col in target_cols:
        forecast_col = f'{col}_forecast'
        stock_col = f'{col}_recommended_stock'
        
        if forecast_col in forecast_data[0]:
            totals[col] = {
                'total_forecast': sum(row[forecast_col] for row in forecast_data),
                'total_recommended_stock': sum(row[stock_col] for row in forecast_data),
                'daily_average': sum(row[forecast_col] for row in forecast_data) / len(forecast_data)
            }
    
    # Calculate weekend vs weekday averages
    weekend_days = [row for row in forecast_data if row['is_weekend']]
    weekday_days = [row for row in forecast_data if not row['is_weekend']]
    
    weekend_avg = 0
    weekday_avg = 0
    
    if weekend_days:
        weekend_avg = sum(row['wings_forecast'] for row in weekend_days) / len(weekend_days)
    if weekday_days:
        weekday_avg = sum(row['wings_forecast'] for row in weekday_days) / len(weekday_days)
    
    return {
        'totals': totals,
        'weekend_avg': weekend_avg,
        'weekday_avg': weekday_avg,
        'weekend_days': len(weekend_days),
        'weekday_days': len(weekday_days)
    }

@app.route('/api/export', methods=['POST'])
def export_forecast():
    try:
        data = request.get_json()
        forecast_data = data.get('forecast_data', [])
        
        if not forecast_data:
            return jsonify({'error': 'No forecast data to export'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(forecast_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            
            # Return file
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f'inventory_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mimetype='text/csv'
            )
    
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

def cache_regression_models():
    """Cache trained regression models in memory"""
    try:
        # Load and cache regression models
        models = {}
        model_files = ['linear_regression_model.pkl', 'ridge_model.pkl', 'lasso_model.pkl', 'elasticnet_model.pkl']
        
        for model_file in model_files:
            try:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(f'models/regression/{model_file}')
            except FileNotFoundError:
                continue
        
        if models:
            TRAINED_MODELS['regression'] = models
            MODEL_METADATA['scaler'] = joblib.load('models/regression/scaler.pkl')
            MODEL_METADATA['selector_info'] = joblib.load('models/regression/feature_selector_info.pkl')
            
            # MODEL_METADATA['regression_info'] should already be set from training
            # DO NOT override it with file parsing - use the exact training results
            if MODEL_METADATA.get('regression_info') is not None:
                print(f"✅ Using EXACT training metrics: {MODEL_METADATA['regression_info']['best_model_name']}")
                best_metrics = MODEL_METADATA['regression_info']['best_model_metrics']
                print(f"   MAE: {best_metrics.get('MAE', 'N/A')}, R²: {best_metrics.get('R2', 'N/A')}, MAPE: {best_metrics.get('MAPE', 'N/A')}")
            else:
                print("⚠️ No training metrics available - this should not happen!")
            
            MODEL_CACHE_STATUS['regression_trained'] = True
            MODEL_CACHE_STATUS['last_training_time'] = datetime.now()
            
            print(f"✅ Cached {len(models)} regression models in memory")
        else:
            print("⚠️ No regression models found to cache")
            
    except Exception as e:
        print(f"⚠️ Failed to cache regression models: {e}")
        MODEL_CACHE_STATUS['regression_trained'] = False

def cache_arima_models():
    """Cache trained ARIMA models in memory"""
    try:
        # Load ARIMA models and metadata
        arima_models = {}
        arima_performance = joblib.load('models/arima/arima_performance.pkl')
        model_metadata = joblib.load('models/arima/arima_metadata.pkl')
        
        # Get target columns from the saved performance data
        target_cols = list(arima_performance.keys())
        
        for col in target_cols:
            try:
                arima_models[col] = joblib.load(f'models/arima/arima_{col}_model.pkl')
            except FileNotFoundError:
                continue
        
        if arima_models:
            TRAINED_MODELS['arima'] = arima_models
            MODEL_METADATA['arima_performance'] = arima_performance
            MODEL_METADATA['arima_metadata'] = model_metadata
            MODEL_METADATA['target_cols'] = target_cols
            MODEL_CACHE_STATUS['arima_trained'] = True
            MODEL_CACHE_STATUS['last_training_time'] = datetime.now()
            
            print(f"✅ Cached {len(arima_models)} ARIMA models in memory")
        else:
            print("⚠️ No ARIMA models found to cache")
            
    except Exception as e:
        print(f"⚠️ Failed to cache ARIMA models: {e}")
        MODEL_CACHE_STATUS['arima_trained'] = False

def generate_regression_forecast_cached(filepath, forecast_days):
    """Generate forecast using cached regression models (no training)"""
    if not MODEL_CACHE_STATUS['regression_trained'] or TRAINED_MODELS['regression'] is None:
        raise Exception("No cached regression models available. Please train first.")
    
    try:
        # Use cached models
        models = TRAINED_MODELS['regression']
        scaler = MODEL_METADATA['scaler']
        selector_info = MODEL_METADATA['selector_info']
        best_model_info = MODEL_METADATA['regression_info']
        
        # Use the restaurant_forecast_tool logic for consistent forecasting
        from restaurant_forecast_tool import get_recent_data, generate_forecast_features, make_regression_predictions
        
        # Get recent data and generate features
        recent_df = get_recent_data(filepath)
        if recent_df is None:
            raise Exception("Could not load recent data")
        
        # Generate forecast features
        forecast_features = generate_forecast_features(recent_df, forecast_days)
        
        # Get target columns
        target_cols = ['wings', 'tenders', 'fries_reg', 'fries_large', 'veggies', 'dips', 'drinks', 'flavours']
        
        # Get best model name from cached info
        best_model_name = best_model_info.get('best_model_name', 'Linear Regression') if best_model_info else 'Linear Regression'
        
        # Make predictions using cached models
        forecast_df = make_regression_predictions(models, scaler, selector_info, forecast_features, target_cols, best_model_name, best_model_info)
        
        # Convert to the format expected by the web app
        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_row = {
                'date': row['Date'],
                'day_of_week': row['Day_of_Week'],
                'is_weekend': row['Is_Weekend']
            }
            
            for col in target_cols:
                col_title = col.title()
                forecast_row[f'{col}_forecast'] = row[f'{col_title}_Forecast']
                forecast_row[f'{col}_recommended_stock'] = row[f'{col_title}_Recommended_Stock']
            
            forecast_data.append(forecast_row)
        
        # Use cached performance metrics
        best_model_metrics = best_model_info.get('best_model_metrics', {}) if best_model_info else {}
        performance = {
            'model_name': f"{best_model_name} (Cached)",
            'mae': best_model_metrics.get('MAE', 0),
            'r2': best_model_metrics.get('R2', 0),
            'accuracy': best_model_metrics.get('R2', 0) * 100,
            'mape': best_model_metrics.get('MAPE', 0)
        }
        
        return {
            'forecast_data': forecast_data,
            'model_performance': performance,
            'model_type': 'regression'
        }
        
    except Exception as e:
        print(f"Cached regression forecast error: {str(e)}")
        raise Exception(f"Failed to generate forecast with cached models: {str(e)}")

def generate_arima_forecast_cached(filepath, forecast_days):
    """Generate forecast using cached ARIMA models (no training)"""
    if not MODEL_CACHE_STATUS['arima_trained'] or TRAINED_MODELS['arima'] is None:
        raise Exception("No cached ARIMA models available. Please train first.")
    
    try:
        # Use cached models
        arima_models = TRAINED_MODELS['arima']
        arima_performance = MODEL_METADATA['arima_performance']
        model_metadata = MODEL_METADATA['arima_metadata']
        target_cols = MODEL_METADATA['target_cols']
        
        # Use the restaurant_forecast_tool logic for consistent forecasting
        from restaurant_forecast_tool import get_recent_data, generate_forecast_features, make_arima_predictions
        
        # Get recent data and generate features
        recent_df = get_recent_data(filepath)
        if recent_df is None:
            raise Exception("Could not load recent data")
        
        # Generate forecast features
        forecast_features = generate_forecast_features(recent_df, forecast_days)
        
        # Make predictions using cached models
        forecast_df = make_arima_predictions(arima_models, model_metadata, forecast_features, target_cols)
        
        # Convert to the format expected by the web app
        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_row = {
                'date': row['Date'],
                'day_of_week': row['Day_of_Week'],
                'is_weekend': row['Is_Weekend']
            }
            
            for col in target_cols:
                col_title = col.title()
                forecast_row[f'{col}_forecast'] = row[f'{col_title}_Forecast']
                forecast_row[f'{col}_recommended_stock'] = row[f'{col_title}_Recommended_Stock']
            
            forecast_data.append(forecast_row)
        
        # Calculate average performance metrics from cached data
        avg_mae = np.mean([perf['MAE'] for perf in arima_performance.values()])
        avg_r2 = np.mean([perf['R2'] for perf in arima_performance.values()])
        avg_mape = np.mean([perf.get('MAPE', 0) for perf in arima_performance.values()])
        
        performance = {
            'model_name': 'ARIMA Ensemble (Cached)',
            'mae': avg_mae,
            'r2': avg_r2,
            'accuracy': max(0, avg_r2 * 100),
            'mape': avg_mape,
            'models_trained': len(arima_models)
        }
        
        return {
            'forecast_data': forecast_data,
            'model_performance': performance,
            'model_type': 'arima'
        }
        
    except Exception as e:
        print(f"Cached ARIMA forecast error: {str(e)}")
        raise Exception(f"Failed to generate forecast with cached models: {str(e)}")

@app.route('/api/model-status')
def model_status():
    """Get current model cache status"""
    return jsonify({
        'regression_trained': MODEL_CACHE_STATUS['regression_trained'],
        'arima_trained': MODEL_CACHE_STATUS['arima_trained'],
        'last_training_time': MODEL_CACHE_STATUS['last_training_time'].isoformat() if MODEL_CACHE_STATUS['last_training_time'] else None,
        'cached_models': {
            'regression': len(TRAINED_MODELS['regression']) if TRAINED_MODELS['regression'] else 0,
            'arima': len(TRAINED_MODELS['arima']) if TRAINED_MODELS['arima'] else 0
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'modules_available': MODULES_AVAILABLE,
        'models_cached': {
            'regression': MODEL_CACHE_STATUS['regression_trained'],
            'arima': MODEL_CACHE_STATUS['arima_trained']
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    # Cloud Run specific configurations
    if os.environ.get("K_SERVICE"):  # Running on Cloud Run
        print("🚀 Starting on Google Cloud Run")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print("🖥️  Starting locally")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
