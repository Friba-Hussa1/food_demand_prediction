import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import seaborn as sns
warnings.filterwarnings('ignore')

def create_directories():
    """Create directories for ARIMA models and results"""
    os.makedirs('models/arima', exist_ok=True)
    os.makedirs('results/arima', exist_ok=True)
    os.makedirs('results/arima/plots', exist_ok=True)

def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset for ARIMA modeling with enhanced features"""
    df = pd.read_csv(dataset_path)
    df = df.sort_values("delivery_date").reset_index(drop=True)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    df.set_index("delivery_date", inplace=True)
    
    # Fill any missing values with forward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Enhanced external regressors for better ARIMA performance
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)  # Friday effect
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)  # Monday effect
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_month'] = df.index.day
    
    # Seasonal indicators
    df['is_month_start'] = (df.index.day <= 5).astype(int)
    df['is_month_end'] = (df.index.day >= 25).astype(int)
    
    # Cyclical encoding for better seasonal capture
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print("Dataset loaded and prepared for ARIMA modeling with enhanced features")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def check_stationarity(series, title):
    """Check if a time series is stationary using ADF test"""
    result = adfuller(series.dropna())
    
    print(f'\nStationarity Test for {title}:')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✅ Series is stationary")
        return True
    else:
        print("❌ Series is not stationary")
        return False

def make_stationary(series, max_diff=2):
    """Make a time series stationary through differencing"""
    original_series = series.copy()
    diff_order = 0
    
    for i in range(max_diff + 1):
        if check_stationarity(series, f"Differenced {i} times"):
            diff_order = i
            break
        if i < max_diff:
            series = series.diff().dropna()
    
    return series, diff_order

def preprocess_series(series):
    """Enhanced preprocessing with trend detection and seasonal decomposition"""
    
    # Calculate series statistics for adaptive preprocessing
    mean_val = series.mean()
    std_val = series.std()
    cv = std_val / mean_val if mean_val > 0 else 0
    scale = mean_val
    
    # Outlier removal with IQR method
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    series_clean = series.clip(lower=lower_bound, upper=upper_bound)
    
    # Apply log transformation for high variance series to stabilize variance
    log_transformed = False
    if cv > 0.3 and scale > 100:  # More selective log transformation
        series_clean = np.log1p(series_clean)
        log_transformed = True
    
    # Detect and handle trend
    if len(series_clean) > 14:
        # Simple trend detection using linear regression slope
        x = np.arange(len(series_clean))
        slope = np.polyfit(x, series_clean, 1)[0]
        trend_strength = abs(slope) / (std_val + 1e-8)
        
        # If strong trend detected, apply light smoothing
        if trend_strength > 0.1:
            series_clean = series_clean.rolling(window=3, center=True).mean().fillna(series_clean)
    
    return series_clean, log_transformed

def auto_arima_grid_search(series, exog=None, max_p=2, max_d=1, max_q=2, seasonal=False):
    """Fast ARIMA search with minimal parameters for speed"""
    
    print(f"\n🔍 Performing fast ARIMA search...")
    
    best_mae = float('inf')
    best_params = None
    best_model = None
    best_model_type = 'ARIMA'
    results = []
    
    # Simple preprocessing
    series_processed, log_transformed = preprocess_series(series)
    
    # Simple train/test split
    train_size = int(len(series_processed) * 0.85)
    train_series = series_processed.iloc[:train_size]
    val_series = series_processed.iloc[train_size:]
    
    if exog is not None:
        train_exog = exog.iloc[:train_size]
        val_exog = exog.iloc[train_size:]
    else:
        train_exog = val_exog = None
    
    # Try only the most common ARIMA models for speed
    arima_combinations = [
        (0, 1, 1), (1, 1, 1), (1, 0, 1), (2, 1, 1), (1, 1, 2)
    ]
    
    for p, d, q in arima_combinations:
        try:
            model = ARIMA(train_series, exog=train_exog, order=(p, d, q))
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=len(val_series), exog=val_exog)
            
            # Transform back if log was applied
            if log_transformed:
                forecast = np.expm1(forecast)
                val_actual = np.expm1(val_series)
            else:
                val_actual = val_series
            
            val_mae = mean_absolute_error(val_actual, forecast)
            
            results.append({
                'model_type': 'ARIMA',
                'p': p, 'd': d, 'q': q, 's': 0,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'Val_MAE': val_mae,
                'log_transformed': log_transformed
            })
            
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = (p, d, q)
                best_model_type = 'ARIMA'
                # Refit on full series
                full_model = ARIMA(series_processed, exog=exog, order=(p, d, q))
                best_model = full_model.fit()
                
        except Exception as e:
            continue
    
    print(f"✅ Best model: {best_model_type} with params: {best_params} (Val MAE: {best_mae:.2f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Val_MAE').reset_index(drop=True)
        results_df.to_csv('results/arima/arima_grid_search_results.csv', index=False)
    
    return best_model, best_params, results_df, best_model_type, log_transformed

def train_arima_models(df):
    """Train fast ARIMA models for each target variable"""
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    arima_models = {}
    arima_params = {}
    arima_performance = {}
    model_metadata = {}
    
    # Use last 25% for testing (consistent with regression models)
    split_index = int(len(df) * 0.75)
    
    # Enhanced external regressors for better performance
    exog_cols = ['day_of_week', 'is_weekend', 'is_friday', 'is_monday', 'month', 
                 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'is_month_start']
    exog_data = df[exog_cols]
    
    print("Training ultra-fast ARIMA models for each inventory item...")
    print("=" * 60)
    
    for col in target_cols:
        print(f"\n📊 Training fast ARIMA for {col.upper()}")
        print("-" * 40)
        
        series = df[col].dropna()
        
        # Align exog data with series
        exog_aligned = exog_data.loc[series.index]
        
        train_series = series.iloc[:split_index]
        test_series = series.iloc[split_index:]
        train_exog = exog_aligned.iloc[:split_index]
        test_exog = exog_aligned.iloc[split_index:]
        
        try:
            # Fast grid search - no seasonal models for speed
            best_model, best_params, grid_results, model_type, log_transformed = auto_arima_grid_search(
                train_series, exog=train_exog, seasonal=False
            )
            
            # Make predictions
            forecast_steps = len(test_series)
            forecast = best_model.forecast(steps=forecast_steps, exog=test_exog)
            
            # Transform back if log transformation was applied
            if log_transformed:
                forecast = np.expm1(forecast)
            
            # Ensure non-negative forecasts
            forecast = np.maximum(forecast, 0)
            
            # Calculate performance metrics including MAPE
            mae = mean_absolute_error(test_series, forecast)
            rmse = np.sqrt(mean_squared_error(test_series, forecast))
            r2 = r2_score(test_series, forecast)
            
            # Calculate MAPE for better scale comparison
            def calculate_mape(y_true, y_pred):
                epsilon = 1e-8
                return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
            
            mape = calculate_mape(test_series.values, forecast)
            
            # Cap R² at reasonable values
            r2 = max(-1.0, min(r2, 0.99))
            
            arima_models[col] = best_model
            arima_params[col] = best_params
            model_metadata[col] = {
                'model_type': model_type,
                'log_transformed': log_transformed,
                'exog_cols': exog_cols
            }
            arima_performance[col] = {
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'R2': r2,
                'AIC': getattr(best_model, 'aic', np.nan),
                'BIC': getattr(best_model, 'bic', np.nan),
                'model_type': model_type
            }
            
            print(f"✅ {col}: {model_type}{best_params} - MAE: {mae:.2f}, MAPE: {mape:.1f}%, R²: {r2:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to train fast ARIMA for {col}: {str(e)}")
            continue
    
    return arima_models, arima_params, arima_performance, split_index, model_metadata

def evaluate_arima_models(df, arima_models, split_index, model_metadata):
    """Evaluate ARIMA models and create predictions"""
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    all_predictions = {}
    all_actuals = {}
    
    # Enhanced external regressors for test period
    exog_cols = ['day_of_week', 'is_weekend', 'is_friday', 'is_monday', 'month', 
                 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'is_month_start']
    exog_data = df[exog_cols]
    
    for col in target_cols:
        if col in arima_models:
            series = df[col].dropna()
            test_series = series.iloc[split_index:]
            
            try:
                # Make predictions - all models are ARIMA type now
                forecast_steps = len(test_series)
                test_exog = exog_data.iloc[split_index:split_index + forecast_steps]
                forecast = arima_models[col].forecast(steps=forecast_steps, exog=test_exog)
                
                all_predictions[col] = forecast.values if hasattr(forecast, 'values') else forecast
                all_actuals[col] = test_series.values
                
            except Exception as e:
                print(f"⚠️ Evaluation failed for {col}: {str(e)}")
                continue
    
    return all_predictions, all_actuals

def create_arima_visualizations(df, arima_models, arima_performance, all_predictions, all_actuals, split_index):
    """Create comprehensive visualizations for ARIMA models"""
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    # 1. Model Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(arima_performance.keys())
    mae_scores = [arima_performance[model]['MAE'] for model in models]
    r2_scores = [arima_performance[model]['R2'] for model in models]
    aic_scores = [arima_performance[model]['AIC'] for model in models]
    
    # MAE comparison
    ax1.bar(models, mae_scores, color='lightblue', alpha=0.7)
    ax1.set_title('ARIMA Model MAE by Item')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(mae_scores):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # R² comparison
    ax2.bar(models, r2_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('ARIMA Model R² by Item')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(r2_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # AIC comparison
    ax3.bar(models, aic_scores, color='lightgreen', alpha=0.7)
    ax3.set_title('ARIMA Model AIC by Item')
    ax3.set_ylabel('AIC')
    ax3.tick_params(axis='x', rotation=45)
    
    # Overall performance summary
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    ax4.text(0.1, 0.8, f'Average MAE: {avg_mae:.2f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Average R²: {avg_r2:.3f}', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f'Models trained: {len(models)}', fontsize=14, transform=ax4.transAxes)
    ax4.set_title('ARIMA Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/arima/plots/arima_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time Series Forecasts
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        if col in all_predictions:
            series = df[col].dropna()
            train_series = series.iloc[:split_index]
            test_series = series.iloc[split_index:]
            predictions = all_predictions[col]
            
            # Plot historical data
            axes[i].plot(train_series.index, train_series.values, label='Training Data', color='blue', alpha=0.7)
            axes[i].plot(test_series.index, test_series.values, label='Actual', color='green', linewidth=2)
            axes[i].plot(test_series.index, predictions, label='ARIMA Forecast', color='red', linewidth=2, linestyle='--')
            
            axes[i].set_title(f'{col.title()} - ARIMA Forecast')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Quantity')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/arima/plots/arima_forecasts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Analysis
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        if col in all_predictions:
            residuals = all_actuals[col] - all_predictions[col]
            
            axes[i].scatter(all_predictions[col], residuals, alpha=0.6)
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[i].set_xlabel(f'Predicted {col}')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'{col} - ARIMA Residuals')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/arima/plots/arima_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ARIMA visualizations saved to results/arima/plots/ directory:")
    print("- arima_performance_comparison.png: Model performance comparison")
    print("- arima_forecasts.png: Time series forecasts")
    print("- arima_residuals.png: Residual analysis")

def generate_arima_forecast(df, arima_models, model_metadata, forecast_days=7):
    """Generate future forecasts using fast ARIMA models"""
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    # Generate future dates
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Create enhanced future exogenous variables
    future_exog = pd.DataFrame(index=future_dates)
    future_exog['day_of_week'] = future_dates.dayofweek
    future_exog['is_weekend'] = (future_dates.dayofweek >= 5).astype(int)
    future_exog['is_friday'] = (future_dates.dayofweek == 4).astype(int)
    future_exog['is_monday'] = (future_dates.dayofweek == 0).astype(int)
    future_exog['month'] = future_dates.month
    future_exog['month_sin'] = np.sin(2 * np.pi * future_dates.month / 12)
    future_exog['month_cos'] = np.cos(2 * np.pi * future_dates.month / 12)
    future_exog['dow_sin'] = np.sin(2 * np.pi * future_dates.dayofweek / 7)
    future_exog['dow_cos'] = np.cos(2 * np.pi * future_dates.dayofweek / 7)
    future_exog['is_month_start'] = (future_dates.day <= 5).astype(int)
    
    forecast_df = pd.DataFrame()
    forecast_df['Date'] = future_dates.strftime('%Y-%m-%d')
    forecast_df['Day_of_Week'] = future_dates.day_name()
    forecast_df['Is_Weekend'] = (future_dates.dayofweek >= 5)
    
    # Generate forecasts for each item
    for col in target_cols:
        if col in arima_models:
            try:
                model = arima_models[col]
                metadata = model_metadata.get(col, {})
                model_type = metadata.get('model_type', 'ARIMA')
                log_transformed = metadata.get('log_transformed', False)
                
                # Generate forecast
                forecast = model.forecast(steps=forecast_days, exog=future_exog)
                
                # Transform back if log transformation was applied
                if log_transformed:
                    forecast = np.expm1(forecast)
                
                # Ensure non-negative forecasts
                forecast = np.maximum(forecast, 0)
                
                forecast_df[f'{col.title()}_Forecast'] = np.round(forecast).astype(int)
                # Add safety stock (20% buffer)
                forecast_df[f'{col.title()}_Recommended_Stock'] = np.round(forecast * 1.2).astype(int)
                
            except Exception as e:
                print(f"⚠️ Forecast failed for {col}, using historical average: {str(e)}")
                # Fallback to historical average
                avg_value = df[col].tail(30).mean()
                forecast_df[f'{col.title()}_Forecast'] = int(avg_value)
                forecast_df[f'{col.title()}_Recommended_Stock'] = int(avg_value * 1.2)
        else:
            # If ARIMA failed for this item, use historical average
            avg_value = df[col].tail(30).mean()
            forecast_df[f'{col.title()}_Forecast'] = int(avg_value)
            forecast_df[f'{col.title()}_Recommended_Stock'] = int(avg_value * 1.2)
    
    return forecast_df

def save_arima_models(arima_models, arima_params, arima_performance, model_metadata, forecast_df=None):
    """Save enhanced ARIMA models and results"""
    # Save models
    for col, model in arima_models.items():
        joblib.dump(model, f'models/arima/arima_{col}_model.pkl')
    
    # Save parameters, performance, and metadata
    joblib.dump(arima_params, 'models/arima/arima_parameters.pkl')
    joblib.dump(arima_performance, 'models/arima/arima_performance.pkl')
    joblib.dump(model_metadata, 'models/arima/arima_metadata.pkl')
    
    # Save enhanced performance summary - TRAINING RESULTS ONLY
    with open('results/arima/arima_performance_summary.txt', 'w') as f:
        f.write("ARIMA MODEL TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("⚠️  NOTE: These are TRAINING performance metrics only.\n")
        f.write("   For actual forecasts, use 'uv run restaurant_forecast_tool.py'\n\n")
        
        for col, perf in arima_performance.items():
            metadata = model_metadata.get(col, {})
            f.write(f"{col.upper()}:\n")
            f.write(f"  Model Type: {perf.get('model_type', 'ARIMA')}\n")
            f.write(f"  Parameters: {arima_params[col]}\n")
            f.write(f"  Log Transformed: {metadata.get('log_transformed', False)}\n")
            f.write(f"  External Regressors: {metadata.get('exog_cols', [])}\n")
            f.write(f"  MAE: {perf['MAE']:.2f}\n")
            f.write(f"  RMSE: {perf['RMSE']:.2f}\n")
            f.write(f"  R²: {perf['R2']:.3f}\n")
            if not np.isnan(perf['AIC']):
                f.write(f"  AIC: {perf['AIC']:.2f}\n")
            if not np.isnan(perf['BIC']):
                f.write(f"  BIC: {perf['BIC']:.2f}\n")
            f.write("\n")
        
        # Overall statistics
        avg_mae = np.mean([perf['MAE'] for perf in arima_performance.values()])
        avg_r2 = np.mean([perf['R2'] for perf in arima_performance.values()])
        
        # Model type distribution
        model_types = [perf.get('model_type', 'ARIMA') for perf in arima_performance.values()]
        type_counts = pd.Series(model_types).value_counts()
        
        f.write("OVERALL TRAINING PERFORMANCE:\n")
        f.write(f"  Average MAE: {avg_mae:.2f}\n")
        f.write(f"  Average R²: {avg_r2:.3f}\n")
        f.write(f"  Models trained: {len(arima_performance)}\n")
        f.write(f"\nMODEL TYPE DISTRIBUTION:\n")
        for model_type, count in type_counts.items():
            f.write(f"  {model_type}: {count}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("💡 FOR ACTUAL FORECASTS:\n")
        f.write("   Use: uv run restaurant_forecast_tool.py --dataset your_data.csv\n")
        f.write("   This will generate accurate forecasts using the trained models.\n")
    
    print("ARIMA training results saved to models/arima/ and results/arima/ directories")
    print("💡 For actual forecasts, use: uv run restaurant_forecast_tool.py --dataset your_data.csv")

def main(dataset_path=None):
    """Main function to run fast ARIMA forecasting pipeline"""
    if dataset_path is None:
        import sys
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            dataset_path = "data/inventory_delivery_forecast_data.csv"  # Default fallback
    
    print("Starting Ultra-Fast ARIMA Time Series Forecasting Pipeline")
    print("=" * 70)
    print(f"📊 Using dataset: {dataset_path}")
    
    # Create directories
    create_directories()
    
    # Load and prepare data
    df = load_and_prepare_data(dataset_path)
    
    # Train fast ARIMA models
    arima_models, arima_params, arima_performance, split_index, model_metadata = train_arima_models(df)
    
    if not arima_models:
        print("❌ No ARIMA models were successfully trained!")
        return None
    
    # Evaluate models
    all_predictions, all_actuals = evaluate_arima_models(df, arima_models, split_index, model_metadata)
    
    # Create visualizations
    create_arima_visualizations(df, arima_models, arima_performance, all_predictions, all_actuals, split_index)
    
    # Generate future forecast
    forecast_df = generate_arima_forecast(df, arima_models, model_metadata, forecast_days=7)
    
    # Save models and results
    save_arima_models(arima_models, arima_params, arima_performance, model_metadata, forecast_df)
    
    # Calculate overall performance for comparison
    overall_mae = np.mean([perf['MAE'] for perf in arima_performance.values()])
    overall_mape = np.mean([perf['MAPE'] for perf in arima_performance.values()])
    overall_r2 = np.mean([perf['R2'] for perf in arima_performance.values()])
    
    print(f"\n✅ Ultra-Fast ARIMA Pipeline completed successfully!")
    print(f"📊 Overall Performance:")
    print(f"   - Average MAE: {overall_mae:.2f}")
    print(f"   - Average MAPE: {overall_mape:.1f}%")
    print(f"   - Average R²: {overall_r2:.3f}")
    print(f"   - Models trained: {len(arima_models)}")
    
    return {
        'models': arima_models,
        'performance': arima_performance,
        'forecast': forecast_df,
        'overall_mae': overall_mae,
        'overall_mape': overall_mape,
        'overall_r2': overall_r2,
        'model_type': 'Ultra-Fast ARIMA',
        'metadata': model_metadata
    }

if __name__ == "__main__":
    main()
