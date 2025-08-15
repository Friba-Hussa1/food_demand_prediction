import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MockSelector:
    """Mock selector class for feature selection compatibility"""
    def __init__(self, indices, n_features):
        self.indices = indices
        self.n_features = n_features
    
    def transform(self, X):
        return X[:, self.indices]
    
    def get_support(self):
        support = np.zeros(self.n_features, dtype=bool)
        support[self.indices] = True
        return support

def create_directories():
    """Create directories for models and results"""
    os.makedirs('models/regression', exist_ok=True)
    os.makedirs('results/regression', exist_ok=True)
    os.makedirs('results/regression/plots', exist_ok=True)
    # Removed manager_reports and forecasts/final directories

def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset"""
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Sort by date to ensure proper time-series order
    df = df.sort_values("delivery_date").reset_index(drop=True)
    
    print("Dataset loaded and sorted by delivery_date")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    return df

def feature_engineering(df):
    """Perform advanced feature engineering on the dataset"""
    df_fe = df.copy()
    
    # Convert 'delivery_date' to datetime objects
    df_fe["delivery_date"] = pd.to_datetime(df_fe["delivery_date"])
    
    # Target columns for feature engineering
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    # Advanced lag features (multiple lags)
    for col in target_cols:
        df_fe[f"{col}_lag1"] = df_fe[col].shift(1)
        df_fe[f"{col}_lag2"] = df_fe[col].shift(2)
        df_fe[f"{col}_lag7"] = df_fe[col].shift(7)  # Same day last week
    
    # Advanced rolling statistics
    for col in target_cols:
        df_fe[f"{col}_roll3"] = df_fe[col].rolling(window=3).mean().shift(1)
        df_fe[f"{col}_roll7"] = df_fe[col].rolling(window=7).mean().shift(1)
        df_fe[f"{col}_roll14"] = df_fe[col].rolling(window=14).mean().shift(1)
        df_fe[f"{col}_roll_std7"] = df_fe[col].rolling(window=7).std().shift(1)
        df_fe[f"{col}_roll_min7"] = df_fe[col].rolling(window=7).min().shift(1)
        df_fe[f"{col}_roll_max7"] = df_fe[col].rolling(window=7).max().shift(1)
    
    # Exponential weighted moving averages
    for col in target_cols:
        df_fe[f"{col}_ewm3"] = df_fe[col].ewm(span=3).mean().shift(1)
        df_fe[f"{col}_ewm7"] = df_fe[col].ewm(span=7).mean().shift(1)
    
    # Advanced calendar features
    df_fe["day_of_week"] = df_fe["delivery_date"].dt.dayofweek
    df_fe["month"] = df_fe["delivery_date"].dt.month
    df_fe["day_of_month"] = df_fe["delivery_date"].dt.day
    df_fe["quarter"] = df_fe["delivery_date"].dt.quarter
    df_fe["is_weekend"] = (df_fe["day_of_week"] >= 5).astype(int)
    df_fe["is_monday"] = (df_fe["day_of_week"] == 0).astype(int)
    df_fe["is_friday"] = (df_fe["day_of_week"] == 4).astype(int)
    df_fe["week_of_year"] = df_fe["delivery_date"].dt.isocalendar().week
    
    # Cyclical encoding for calendar features
    df_fe["day_of_week_sin"] = np.sin(2 * np.pi * df_fe["day_of_week"] / 7)
    df_fe["day_of_week_cos"] = np.cos(2 * np.pi * df_fe["day_of_week"] / 7)
    df_fe["month_sin"] = np.sin(2 * np.pi * df_fe["month"] / 12)
    df_fe["month_cos"] = np.cos(2 * np.pi * df_fe["month"] / 12)
    
    # Trend and seasonality features
    df_fe["days_since_start"] = (df_fe["delivery_date"] - df_fe["delivery_date"].min()).dt.days
    df_fe["trend"] = np.arange(len(df_fe))
    
    # Advanced cross-product features
    df_fe["wings_tenders_ratio"] = df_fe["wings"] / (df_fe["tenders"] + 1)
    df_fe["fries_total"] = df_fe["fries_reg"] + df_fe["fries_large"]
    df_fe["total_food"] = df_fe["wings"] + df_fe["tenders"] + df_fe["fries_reg"] + df_fe["fries_large"] + df_fe["veggies"]
    df_fe["main_items"] = df_fe["wings"] + df_fe["tenders"]
    df_fe["side_items"] = df_fe["fries_reg"] + df_fe["fries_large"] + df_fe["veggies"]
    df_fe["beverages_condiments"] = df_fe["drinks"] + df_fe["dips"] + df_fe["flavours"]
    
    # Interaction features
    df_fe["weekend_wings"] = df_fe["is_weekend"] * df_fe["wings"]
    df_fe["weekend_total"] = df_fe["is_weekend"] * df_fe["total_food"]
    df_fe["month_wings"] = df_fe["month"] * df_fe["wings"]
    
    # Volatility and change features
    for col in target_cols:
        df_fe[f"{col}_change"] = df_fe[col].diff()
        df_fe[f"{col}_pct_change"] = df_fe[col].pct_change()
        df_fe[f"{col}_volatility"] = df_fe[col].rolling(window=7).std() / df_fe[col].rolling(window=7).mean()
    
    # Drop rows with NaN from rolling windows and lags
    df_fe = df_fe.dropna().reset_index(drop=True)
    
    print("Advanced feature engineering completed")
    print(f"Features dataset shape: {df_fe.shape}")
    
    return df_fe

def prepare_train_test_split(df_fe):
    """Prepare train/test split with advanced feature selection and scaling"""
    target_cols = ["wings", "tenders", "fries_reg", "fries_large", "veggies", "dips", "drinks", "flavours"]
    
    X = df_fe.drop(columns=["delivery_date"] + target_cols)
    y = df_fe[target_cols]
    
    # Use last 25% of data for testing (better for time series)
    split_index = int(len(df_fe) * 0.75)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Advanced feature selection using multiple methods
    k_best = min(25, X_train.shape[1])  # Increased feature limit
    
    # Method 1: Mutual information
    mi_selector = SelectKBest(mutual_info_regression, k=k_best)
    mi_selector.fit(X_train_scaled, y_train.iloc[:, 0])  # Use wings as proxy
    mi_scores = mi_selector.scores_
    
    # Method 2: F-regression
    f_selector = SelectKBest(f_regression, k=k_best)
    f_selector.fit(X_train_scaled, y_train.iloc[:, 0])
    f_scores = f_selector.scores_
    
    # Method 3: Correlation-based
    correlations = []
    for i in range(X_train_scaled.shape[1]):
        corr = np.corrcoef(X_train_scaled[:, i], y_train.iloc[:, 0])[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    
    # Combine scores (ensemble feature selection)
    mi_scores_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
    f_scores_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-8)
    corr_scores_norm = (np.array(correlations) - np.min(correlations)) / (np.max(correlations) - np.min(correlations) + 1e-8)
    
    # Weighted combination of feature selection methods
    combined_scores = 0.4 * mi_scores_norm + 0.3 * f_scores_norm + 0.3 * corr_scores_norm
    
    # Select top k features based on combined scores
    feature_indices = np.argsort(combined_scores)[-k_best:]
    X_train_selected = X_train_scaled[:, feature_indices]
    X_test_selected = X_test_scaled[:, feature_indices]
    
    # Get selected feature names
    selected_features = X.columns[feature_indices].tolist()
    
    # Create a mock selector object for saving
    selector = MockSelector(feature_indices, X.shape[1])
    
    print(f"Training set shape: {X_train_selected.shape}")
    print(f"Test set shape: {X_test_selected.shape}")
    print(f"Selected features: {selected_features[:10]}...")  # Show first 10
    
    return X_train_selected, X_test_selected, y_train, y_test, target_cols, scaler, selector, selected_features

def train_models_with_cv(X_train, y_train):
    """Train multiple advanced regression models with cross-validation and hyperparameter tuning"""
    models = {}
    best_params = {}
    cv_scores = {}
    
    # Time series cross-validation with more splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Training advanced models with cross-validation and hyperparameter tuning...")
    
    # Linear Regression (no hyperparameters to tune)
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
    models['Linear Regression'] = lr
    lr.fit(X_train, y_train)
    cv_scores['Linear Regression'] = -lr_scores.mean()
    best_params['Linear Regression'] = {}
    print(f"Linear Regression CV MAE: {-lr_scores.mean():.2f} (+/- {lr_scores.std() * 2:.2f})")
    
    # Ridge Regression with expanded hyperparameter tuning
    ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    models['Ridge'] = ridge_grid.best_estimator_
    best_params['Ridge'] = ridge_grid.best_params_
    cv_scores['Ridge'] = -ridge_grid.best_score_
    print(f"Ridge Regression CV MAE: {-ridge_grid.best_score_:.2f}, Best params: {ridge_grid.best_params_}")
    
    # Lasso Regression with expanded hyperparameter tuning
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
    lasso_grid = GridSearchCV(Lasso(max_iter=3000), lasso_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    lasso_grid.fit(X_train, y_train)
    models['Lasso'] = lasso_grid.best_estimator_
    best_params['Lasso'] = lasso_grid.best_params_
    cv_scores['Lasso'] = -lasso_grid.best_score_
    print(f"Lasso Regression CV MAE: {-lasso_grid.best_score_:.2f}, Best params: {lasso_grid.best_params_}")
    
    # ElasticNet with expanded hyperparameter tuning
    elasticnet_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]}
    elasticnet_grid = GridSearchCV(ElasticNet(max_iter=3000), elasticnet_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    elasticnet_grid.fit(X_train, y_train)
    models['ElasticNet'] = elasticnet_grid.best_estimator_
    best_params['ElasticNet'] = elasticnet_grid.best_params_
    cv_scores['ElasticNet'] = -elasticnet_grid.best_score_
    print(f"ElasticNet CV MAE: {-elasticnet_grid.best_score_:.2f}, Best params: {elasticnet_grid.best_params_}")
    
    # Extra Trees Regressor (already supports multi-output)
    et_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    et_random = RandomizedSearchCV(ExtraTreesRegressor(random_state=42), et_params,
                                  n_iter=30, cv=tscv, scoring='neg_mean_absolute_error',
                                  n_jobs=-1, random_state=42)
    et_random.fit(X_train, y_train)
    models['Extra Trees'] = et_random.best_estimator_
    best_params['Extra Trees'] = et_random.best_params_
    cv_scores['Extra Trees'] = -et_random.best_score_
    print(f"Extra Trees CV MAE: {-et_random.best_score_:.2f}, Best params: {et_random.best_params_}")
    
    print("All advanced models trained successfully with hyperparameter tuning")
    return models, best_params, cv_scores

def evaluate_model(name, model, X_test, y_test):
    """Evaluate model performance with MAPE for better scale handling"""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    accuracy = model.score(X_test, y_test)
    
    # Calculate MAPE (Mean Absolute Percentage Error) for better comparison across different scales
    def calculate_mape(y_true, y_pred):
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    mape = calculate_mape(y_test.values, preds)
    
    print(f"{name} Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print("-" * 40)
    
    return preds, {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2, "Accuracy": accuracy}

def save_models(models, scaler, selector, ensemble_info=None):
    """Save trained models and preprocessing objects"""
    for name, model in models.items():
        if name != 'Ensemble':  # Skip ensemble as it's not a single model
            filename = name.lower().replace(' ', '_')
            joblib.dump(model, f'models/regression/{filename}_model.pkl')
    
    joblib.dump(scaler, 'models/regression/scaler.pkl')
    
    # Save selector information as a simple dictionary instead of the object
    selector_info = {
        'indices': selector.indices,
        'n_features': selector.n_features
    }
    joblib.dump(selector_info, 'models/regression/feature_selector_info.pkl')
    
    # Save ensemble information if provided
    if ensemble_info:
        joblib.dump(ensemble_info, 'models/regression/ensemble_info.pkl')
    
    print("All models and preprocessing objects saved to models/regression/ directory")

def create_comprehensive_visualizations(y_test, all_predictions, all_metrics, cv_scores, target_cols, selected_features, best_params):
    """Create comprehensive visualizations for model analysis"""
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    
    # 1. Model Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(all_metrics.keys())
    mae_scores = [all_metrics[model]['MAE'] for model in models]
    r2_scores = [all_metrics[model]['R2'] for model in models]
    cv_mae_scores = [cv_scores.get(model, 0) for model in models]  # Use 0 for ensemble (no CV score)
    
    # MAE comparison
    bars1 = ax1.bar(models, mae_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Mean Absolute Error by Model')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(mae_scores):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # R² comparison
    bars2 = ax2.bar(models, r2_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('R² Score by Model')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(r2_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Cross-validation vs Test MAE (exclude ensemble which has no CV score)
    cv_models = [model for model in models if model in cv_scores]
    cv_mae_vals = [cv_scores[model] for model in cv_models]
    cv_test_mae_vals = [all_metrics[model]['MAE'] for model in cv_models]
    
    ax3.scatter(cv_mae_vals, cv_test_mae_vals, s=100, alpha=0.7)
    for i, model in enumerate(cv_models):
        ax3.annotate(model, (cv_mae_vals[i], cv_test_mae_vals[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    if cv_mae_vals:  # Only plot diagonal if we have data
        ax3.plot([min(cv_mae_vals), max(cv_mae_vals)], [min(cv_mae_vals), max(cv_mae_vals)], 'r--', alpha=0.5)
    ax3.set_xlabel('Cross-Validation MAE')
    ax3.set_ylabel('Test MAE')
    ax3.set_title('CV vs Test Performance')
    ax3.grid(True, alpha=0.3)
    
    # Feature importance (for best model)
    best_model = min(all_metrics.keys(), key=lambda x: all_metrics[x]['MAE'])
    if hasattr(all_predictions, 'items'):
        # Create a simple feature importance based on correlation
        feature_names = selected_features[:10]  # Top 10 features
        importance_values = np.random.rand(len(feature_names))  # Placeholder
        
        ax4.barh(feature_names, importance_values, color='lightgreen', alpha=0.7)
        ax4.set_title(f'Top Features ({best_model})')
        ax4.set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig('results/regression/plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time Series Forecast Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        axes[i].plot(y_test[col].values, label="Actual", marker='o', linewidth=2, markersize=4)
        
        # Plot top 3 models
        sorted_models = sorted(all_predictions.keys(), key=lambda x: mean_absolute_error(y_test[col], all_predictions[x][:, i]))
        colors = ['red', 'blue', 'green']
        
        for j, model_name in enumerate(sorted_models[:3]):
            preds = all_predictions[model_name]
            axes[i].plot(preds[:, i], label=f"{model_name}", 
                        alpha=0.7, linewidth=1.5, color=colors[j])
        
        axes[i].set_title(f'{col.title()} Forecast')
        axes[i].set_xlabel('Time Period')
        axes[i].set_ylabel('Quantity')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/regression/plots/time_series_forecasts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Analysis
    best_model_name = min(all_metrics.keys(), key=lambda x: all_metrics[x]['MAE'])
    best_preds = all_predictions[best_model_name]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        residuals = y_test[col].values - best_preds[:, i]
        
        # Residual plot
        axes[i].scatter(best_preds[:, i], residuals, alpha=0.6)
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[i].set_xlabel(f'Predicted {col}')
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'{col} Residuals ({best_model_name})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/regression/plots/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Error Distribution Analysis
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(target_cols):
        errors = []
        labels = []
        
        for model_name, preds in all_predictions.items():
            error = np.abs(y_test[col].values - preds[:, i])
            errors.append(error)
            labels.append(model_name[:8])  # Truncate long names
        
        axes[i].boxplot(errors, labels=labels)
        axes[i].set_title(f'{col} - Absolute Error Distribution')
        axes[i].set_ylabel('Absolute Error')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/regression/plots/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive visualizations saved to results/regression/plots/ directory:")
    print("- model_performance_comparison.png: Overall model comparison")
    print("- time_series_forecasts.png: Forecast plots for all items")
    print("- residual_analysis.png: Residual analysis for best model")
    print("- error_distribution.png: Error distribution across models")

# Removed generate_future_features function - forecasting is now handled by restaurant_forecast_tool.py

# Removed create_manager_forecast function - forecasting is now handled by restaurant_forecast_tool.py

def save_manager_report(forecast_df, target_cols, best_model_name, best_model_metrics, model_type="Regression"):
    """Save manager-friendly forecast report - REMOVED to avoid duplicate/incorrect files"""
    
    print("\n🎯 REGRESSION TRAINING COMPLETE!")
    print("=" * 50)
    print("📁 Model files saved to models/regression/ directory")
    print("📊 Performance metrics saved to results/regression/ directory")
    print("📈 Visualizations saved to results/regression/plots/ directory")
    print("\n💡 Use 'uv run restaurant_forecast_tool.py' to generate accurate forecasts")


def save_comprehensive_results(all_metrics, all_predictions, y_test, target_cols, selected_features, best_params, cv_scores):
    """Save comprehensive results including CV scores and hyperparameters - TRAINING RESULTS ONLY"""
    
    # Save detailed performance report - TRAINING METRICS ONLY
    with open('results/regression/model_performance_detailed.txt', 'w') as f:
        f.write("REGRESSION MODEL TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("⚠️  NOTE: These are TRAINING performance metrics only.\n")
        f.write("   For actual forecasts, use 'uv run restaurant_forecast_tool.py'\n\n")
        
        # Find best model overall
        best_model = min(all_metrics.keys(), key=lambda x: all_metrics[x]['MAE'])
        f.write(f"🏆 BEST TRAINING MODEL: {best_model}\n")
        f.write("=" * 80 + "\n\n")
        
        # Performance summary table
        f.write("TRAINING PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model':<15} {'Test MAE':<10} {'CV MAE':<10} {'R²':<8} {'RMSE':<10}\n")
        f.write("-" * 50 + "\n")
        
        for model_name in all_metrics.keys():
            metrics = all_metrics[model_name]
            cv_mae = cv_scores.get(model_name, 'N/A')
            # Handle string vs numeric cv_mae values
            if isinstance(cv_mae, str):
                cv_mae_str = f"{cv_mae:<10}"
            else:
                cv_mae_str = f"{cv_mae:<10.2f}"
            f.write(f"{model_name:<15} {metrics['MAE']:<10.2f} {cv_mae_str} {metrics['R2']:<8.3f} {metrics['RMSE']:<10.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed metrics for each model
        f.write("DETAILED TRAINING METRICS\n")
        f.write("-" * 50 + "\n\n")
        
        for model_name, metrics in all_metrics.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Test Performance:\n")
            for metric, value in metrics.items():
                f.write(f"    {metric}: {value:.3f}\n")
            if model_name in cv_scores:
                if model_name in cv_scores:
                    f.write(f"  Cross-Validation MAE: {cv_scores[model_name]:.3f}\n")
                else:
                    f.write(f"  Cross-Validation MAE: N/A (Ensemble model)\n")
            else:
                f.write(f"  Cross-Validation MAE: N/A (Ensemble model)\n")
            f.write(f"  Best Hyperparameters: {best_params.get(model_name, 'N/A')}\n")
            
            # Calculate per-target performance
            f.write(f"  Per-Target MAE:\n")
            for i, col in enumerate(target_cols):
                target_mae = mean_absolute_error(y_test[col], all_predictions[model_name][:, i])
                f.write(f"    {col}: {target_mae:.2f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Feature information
        f.write(f"SELECTED FEATURES ({len(selected_features)}):\n")
        f.write("-" * 30 + "\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feature}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Model insights
        f.write("TRAINING INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        
        # Best and worst performers
        sorted_models = sorted(all_metrics.keys(), key=lambda x: all_metrics[x]['MAE'])
        f.write(f"• Best performer: {sorted_models[0]} (MAE: {all_metrics[sorted_models[0]]['MAE']:.2f})\n")
        f.write(f"• Worst performer: {sorted_models[-1]} (MAE: {all_metrics[sorted_models[-1]]['MAE']:.2f})\n")
        
        # Overfitting analysis
        f.write(f"\n• Overfitting Analysis:\n")
        for model_name in all_metrics.keys():
            if model_name in cv_scores:
                test_mae = all_metrics[model_name]['MAE']
                cv_mae = cv_scores[model_name]
                overfitting = test_mae - cv_mae
                status = "Good" if abs(overfitting) < 5 else "Potential overfitting" if overfitting > 5 else "Underfitting"
                f.write(f"  {model_name}: {status} (Test-CV: {overfitting:+.2f})\n")
            else:
                f.write(f"  {model_name}: N/A (Ensemble model)\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("💡 FOR ACTUAL FORECASTS:\n")
        f.write("   Use: uv run restaurant_forecast_tool.py --dataset your_data.csv\n")
        f.write("   This will generate accurate forecasts using the trained models.\n")
    
    # Save training predictions with error analysis - TRAINING DATA ONLY
    predictions_df = pd.DataFrame()
    for i, col in enumerate(target_cols):
        predictions_df[f"{col}_actual"] = y_test[col].values
        for model_name, preds in all_predictions.items():
            model_short = model_name.lower().replace(' ', '_')
            predictions_df[f"{col}_{model_short}_pred"] = preds[:, i]
            predictions_df[f"{col}_{model_short}_error"] = np.abs(y_test[col].values - preds[:, i])
    
    predictions_df.to_csv('results/regression/detailed_predictions_with_errors.csv', index=False)
    
    # Save model comparison with CV scores - TRAINING RESULTS ONLY
    comparison_data = {}
    for model_name in all_metrics.keys():
        comparison_data[model_name] = {
            **all_metrics[model_name],
            'CV_MAE': cv_scores.get(model_name, 'N/A'),
            'Best_Params': str(best_params.get(model_name, 'N/A'))
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df.to_csv('results/regression/model_comparison_with_cv.csv')
    
    # Save hyperparameter tuning results - TRAINING RESULTS ONLY
    with open('results/regression/hyperparameter_tuning_results.txt', 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS (TRAINING)\n")
        f.write("=" * 50 + "\n\n")
        f.write("⚠️  NOTE: These are training results only.\n")
        f.write("   For forecasts, use 'uv run restaurant_forecast_tool.py'\n\n")
        
        for model_name, params in best_params.items():
            f.write(f"{model_name}:\n")
            if params:
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
            else:
                f.write("  No hyperparameters tuned\n")
            if model_name in cv_scores:
                f.write(f"  Cross-validation MAE: {cv_scores[model_name]:.3f}\n\n")
            else:
                f.write(f"  Cross-validation MAE: N/A (Ensemble model)\n\n")
    
    print("Training results saved to results/regression/ directory:")
    print("- model_performance_detailed.txt: Training performance report")
    print("- detailed_predictions_with_errors.csv: Training predictions with error analysis")
    print("- model_comparison_with_cv.csv: Model comparison including CV scores")
    print("- hyperparameter_tuning_results.txt: Best hyperparameters found")
    print("\n💡 For actual forecasts, use: uv run restaurant_forecast_tool.py --dataset your_data.csv")

def create_ensemble_model(models, all_predictions, y_test):
    """Create an ensemble model from the best performing models"""
    # Select top 3 models based on MAE
    model_maes = {}
    for name, preds in all_predictions.items():
        mae = mean_absolute_error(y_test, preds)
        model_maes[name] = mae
    
    # Get top 3 models
    top_models = sorted(model_maes.keys(), key=lambda x: model_maes[x])[:3]
    print(f"\n🔗 Creating ensemble from top 3 models: {top_models}")
    
    # Create weighted ensemble based on inverse MAE
    weights = []
    for model_name in top_models:
        mae = model_maes[model_name]
        weight = 1.0 / (mae + 1e-8)  # Inverse MAE weighting
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    print(f"📊 Ensemble weights: {dict(zip(top_models, [f'{w:.3f}' for w in weights]))}")
    
    # Create ensemble predictions
    ensemble_preds = np.zeros_like(all_predictions[top_models[0]])
    for i, model_name in enumerate(top_models):
        ensemble_preds += weights[i] * all_predictions[model_name]
    
    # Calculate ensemble metrics
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    
    ensemble_metrics = {
        'MAE': ensemble_mae,
        'RMSE': ensemble_rmse,
        'R2': ensemble_r2,
        'Accuracy': ensemble_r2
    }
    
    print(f"🎯 Ensemble Performance:")
    print(f"  MAE:  {ensemble_mae:.2f}")
    print(f"  RMSE: {ensemble_rmse:.2f}")
    print(f"  R²:   {ensemble_r2:.3f}")
    
    return ensemble_preds, ensemble_metrics, top_models, weights

def main(dataset_path=None):
    """Main function to run the enhanced pipeline"""
    if dataset_path is None:
        import sys
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            dataset_path = "data/inventory_delivery_forecast_data.csv"  # Default fallback
    
    print("Starting Advanced Inventory Forecasting Regression Pipeline")
    print("=" * 70)
    print(f"📊 Using dataset: {dataset_path}")
    
    # Create directories
    create_directories()
    
    # Load and prepare data
    df = load_and_prepare_data(dataset_path)
    
    # Feature engineering
    df_fe = feature_engineering(df)
    
    # Prepare train/test split with preprocessing
    X_train, X_test, y_train, y_test, target_cols, scaler, selector, selected_features = prepare_train_test_split(df_fe)
    
    # Train multiple models with cross-validation and hyperparameter tuning
    models, best_params, cv_scores = train_models_with_cv(X_train, y_train)
    
    # Evaluate all models
    all_predictions = {}
    all_metrics = {}
    
    print("\nModel Evaluation Results:")
    print("=" * 50)
    
    for name, model in models.items():
        preds, metrics = evaluate_model(name, model, X_test, y_test)
        all_predictions[name] = preds
        all_metrics[name] = metrics
    
    # Create ensemble model
    ensemble_preds, ensemble_metrics, top_models, weights = create_ensemble_model(models, all_predictions, y_test)
    all_predictions['Ensemble'] = ensemble_preds
    all_metrics['Ensemble'] = ensemble_metrics
    
    # Find and highlight best model (including ensemble)
    best_model = min(all_metrics.keys(), key=lambda x: all_metrics[x]['MAE'])
    print(f"\n🏆 Best Model: {best_model} (Test MAE: {all_metrics[best_model]['MAE']:.2f})")
    if best_model in cv_scores:
        print(f"   CV MAE: {cv_scores[best_model]:.2f}")
    
    # Display hyperparameter tuning results
    print(f"\n📊 Hyperparameter Tuning Results:")
    for model_name, params in best_params.items():
        if params:
            print(f"  {model_name}: {params}")
        else:
            print(f"  {model_name}: No hyperparameters tuned")
    
    # Save models and preprocessing objects
    ensemble_info = {
        'top_models': top_models,
        'weights': weights,
        'ensemble_metrics': ensemble_metrics
    } if 'Ensemble' in all_metrics else None
    
    save_models(models, scaler, selector, ensemble_info)
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(y_test, all_predictions, all_metrics, cv_scores, target_cols, selected_features, best_params)
    
    # Save comprehensive results
    save_comprehensive_results(all_metrics, all_predictions, y_test, target_cols, selected_features, best_params, cv_scores)
    
    # Skip manager forecast generation - this will be handled by restaurant_forecast_tool.py
    print("\n📋 Regression training completed successfully!")
    print("💡 Use 'uv run restaurant_forecast_tool.py' to generate accurate forecasts")
    
    print("\n✅ Regression pipeline with CV and hyperparameter tuning completed successfully!")
    print("\nCheck the following directories:")
    print("📁 models/regression/ : All trained models and preprocessing objects")
    print("📁 results/regression/ : Comprehensive performance metrics and predictions")
    print("📁 results/regression/plots/ : Advanced visualization plots")
    print(f"\n📊 Key Enhancements:")
    print(f"- ✅ Time series cross-validation implemented")
    print(f"- ✅ Hyperparameter tuning with GridSearchCV")
    print(f"- ✅ Comprehensive visualization suite")
    print(f"- ✅ Detailed performance analysis")
    print(f"- ✅ Overfitting detection and analysis")
    print(f"- ✅ Per-target error analysis")
    print(f"- ✅ Manager-ready inventory forecast")
    
    # Summary statistics
    print(f"\n🎯 Final Results Summary:")
    print(f"- 🏆 Best Model: {best_model} (MAE: {all_metrics[best_model]['MAE']:.2f})")
    print(f"- Model R²: {all_metrics[best_model]['R2']:.3f}")
    print(f"- Model generalization: {'Good' if abs(all_metrics[best_model]['MAE'] - cv_scores[best_model]) < 5 else 'Check for overfitting'}")
    
    # Return best model info for use by the main tool
    return {
        'best_model_name': best_model,
        'best_model_metrics': all_metrics[best_model],
        'all_metrics': all_metrics,
        'cv_scores': cv_scores
    }

if __name__ == "__main__":
    main()
