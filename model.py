import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib



# --- Configuration ---
data_filename = 'synthetic_hospital_inventory_demand.csv'
target_column = 'Actual_Consumption'
test_set_duration = '1Y' # Use last 1 year for testing
validation_set_duration = '1Y' # Use year before test set for validation (optional, but good practice)

# --- Load Data ---
df = pd.read_csv(data_filename, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"Loaded data shape: {df.shape}")

# --- Feature Engineering ---
# Ensure correct types for LightGBM
df['Day_of_Week'] = df['Day_of_Week'].astype('category')
# Add more time features if desired (e.g., day of month, quarter)
# df['Day_of_Month'] = df['Date'].dt.day
# df['Quarter'] = df['Date'].dt.quarter

# Define features to use
# Exclude Date, Year (already captured by other time features/trend), and potentially redundant lags if desired
features = [
    'Month', 'Week_of_Year', 'Is_Weekend', 'Day_of_Week', # Can use Day_of_Week if enabling categorical feature in LGBM
    'Simulated_Patient_Visits',
    'Consumption_Lag_1D', 'Consumption_Lag_7D',
    'Consumption_Avg_7D', 'Consumption_Avg_30D'
    # Add other engineered features here
]

# Handle initial NaNs in lagged features (simple fill with 0 or bfill)
# Handle initial NaNs in lagged features (simple fill with 0 or bfill)
# Excluding 'Day_of_Week' from the fillna operation to avoid TypeError
numerical_features = [f for f in features if f != 'Day_of_Week']  
df[numerical_features] = df[numerical_features].fillna(0) # Simple fill with 0 for demonstration

# Filling NaNs in 'Day_of_Week' with the most frequent category if needed
df['Day_of_Week'] = df['Day_of_Week'].fillna(df['Day_of_Week'].mode()[0])

print(f"Features being used: {features}")

# --- Data Splitting (Time Series Aware) ---
# Determine split points
test_split_date = df['Date'].max() - pd.Timedelta(days=365 * 1) # Approx 1 year
validation_split_date = test_split_date - pd.Timedelta(days=365 * 1) # Approx 1 year before test

# Create train, validation, test sets
train_df = df[df['Date'] < validation_split_date].copy()
val_df = df[(df['Date'] >= validation_split_date) & (df['Date'] < test_split_date)].copy()
test_df = df[df['Date'] >= test_split_date].copy()

X_train, y_train = train_df[features], train_df[target_column]
X_val, y_val = val_df[features], val_df[target_column]
X_test, y_test = test_df[features], test_df[target_column]

print(f"Train set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Model Training (LightGBM) ---
# Basic parameters - these should be tuned for best performance
lgbm_params = {
    'objective': 'regression_l1', # MAE loss
    'metric': 'mae',
    'n_estimators': 1000,         # Number of trees
    'learning_rate': 0.05,
    'feature_fraction': 0.8,      # Randomly select 80% of features per tree
    'bagging_fraction': 0.8,      # Randomly select 80% of data per tree iteration
    'bagging_freq': 1,
    'lambda_l1': 0.1,             # L1 regularization
    'lambda_l2': 0.1,             # L2 regularization
    'num_leaves': 31,             # Max number of leaves in one tree
    'verbose': -1,                # Suppress verbose output
    'n_jobs': -1,                 # Use all available cores
    'seed': 42,
    'boosting_type': 'gbdt',
    # 'categorical_feature': ['Day_of_Week'] # Alternative: Specify categorical features by name/index
}

model = lgb.LGBMRegressor(**lgbm_params)

# Train with early stopping based on validation set
print("Training LightGBM model...")
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]) # Stop if validation MAE doesn't improve for 50 rounds

# --- Prediction & Evaluation ---
print("\nEvaluating model performance...")
y_pred_test = model.predict(X_test)
y_pred_val = model.predict(X_val)

# Calculate metrics
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nValidation Set Performance:")
print(f"  MAE: {mae_val:.2f}")
print(f"  RMSE: {rmse_val:.2f}")

print(f"\nTest Set Performance:")
print(f"  MAE: {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")

# Calculate MAPE (Mean Absolute Percentage Error) - careful with zero actual values
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero - replace 0 with a small number or filter them out
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
print(f"  MAPE: {mape_test:.2f}%")


# --- Feature Importance ---
print("\nFeature Importances:")
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=20, height=0.8)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()

# --- Plot Predictions vs Actuals (on Test Set) ---
plt.figure(figsize=(15, 7))
plt.plot(test_df['Date'], y_test, label='Actual Consumption', alpha=0.7)
plt.plot(test_df['Date'], y_pred_test, label='Predicted Consumption', linestyle='--')
plt.title('Actual vs Predicted Consumption (Test Set)')
plt.xlabel('Date')
plt.ylabel('Tablets Consumed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(model, 'restocking_model.pkl')
print("Model saved to 'restocking_model.pkl'")