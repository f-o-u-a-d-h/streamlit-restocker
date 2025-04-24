import joblib
import pandas as pd
import datetime

# Load the trained model
model = joblib.load('restocking_model.pkl')

# Define features used during training
features = [
    'Month', 'Week_of_Year', 'Is_Weekend', 'Day_of_Week',
    'Simulated_Patient_Visits',
    'Consumption_Lag_1D', 'Consumption_Lag_7D',
    'Consumption_Avg_7D', 'Consumption_Avg_30D'
]

def calculate_ideal_restocking(data):
    """
    Generate a restocking prediction using the trained LightGBM model.

    Parameters:
        data (dict): User input data containing visit counts, consumption, and date.

    Returns:
        int: Recommended quantity to restock.
    """
    now = datetime.datetime.strptime(data["date"], "%Y-%m-%d")

    # Simulate engineered features
    input_features = {
        'Month': now.month,
        'Week_of_Year': now.isocalendar()[1],
        'Is_Weekend': int(now.weekday() >= 5),
        'Day_of_Week': now.weekday(),
        'Simulated_Patient_Visits': (
            data['num_outpatient_visits'] + 
            data['num_emergency_visits'] + 
            data['num_inpatient_visits']
        ),
        'Consumption_Lag_1D': data['quantity_of_medicine_consumed'],
        'Consumption_Lag_7D': data['quantity_of_medicine_consumed'],
        'Consumption_Avg_7D': data['quantity_of_medicine_consumed'],
        'Consumption_Avg_30D': data['quantity_of_medicine_consumed'],
    }

    df_input = pd.DataFrame([input_features])
    df_input['Day_of_Week'] = df_input['Day_of_Week'].astype('category')

    prediction = model.predict(df_input)[0]
    restocking_amount = max(0, int(prediction - data['quantity_of_medicine_in_stock_remaining']))
    return restocking_amount
