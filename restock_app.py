import streamlit as st
import pandas as pd
import datetime
import joblib

# Load model
model = joblib.load('restocking_model.pkl')

# Feature list used in training
features = [
    'Month', 'Week_of_Year', 'Is_Weekend', 'Day_of_Week',
    'Simulated_Patient_Visits',
    'Consumption_Lag_1D', 'Consumption_Lag_7D',
    'Consumption_Avg_7D', 'Consumption_Avg_30D'
]

# Prediction function
def predict(data):
    now = datetime.datetime.strptime(data["date"], "%Y-%m-%d")

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
    return max(0, int(prediction - data['quantity_of_medicine_in_stock_remaining']))

# --- Streamlit UI ---

st.title("💊 Medicine Restocking Predictor")

with st.form("restock_form"):
    date = st.date_input("Date", datetime.date.today())
    outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=2500, value=10)
    emergency = st.number_input("Emergency Visits", min_value=0, max_value=250, value=5)
    inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=250, value=2)
    consumed = st.number_input("Medicine Consumed", min_value=0, max_value=3000, value=100)
    in_stock = st.number_input("Stock Remaining", min_value=0, max_value=5000, value=50)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            "date": date.strftime("%Y-%m-%d"),
            "num_outpatient_visits": outpatient,
            "num_emergency_visits": emergency,
            "num_inpatient_visits": inpatient,
            "quantity_of_medicine_consumed": consumed,
            "quantity_of_medicine_in_stock_remaining": in_stock
        }
        result = predict(input_data)
        st.success(f"🔄 Recommended Restocking Amount: **{result}** tablets")
