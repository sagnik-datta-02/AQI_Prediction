import streamlit as st
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import gzip

# Load the compressed pickle file with only the model
with gzip.open('random_compressed2334.pkl.gz', 'rb') as f:
    model = pickle.load(f)
def predict_aqi_rf(model, scaler, input_values):
    """
    Predicts AQI based on input feature values using a trained Random Forest model.

    Parameters:
    - model: Trained RandomForestRegressor model
    - scaler: StandardScaler used for feature scaling during training
    - input_values: List or array containing input values for each feature

    Returns:
    - Predicted AQI value
    """

    # Ensure input_values is a 2D array
    input_values = np.array(input_values).reshape(1, -1)

    # Scale the input features using the same scaler used during training
    input_scaled = scaler.transform(input_values)

    # Make predictions
    aqi_prediction = model.predict(input_scaled)

    return aqi_prediction[0]

def main():
    # Custom CSS styles
    st.markdown(
        """
        <style>
            .main {
                background-color: #f4f4f4;
                padding: 20px;
                margin: auto;
                max-width: 800px;
            }
            .title {
                text-align: center;
                color: #1f78b4;
            }
            .project-info {
                margin-bottom: 20px;
                padding: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                overflow-y: auto;
                max-height: 300px; /* Set your desired max height */
            }
            .sidebar {
                background-color: #ffffff;
                padding: 20px;
                margin-top: 20px;
                margin-right: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px #888888;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            .result {
                text-align: center;
                color: #4CAF50;
                font-size: 24px;
                margin-top: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown('<h1 class="title">Air Quality Index Prediction</h1>', unsafe_allow_html=True)

    # Using st.columns instead of st.beta_columns
    col1, col2 = st.columns(2)

    with col1:
        st.sidebar.header('Input Parameters')
        area = st.sidebar.number_input("Area Number", key=0, placeholder="Type a number...")
        PM25 = st.sidebar.number_input("PM 2.5", key=1, placeholder="Type a number...")
        PM10 = st.sidebar.number_input("PM 10", key=2, placeholder="Type a number...")
        NO = st.sidebar.number_input("NO", key=3, placeholder="Type a number...")
        NO2 = st.sidebar.number_input("NO2", key=4, placeholder="Type a number...")
        NOx = st.sidebar.number_input("NOx", key=5, placeholder="Type a number...")
        NH3 = st.sidebar.number_input("NH3", key=12, placeholder="Type a number...")

    with col2:
        CO = st.sidebar.number_input("CO", key=6, placeholder="Type a number...")
        SO2 = st.sidebar.number_input("SO2", key=7, placeholder="Type a number...")
        O3 = st.sidebar.number_input("O3", key=8, placeholder="Type a number...")
        Benzene = st.sidebar.number_input("Benzene", key=9, placeholder="Type a number...")
        Toluene = st.sidebar.number_input("Toluene", key=10, placeholder="Type a number...")
        Xylene = st.sidebar.number_input("Xylene", key=11, placeholder="Type a number...")

    data = [area, PM25, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]
    model=''
    # Load the model and scaler
    with gzip.open('random_compressed2334.pkl.gz', 'rb') as f:
        model = pickle.load(f)

    scaler_in = open('standardscaler.pkl', 'rb')
    scaler = pickle.load(scaler_in)
    scaler_in.close()

    st.markdown('<div class="button">', unsafe_allow_html=True)
    if st.button("Predict"):
        predicted_aqi_rf = predict_aqi_rf(model, scaler, data)
        st.markdown('<p class="result">Predicted AQI: {:.2f}</p>'.format(predicted_aqi_rf), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="project-info">', unsafe_allow_html=True)
    # Project Information
    st.text("Project Information:")
    st.text("Contributors:")
    st.text("1. Sagnik Datta - 44")
    st.text("2. Swapnendu Banerjee - 50")
    st.text("3. Moyukh Chowdhury - 46")
    st.text("4. Urjita Ray - 36")
    st.text("Department: Computer Science & Engineering, RCCIIT")
    st.text("Lab: IT/Python Workshop PCC CS393")
    st.text("SEM 3 CSE A Grp-2")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
