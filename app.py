import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from utils.util import get_prediction, ordinal_encode_test

model = joblib.load(r'model/extratree.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", 
                   layout="centered")


#creating option list for dropdown menu
opt_day_of_week = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

opt_driver_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

opt_accident_cause = ['No distancing', 'Changing lane to the right', 'Changing lane to the left', 'Driving carelessly',  'No priority to vehicle', 'Moving Backward',
                 'No priority to pedestrian', 'Other', 'Overtaking', 'Driving under the influence of drugs', 'Driving to the left',  'Getting off the vehicle improperly', 
                 'Driving at high speed', 'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',  'Unknown', 'Improper parking']

opt_junction_type = ['Y Shape', 'No junction', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape', 'X Shape']

opt_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
             'Two-way (divided with solid lines road marking)', 'Unknown']

opt_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q', 'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
                        'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi', 'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

opt_driver_experience = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

features = ['vehicles_involved', 'driver_age', 'day_of_week', 'casualties', 'hour', 'driving_experience', 'vehicle_type', 'junction_type', 'accident_cause', 'lanes']


st.markdown("<h1 style='text-align: center; color: green;'>Accident Severity Prediction 🚧</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    with st.form('prediction_form'):

        st.subheader(":gray[Enter the following information:]")
        
        day_of_week = st.selectbox(":orange[Select day of week:]", options=opt_day_of_week)
        hour = st.slider(":orange[Select hour of the day:]", 0, 23, value=0, format="%d")
        vehicles_involved = st.slider(":orange[Select number of vehicles involved:]", 1, 7, value=0, format="%d")
        casualties = st.slider(":orange[Select number of casualties:]", 1, 8, value=0, format="%d")
        driver_age = st.selectbox(":orange[Select driver Age:]", options=opt_driver_age)
        accident_cause = st.selectbox(":orange[Select cause of Accident:]", options=opt_accident_cause)
        junction_type = st.selectbox(":orange[Select junction type:]", opt_junction_type)
        lanes = st.selectbox(":orange[Select Lanes:]", options=opt_lanes)
        vehicle_type = st.selectbox(":orange[Select vehicle type:]", options=opt_vehicle_type)
        driving_experience = st.selectbox(":orange[Select driving experience:]", options=opt_driver_experience)
        
        submit = st.form_submit_button("Predict")

    if submit:
        day_of_week = ordinal_encode_test(day_of_week, opt_day_of_week)
        driver_age =  ordinal_encode_test(driver_age, opt_driver_age)
        accident_cause = ordinal_encode_test(accident_cause, opt_accident_cause)
        junction_type = ordinal_encode_test(junction_type, opt_junction_type)
        lanes = ordinal_encode_test(lanes, opt_lanes)
        vehicle_type = ordinal_encode_test(vehicle_type, opt_vehicle_type)
        driving_experience = ordinal_encode_test(driving_experience, opt_driver_experience) 
        

        data = np.array([day_of_week, hour, vehicles_involved, casualties, driver_age, accident_cause, junction_type, lanes, vehicle_type, driving_experience]).reshape(1,-1)
        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred}")

if __name__ == '__main__':
    main()