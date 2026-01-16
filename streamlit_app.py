
import streamlit as st
st.write("Scaler exists?", scaler_path.exists(), scaler_path)


import streamlit as st
import numpy as np 
import joblib

scaler = joblib.load('scaler.pkl')

st.title("Restaurant Rating Predictor")

st.set_page_config(layout="wide")
st.set_page_config(page_title="Restaurant Rating Predictor", page_icon=":fork_and_knife:")
st.caption("Predicting Restaurant Ratings based on various features using Machine Learning")
st.divider()


averagecost = st.number_input("Please enter the estimated average cost for 2:", min_value=50, max_value=99999, value=1000, step=200)
tablebooking = st.selectbox("Does the restaurant accept table bookings?", ("Yes", "No"))
onlinedelivery = st.selectbox("Does the restaurant offer online delivery?", ("Yes", "No"))
pricerange = st.selectbox("Select the price range of the restaurant (1 Cheapest, 4 Most Expensive)", (1, 2, 3, 4))


predictbutton = st.button("Predict Rating")

st.divider()

model = joblib.load('restaurant_rating_predictor_model.pkl')

bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

# Has table booking 0 is No, 1 is Yes
# Has online delivery 0 is No, 1 is Yes

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_X_values =np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()

    prediction = model.predict(X)

    st.write(prediction)
   

