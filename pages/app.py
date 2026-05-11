import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- UI ---------------- #

st.title("🏭 Know what the air's like in your city before you go out")
st.write("Get ahead of air pollution with predictions for your city's Air Quality Index")

st.image('air.jpeg', caption='Predict AQI', use_column_width=True)

# ---------------- Prediction Function ---------------- #

def make_prediction(model, feature_set):
    y_predict = model.predict(feature_set)
    # Ensure we get a scalar
    return round(float(np.squeeze(y_predict)), 2)
# ---------------- Load Models ---------------- #

mreg = pickle.load(open("Models/Multiple Regression.pkl", 'rb'))
preg = pickle.load(open("Models/pregression.pkl", 'rb'))
dec_tree = pickle.load(open("Models/Decision tree.pkl", 'rb'))
rt_reg = pickle.load(open("Models/RandomForest.pkl", 'rb'))
svr_reg = pickle.load(open("Models/svrression.pkl", 'rb'))
poly_reg = pickle.load(open("Models/ploy_reg.pkl", "rb"))

# Load OneHotEncoder
ohe = pickle.load(open("Models/OneHotEncoder_Featureset.pkl", "rb"))

# ---------------- User Inputs ---------------- #

cities = ["Ahmedabad", "Mumbai", "Delhi", "Chennai", "Bangalore"]
selected_city = st.selectbox("Select City:", cities)

pm2_5 = st.slider("PM2.5 (Fine Particulate Matter)", 0.0, 50.0, 3.0)
pm10 = st.slider("PM10 (Coarse Particulate Matter)", 0.0, 50.0, 3.0)

model_names = [
    'Multiple Regression',
    'Polynomial Regression',
    'Decision Tree',
    'Random Forest',
    'SVR'
]

selected_model = st.selectbox("Select Model", model_names)

predict_button = st.button("Predict Air Quality")

# ---------------- Prediction ---------------- #

if predict_button:

    user_data = {
        "PM2.5": pm2_5,
        "PM10": pm10,
        "NO": 17.574730,
        "NO2": 28.560659,
        "Nox": 32.309123,
        "NH3": 23.483476,
        "CO": 2.248598,
        "SO2": 14.531977,
        "O3": 34.491430,
        "Benzene": 3.280840,
        "Toluene": 8.700972,
        "Xylene": 166.463581
    }

    user_df = pd.DataFrame([user_data])

    # ---------------- Encode City ---------------- #

    city_encoded = ohe.transform([[selected_city]]).toarray()
    city_df = pd.DataFrame(city_encoded)

    # ---------------- Combine Features ---------------- #

    feature_set = pd.concat([city_df, user_df], axis=1)

    # Convert to numpy
    feature_set = feature_set.values

    # Fix feature mismatch (model expects 37 features)
    if feature_set.shape[1] < 37:
        padding = np.zeros((1, 37 - feature_set.shape[1]))
        feature_set = np.hstack((feature_set, padding))

    # ---------------- Model Prediction ---------------- #

    if selected_model == 'Multiple Regression':
        y_predict = make_prediction(mreg, feature_set)

    elif selected_model == 'Polynomial Regression':
        poly_features = poly_reg.fit_transform(feature_set)
        y_predict = preg.predict(poly_features)
        y_predict = round(float(y_predict), 2)

    elif selected_model == 'Decision Tree':
        y_predict = make_prediction(dec_tree, feature_set)

    elif selected_model == 'Random Forest':
        y_predict = make_prediction(rt_reg, feature_set)

    elif selected_model == 'SVR':
        y_predict = make_prediction(svr_reg, feature_set)

    # ---------------- Output ---------------- #

    st.success(f"Predicted AQI for {selected_city}: **{y_predict}**")