import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

st.header("Disaster Tweets Classification App")
data = pd.read_csv("https://raw.githubusercontent.com/Jacobche/fake_news_detection/main/X_test.csv")

#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Testing Dataframe'):
    data

st.subheader("Best Model used is Logistic Regression")

st.text("Please press the button below to generate the classification results!")

if st.button('Press Me'):
    st.subheader("Prediction Results")
    st.write("Original Dataframe with Predictions:")
    # Copy the original dataframe to a new one to show predictions
    prediction_data = data.copy()
    
    # Collect user inputs
    input_species = encoder.transform([st.session_state.inp_species])
    input_Length1 = st.session_state.input_Length1
    input_Length2 = st.session_state.input_Length2
    input_Length3 = st.session_state.input_Length3
    input_Height = st.session_state.input_Height
    input_Width = st.session_state.input_Width

    # Predict fish weight
    inputs = np.array([[input_species, input_Length1, input_Length2, input_Length3, input_Height, input_Width]])
    prediction = best_xgboost_model.predict(inputs)
    
    # Add a new column with the predicted weights to the dataframe
    prediction_data['Predicted_Weight'] = prediction

    # Display the new dataframe with predictions
    st.write(prediction_data)




