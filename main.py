import streamlit as st
import numpy as np
import pandas as pd
import ast
import string
import re
import nltk
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

st.header("Disaster Tweets Classification App")
data = pd.read_csv("https://raw.githubusercontent.com/Jacobche/fake_news_detection/main/X_test.csv")
X_test_df = data['combined_string']

#load countvectorizer
count_vectorizer = pickle.load(open("count_vectorizer.pkl", "rb"))
X_test = count_vectorizer.transform(X_test_df)

# load model
with open("best_model.json", "r") as f:
    json_dict = json.load(f)

coef_array = np.array(json_dict["coef"])
intercept_array = np.array(json_dict["intercept"])
classes_array = np.array(json_dict["classes"])
n_iter = json_dict["n_iter"]

LR = LogisticRegression()
LR.coef_ = coef_array
LR.intercept_ = intercept_array
LR.classes_ = classes_array
LR.n_iter_ = n_iter

if st.checkbox('Show Testing Dataframe'):
    data

st.subheader("Best Model used is Logistic Regression")

st.text("Please press the button below to generate the classification results!")

if st.button('Prediction Now'):
    st.subheader("Prediction Results")
    st.write("Original Dataframe with Predictions:")
    # Predict
    prediction = LR.predict(X_test)
    
    # Add a new column with the predicted results to the dataframe
    data['predicted_result'] = prediction

    # Display the new dataframe with predictions
    st.write(data)




