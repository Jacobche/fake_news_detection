import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle

from sklearn.linear_model import LogisticRegression

st.header("Disaster Tweets Detection App")
data = pd.read_csv("https://raw.githubusercontent.com/Jacobche/fake_news_detection/main/Combined_test.csv")
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
    data['text']

st.subheader("Best ML Model used is Logistic Regression")

# Add a unique key to the text_input widget
user_input = st.text_input('Enter Text Input', key="user_input")


st.write("Click the button below to generate the prediction results!")

if st.button('Predict Now'):
    st.subheader("Prediction Results")
    st.write("Original Dataframe with Predictions:")
    # Predict
    prediction = LR.predict(X_test)
    
    # Add a new column with the predicted results to the dataframe
    data['predicted_result'] = prediction
    data = data.drop('combined_string', axis=1)

    # Filter the dataframe by user input
    if user_input:
        filtered_df = data[data['text'].str.contains(user_input)]
        # Display the filtered dataframe
        st.write(filtered_df)
    else:
        # If there is no user input, show the entire dataframe
        st.write(data)
