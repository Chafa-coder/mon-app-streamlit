# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:42:38 2025

@author: cbent
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

st.write("""
         # Penguin prediction App
         This app predicts the **Palmer Penguin** species !
         """)
st.sidebar.header('User Input Features')
# st.sidebar.markdown("""
#                     [ Example CSV Input ...](https....))
# """)

# Collects user inputs features into a dataframe
def user_input_features():
    
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bill_length_mm = st.sidebar.slider('Bill length in mm', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth in mm', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length in mm', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body mass in (g)', 2700.0, 6300.0, 4207.0)
    
    data = {'island' : island,
            'bill_length_mm' : bill_length_mm,
            'bill_depth_mm' : bill_depth_mm,
            'flipper_length_mm' : flipper_length_mm,
            'body_mass_g' : body_mass_g,
            'sex' : sex}
    
    features = pd.DataFrame(data, index=[0])
    return  features

input_df = user_input_features()


pinguins_raw = pd.read_csv("penguins_cleaned.CSV")
pinguins = pinguins_raw.drop(columns = ['species'])
df = pd.concat([input_df, pinguins], axis = 0)

encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col).astype(int)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]
    
df = df[:1] # select the 1st only

# display the user input features
st.subheader('User Input features')

st.write(df)

# read saved classification model
load_clf = joblib.load("model_clf_pinguins")

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader("Predictions")
pinguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(pinguins_species[prediction])

st.subheader('Predciction probability')
st.write(prediction_proba)


    
    
 
    
    
    

    
