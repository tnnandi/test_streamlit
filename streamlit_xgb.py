# run as "streamlit run *.py"
# use xgboost version 0.90

import streamlit as st
import joblib
import pandas as pd

st.write("# Prediction of Formation Energy")

# incput features 'Ba', 'Ca', 'La', 'Co', 'Cu', 'Mn', 'Mo', 'Ni', 'Ti', 'W', 'Zn', 'Zr'

col1, col2, col3 = st.columns(3)

# getting user input

#Ba = col1.selectbox("Enter your gender",["Male", "Female"])
Ba = col1.number_input("Enter Ba content")
Ca = col1.number_input("Enter Ca content")
La = col1.number_input("Enter La content")
Co = col1.number_input("Enter Co content")
Cu = col1.number_input("Enter Cu content")
Mn = col1.number_input("Enter Mn content")
Mo = col1.number_input("Enter Mo content")
Ni = col1.number_input("Enter Ni content")
Ti = col1.number_input("Enter Ti content")
W = col1.number_input("Enter W content")
Zn = col1.number_input("Enter Zn content")
Zr = col1.number_input("Enter Zr content")



df_pred = pd.DataFrame([[Ba, Ca, La, Co, Cu, Mn, Mo, Ni, Ti, W, Zn, Zr]],

columns= ['Ba', 'Ca', 'La', 'Co', 'Cu', 'Mn', 'Mo', 'Ni', 'Ti', 'W', 'Zn', 'Zr'])


model = joblib.load('xgb_model.pkl')
prediction = model.predict(df_pred)

if st.button('Predict'):
    
    st.write('<p class="big-font"> prediction </p>', prediction, unsafe_allow_html=True)

#     if(prediction[0]==0):
#         st.write('<p class="big-font">You likely will not develop heart disease in 10 years.</p>',unsafe_allow_html=True)

#     else:
#         st.write('<p class="big-font">You are likely to develop heart disease in 10 years.</p>',unsafe_allow_html=True)
