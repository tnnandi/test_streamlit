#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st 
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def poly_feat_maker(data, degree=3, include_bias = False): #turning of include_bias doesn't make a constant column, which isn't necessary given normer centers the data
	poly = PolynomialFeatures(degree)
	data = poly.fit_transform(data)
	feat_names = poly.get_feature_names_out()
	return data



# In[23]:



st.write("# Prediction of Formation Energy (eV)")

# incput features 'Ba', 'Ca', 'La', 'Co', 'Cu', 'Mn', 'Mo', 'Ni', 'Ti', 'W', 'Zn', 'Zr'

col1, col2, col3 = st.columns(3)

# getting user input

#Ba = col1.selectbox("Enter your gender",["Male", "Female"])

bandgap = col1.number_input("Enter Band Gap (eV)")
density = col1.number_input("Enter density (g/cm3")
atom_density = col1.number_input("Enter atomic density (ang3/atom)")
magnetization = col1.number_input("Enter Magnetization (bohr magneton/F.U.)")
h_wt_frac = col1.number_input("Enter Hydrogen Weight Fraction")
d_char = col1.number_input("Enter avg. d-char")
f_char = col1.number_input("Enter avg. f-char")
electronegativity = col1.number_input("Enter avg. electronegativity")
T = col1.number_input("Enter Temperature (degC)")
P = col1.number_input("Enter Absolute Pressure (atm.)")

input_data = [bandgap, density, atom_density, magnetization, h_wt_frac, d_char, f_char, electronegativity, T, P]
#col_names= ['Egap', 'Dens', 'Dens_atom', 'Mag', 'H_wt_frac', 'D Char', 'F Char', 'Eneg', 'Temp.', 'Pres.']


input_array = np.array(input_data)
input_array_2d = np.array([input_array])

    


# In[38]:


model = joblib.load('simpletreetest_model.pkl')
feat_names = ['Band Gap (eV)','Density (g/cm3)','Atomic Density (A3/atom)','Magnetizon (Bohr Magnetons/F.U.)','H Wt. Frac','d Character','f Character','Electronegativity','Temperature (degC)','Pressure (Atmospheres Absolute)']


# In[35]:


input_array_2d


# In[39]:


df_pred = pd.DataFrame(input_array_2d,columns = feat_names)
print(df_pred)


# In[42]:



pca_loaded = joblib.load('pca_model.pkl')
scaler_loaded = joblib.load('scaler_model.pkl')


df_norm = scaler_loaded.transform(df_pred)
df_pca = pca_loaded.transform(df_norm)
df_poly = poly_feat_maker(df_pca)

prediction = model.predict(df_poly)

if st.button('Predict'):
    
    st.write('<p class="big-font"> prediction </p>', prediction, unsafe_allow_html=True)


# In[ ]:




