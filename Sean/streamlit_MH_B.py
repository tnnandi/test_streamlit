#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st 
import joblib
import pandas as pd
import numpy as np
import re #used for string splitting
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from PIL import Image
img = Image.open('Sean/netl_logo.png')
#st.set_page_config(page_title='Metal Hydride formation energy predictor', page_icon=img)

st.set_page_config(
    page_title="Metal Hyride Formation Energy Prediction",
    page_icon=img,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

#image = Image.open('netl_logo.jpg')

st.image(img, caption='NETL')#, width=25)



def poly_feat_maker(data, degree=3, include_bias = False): #turning of include_bias doesn't make a constant column, which isn't necessary given normer centers the data
	poly = PolynomialFeatures(degree)
	data = poly.fit_transform(data)
	feat_names = poly.get_feature_names_out()
	return data


class Element:
    def __init__(self, atomic_number, atomic_mass, period, group, d, f, elecneg):
        self.number = atomic_number
        self.mass = atomic_mass
        self.period = period
        self.group = group
        if d == 10:
            self.dchar = 0
        else:
            self.dchar = d
        if f == 14:
            self.fchar = 0
        else:
            self.fchar = f
        self.electronegativity = elecneg

element_data_list=[[1,'H',1.007,1,1,0,0,2.2],[2,'He',4.002,1,18,0,0,4.16],[3,'Li',6.941,2,1,0,0,0.98],[4,'Be',9.012,2,2,0,0,1.57],[5,'B',10.811,2,13,0,0,2.04],[6,'C',12.011,2,14,0,0,2.55],[7,'N',14.007,2,15,0,0,3.04],
[8,'O',15.999,2,16,0,0,3.44],[9,'F',18.998,2,17,0,0,3.98],[10,'Ne',20.18,2,18,0,0,4.787],[11,'Na',22.99,3,1,0,0,0.93],[12,'Mg',24.305,3,2,0,0,1.31],[13,'Al',26.982,3,13,0,0,1.61],[14,'Si',28.086,3,14,0,0,1.9],
[15,'P',30.974,3,15,0,0,2.19],[16,'S',32.065,3,16,0,0,2.58],[17,'Cl',35.453,3,17,0,0,3.16],[18,'Ar',39.948,3,18,0,0,3.242],[19,'K',39.098,4,1,0,0,0.82],[20,'Ca',40.078,4,2,0,0,1],[21,'Sc',44.956,4,3,1,0,1.36],
[22,'Ti',47.867,4,4,2,0,1.54],[23,'V',50.942,4,5,3,0,1.63],[24,'Cr',51.996,4,6,5,0,1.66],[25,'Mn',54.938,4,7,5,0,1.55],[26,'Fe',55.845,4,8,6,0,1.83],[27,'Co',58.933,4,9,7,0,1.88],[28,'Ni',58.693,4,10,8,0,1.91],
[29,'Cu',63.546,4,11,10,0,1.9],[30,'Zn',65.38,4,12,10,0,1.65],[31,'Ga',69.723,4,13,10,0,1.81],[32,'Ge',72.64,4,14,10,0,2.01],[33,'As',74.922,4,15,10,0,2.18],[34,'Se',78.96,4,16,10,0,2.55],
[35,'Br',79.904,4,17,10,0,2.96],[36,'Kr',83.798,4,18,10,0,2.966],[37,'Rb',85.468,5,1,0,0,0.82],[38,'Sr',87.62,5,2,0,0,0.95],[39,'Y',88.906,5,3,1,0,1.22],[40,'Zr',91.224,5,4,2,0,1.33],[41,'Nb',92.906,5,5,4,0,1.6],
[42,'Mo',95.96,5,6,5,0,2.16],[43,'Tc',98,5,7,5,0,1.9],[44,'Ru',101.07,5,8,7,0,2.2],[45,'Rh',102.906,5,9,8,0,2.28],[46,'Pd',106.42,5,10,10,0,2.2],[47,'Ag',107.868,5,11,10,0,1.93],[48,'Cd',112.411,5,12,10,0,1.69],
[49,'In',114.818,5,13,10,0,1.78],[50,'Sn',118.71,5,14,10,0,1.96],[51,'Sb',121.76,5,15,10,0,2.05],[52,'Te',127.6,5,16,10,0,2.1],[53,'I',126.904,5,17,10,0,2.66],[54,'Xe',131.293,5,18,10,0,2.582],
[55,'Cs',132.905,6,1,0,0,0.79],[56,'Ba',137.327,6,2,0,0,0.89],[57,'La',138.905,6,3,1,0,1.1],[58,'Ce',140.116,6,3,1,1,1.12],[59,'Pr',140.908,6,3,0,3,1.13],[60,'Nd',144.242,6,3,0,4,1.14],[61,'Pm',145,6,3,0,5,1.13],
[62,'Sm',150.36,6,3,0,6,1.17],[63,'Eu',151.964,6,3,0,7,1.2],[64,'Gd',157.25,6,3,1,7,1.2],[65,'Tb',158.925,6,3,0,9,1.22],[66,'Dy',162.5,6,3,0,10,1.23],[67,'Ho',164.93,6,3,0,11,1.24],[68,'Er',167.259,6,3,0,12,1.24],
[69,'Tm',168.934,6,3,0,13,1.25],[70,'Yb',173.054,6,3,0,14,1.1],[71,'Lu',174.967,6,3,1,14,1.27],[72,'Hf',178.49,6,4,2,14,1.3],[73,'Ta',180.948,6,5,3,14,1.5],[74,'W',183.84,6,6,4,14,2.36],[75,'Re',186.207,6,7,5,14,1.9],
[76,'Os',190.23,6,8,6,14,2.2],[77,'Ir',192.217,6,9,7,14,2.2],[78,'Pt',195.084,6,10,9,14,2.28],[79,'Au',196.967,6,11,10,14,2.54],[80,'Hg',200.59,6,12,10,14,2],[81,'Tl',204.383,6,13,10,14,1.62],
[82,'Pb',207.2,6,14,10,14,2.33],[83,'Bi',208.98,6,15,10,14,2.02],[84,'Po',210,6,16,10,14,2],[85,'At',210,6,17,10,14,2.2],[86,'Rn',222,6,18,10,14,2.6],[87,'Fr',223,7,1,0,0,0.7],[88,'Ra',226,7,2,0,0,0.89],
[89,'Ac',227,7,3,1,0,1.1],[90,'Th',232.038,7,3,2,0,1.3],[91,'Pa',231.036,7,3,1,2,1.5],[92,'U',238.029,7,3,1,3,1.38],[93,'Np',237,7,3,1,4,1.36],[94,'Pu',244,7,3,0,6,1.28],[95,'Am',243,7,3,0,7,1.3],
[96,'Cm',247,7,3,1,7,1.3],[97,'Bk',247,7,3,0,9,1.3],[98,'Cf',251,7,3,0,10,1.3],[99,'Es',252,7,3,0,11,1.3],[100,'Fm',257,7,3,0,12,1.3],[101,'Md',258,7,3,0,13,1.3],[102,'No',259,7,3,0,14,1.3]]

for element_list in element_data_list:
    globals()[element_list[1]] = Element(int(element_list[0]), float(element_list[2]), int(element_list[3]), int(element_list[4]), int(element_list[5]), int(element_list[6]), float(element_list[7]))


def split_gen(x): #this should split input composition strings into element strings and floats
	for f, s in re.findall(r'([\d.]+)|([^\d.]+)', x):
		if f:
			float(f)
			yield f
		else:
			yield s

# create a dictionary of the material using symbol and floats (e.g. for Be0.7Al0.3H2 it should be {'Be':0.7, 'Al':0.3, 'H':2}
# def element_dict_maker(split_gen_output):	
	# split_gen_output = list(split_gen_output)
	# element_dict = {}
	# n = 0
	# while n<len(split_gen_output):
		# number_of_element = float(split_gen_output[n+1])
		# element_dict[split_gen_output[n]] = number_of_element
		# n+=2
	# return element_dict

def element_dict_maker(material):
	elem_comp = re.findall(r'[A-Z][a-z]*|\d+(?:\.\d+)?', re.sub('[A-Z][a-z]*(?![\da-z])', r'\g<0>1', material)) # sub adds a 1 after all atoms not followed by a number.
	elems = []
	atoms = []
	for index, item in enumerate(elem_comp):
		if index % 2:
			atoms.append(item)
		else:
			elems.append(item)

	elem_comp_dict = {}
	for elem, atom in zip(elems, atoms):
		elem_comp_dict.update({elem: float(atom)})
		
	return elem_comp_dict



st.write("# Prediction of Formation Energy (eV)")

# incput features 'Ba', 'Ca', 'La', 'Co', 'Cu', 'Mn', 'Mo', 'Ni', 'Ti', 'W', 'Zn', 'Zr'

col1, col2, col3 = st.columns(3)

# getting user input


# inputs for inference: material (H Wt Frac, d character, f character and electronegativity will be calculated), bandgap, density, atomic density, magnetizon, temeprature, pressure
# "material" should be a string of alternating chemical symbols and floats/integers like 'Be0.7Al0.3H2'

material = col1.text_input("Enter material (e.g., Be0.7Al0.3H2)")
bandgap = col1.number_input("Enter Band Gap (eV)")
density = col1.number_input("Enter density (g/cm3")
atom_density = col1.number_input("Enter atomic density (ang3/atom)")
magnetization = col1.number_input("Enter Magnetization (bohr magneton/F.U.)")
#h_wt_frac = col1.number_input("Enter Hydrogen Weight Fraction")
#d_char = col1.number_input("Enter avg. d-char")
#f_char = col1.number_input("Enter avg. f-char")
#electronegativity = col1.number_input("Enter avg. electronegativity")
T = col1.number_input("Enter Temperature (degC)")
P = col1.number_input("Enter Absolute Pressure (atm.)")

element_string = material #this should be a string of alternating chemical symbols and floats/integers like 'Be0.7Al0.3H2'
#element_string = 'Be0.7Al0.3H2'
el_dict = element_dict_maker(element_string)
print("^^^^^^^^^^^^^^^^^^^^^^^ el_dict ^^^^^^^^^^^^^^^^^^", el_dict)

d_tot = 0
f_tot = 0
eneg_tot = 0
n_sites = 0 
mass_tot = 0

for element in el_dict:
	if element != 'H':
		obj = globals()[element]
		num_element = el_dict[element]
		print("obj.mass = ", obj.mass, type(obj.mass))
		print("num_element = ", num_element, type(num_element))
		mass_tot += obj.mass * num_element #non-hydrogen mass
		d_tot += obj.dchar * num_element
		f_tot += obj.fchar * num_element
		eneg_tot += obj.electronegativity * num_element
		n_sites += num_element

n_sites = max(n_sites, 1e-10)  #just to avoid division by 0 during app launch though it doesn't affect any calculations
#print('n_sites = ', n_sites)
#calculating average d-char/f-char/electronegativity
d_char = d_tot / n_sites
f_char = f_tot / n_sites
electronegativity = eneg_tot / n_sites
	
#print('el_dict = ', el_dict) 
if 'H' in el_dict.keys():
    mass_H = el_dict['H'] * H.mass
    h_wt_frac = mass_H / (mass_H + mass_tot)
else:
    h_wt_frac = 0
#print('h_wt_frac = ', h_wt_frac)

input_data = [bandgap, density, atom_density, magnetization, h_wt_frac, d_char, f_char, electronegativity, T, P]
#col_names= ['Egap', 'Dens', 'Dens_atom', 'Mag', 'H_wt_frac', 'D Char', 'F Char', 'Eneg', 'Temp.', 'Pres.']


input_array = np.array(input_data)
input_array_2d = np.array([input_array])


model = joblib.load('Sean/simpletreetest_model.pkl')
feat_names = ['Band Gap (eV)','Density (g/cm3)','Atomic Density (A3/atom)','Magnetizon (Bohr Magnetons/F.U.)','H Wt. Frac','d Character','f Character','Electronegativity','Temperature (degC)','Pressure (Atmospheres Absolute)']

#input_array_2d

df_pred = pd.DataFrame(input_array_2d,columns = feat_names)
#print(df_pred)

pca_loaded = joblib.load('Sean/pca_model.pkl')
scaler_loaded = joblib.load('Sean/scaler_model.pkl')


df_norm = scaler_loaded.transform(df_pred)  # scale features based on training data scalers
df_pca = pca_loaded.transform(df_norm)      # carry out the same PCA feature transform as the training data
df_poly = poly_feat_maker(df_pca)

# get the predictions from the input features
prediction = model.predict(df_poly)

if st.button('Predict'):
    
    st.write('<p class="big-font"> Predicted Formation Energy (eV) </p>', str(round(prediction[0], 4)), unsafe_allow_html=True)






