import pickle
import numpy as np
from sklearn.linear_model import Perceptron
import streamlit as st

model = pickle.load(open('per_model-66130701704.sav', 'rb'))

st.title("Iris Species Prediction using Perceptrin ")

x1 = st.slider('Select Input1', 0.0, 10.0, 3.0)
x2 = st.slider('Select Input2', 0.0, 10.0, 5.0)
x3 = st.slider('Select Input3', 0.0, 10.0, 4.0)
x4 = st.slider('Select Input4', 0.0, 10.0, 7.0)

xnew = np.array([[x1,x2,x3,x4]])
#xnew = np.array([6,3,5,3]).reshape(1,-1)

pred = model.predict(xnew)
st.write("## Preddiction Result:")
st.write('Speccies:',pred[0])
