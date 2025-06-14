import streamlit as st
import joblib
import numpy as np

# ------------------- LOAD MODEL DAN SCALER -------------------
model = joblib.load('best_model .pkl')
scaler = joblib.load('scaler.pkl')

st.title("Aplikasi Prediksi Klasifikasi (16 Fitur)")

st.markdown("Masukkan nilai 16 fitur berikut:")

# ------------------- INPUT 16 FITUR -------------------
inputs = []
for i in range(16):
    val = st.number_input(f"Fitur {i+1}", value=0.0)
    inputs.append(val)

input_array = np.array([inputs])  # shape (1, 16)

# ------------------- PREDIKSI -------------------
if st.button("Prediksi"):
    if np.all(np.array(inputs) == 0.0):
        st.warning("Silakan isi fitur terlebih dahulu.")
    else:
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Hasil Prediksi: {prediction}")
