import streamlit as st
import pickle
import numpy as np
import os

# Faylning mavjudligini tekshirish
if not os.path.exists('fayl.pkl'):
    st.error("Pickle fayli topilmadi! Iltimos, faylni to'g'ri joyga qo'ying.")
else:
    try:
        with open('fayl.pkl', 'rb') as f:
            model = pickle.load(f)
        st.title("XGBoost Modeli Bashorati")

        st.write("Quyidagi maydonlarni to'ldiring:")

        feature1 = st.number_input("Xususiyat 1:", value=0.0, format="%.2f")
        feature2 = st.number_input("Xususiyat 2:", value=0.0, format="%.2f")
        feature3 = st.number_input("Xususiyat 3:", value=0.0, format="%.2f")
        feature4 = st.number_input("Xususiyat 4:", value=0.0, format="%.2f")

        if st.button("Bashorat qilish"):
            user_input = np.array([[feature1, feature2, feature3, feature4]])

            prediction = model.predict(user_input)
            prediction_proba = model.predict_proba(user_input)

            st.write(f"Bashorat: {prediction[0]}")
            st.write(f"Bashorat ehtimolligi: {prediction_proba[0]}")

        st.write("Ushbu ilova XGBoost modeli yordamida ishlaydi va ma'lumotlaringizni tahlil qiladi.")

    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")
