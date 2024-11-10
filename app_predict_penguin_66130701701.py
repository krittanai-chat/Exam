import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.pipeline import Pipeline

# โหลดโมเดลที่ฝึกเสร็จแล้ว
with open('model_penguin_66130701701.pkl', 'rb') as file:
    model = pickle.load(file)

# ฟังก์ชันในการทำนาย
def predict_penguin_species(island, culmen_length, culmen_depth, flipper_length, body_mass, sex):
    # สร้าง DataFrame สำหรับข้อมูลที่ป้อนเข้า
    input_data = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length],
        'culmen_depth_mm': [culmen_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'sex': [sex]
    })
    
    # จัดเรียงคอลัมน์ให้ตรงกับลำดับที่โมเดลต้องการ
    input_data = input_data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    
    # ทำนายผลจากโมเดล
    prediction = model.predict(input_data)
    return prediction[0]

# การตั้งค่าหน้า Streamlit
st.title("การทำนายสปีชีส์ของเพนกวิน")
st.write("กรุณากรอกข้อมูลของเพนกวินเพื่อทำนายสปีชีส์")

# ช่องกรอกข้อมูลของผู้ใช้
island = st.selectbox("เกาะ", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length = st.number_input("ความยาว Culmen (mm)", min_value=0.0, max_value=100.0, value=39.1)
culmen_depth = st.number_input("ความลึก Culmen (mm)", min_value=0.0, max_value=100.0, value=18.7)
flipper_length = st.number_input("ความยาว Flipper (mm)", min_value=0, max_value=300, value=181)
body_mass = st.number_input("น้ำหนักตัว (g)", min_value=0, max_value=10000, value=3750)
sex = st.selectbox("เพศ", ['MALE', 'FEMALE'])

# เมื่อผู้ใช้คลิกที่ปุ่ม 'Predict'
if st.button("ทำนายสปีชีส์"):
    species = predict_penguin_species(island, culmen_length, culmen_depth, flipper_length, body_mass, sex)
    st.write(f"สปีชีส์ที่ทำนายคือ: {species}")
