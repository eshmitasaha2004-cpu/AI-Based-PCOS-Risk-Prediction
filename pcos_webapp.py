import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI PCOS Detection System", layout="centered")

st.title("🩺 AI-Based PCOS Prediction System")
# Sidebar GitHub Link
st.sidebar.title("🔗 Project Links")
st.sidebar.markdown(
    "[GitHub Repository](https://github.com/eshmitasaha2004-cpu/AI-Based-PCOS-Risk-Prediction)"
)

st.markdown("Enter patient details below to predict PCOS risk.")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("pcos_50000_dataset.csv")
    return data

df = load_data()

# ----------------------------
# Train Model
# ----------------------------
X = df.drop("PCOS", axis=1)
y = df["PCOS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------------------
# User Input Section
# ----------------------------
st.header("Patient Information")

age = st.number_input("Age", 15, 50, 25)
bmi = st.number_input("BMI", 15.0, 45.0, 24.0)

hairfall = st.selectbox("Hairfall", [0, 1])
acne = st.selectbox("Acne", [0, 1])
irregular_cycle = st.selectbox("Irregular Cycle", [0, 1])
weight_gain = st.selectbox("Weight Gain", [0, 1])
hirsutism = st.selectbox("Hirsutism", [0, 1])

lh = st.number_input("LH Level", 0.0, 20.0, 5.0)
fsh = st.number_input("FSH Level", 0.0, 20.0, 5.0)
amh = st.number_input("AMH Level", 0.0, 20.0, 3.0)
cycle_length = st.number_input("Cycle Length (days)", 20, 60, 28)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict PCOS Risk"):

    input_data = np.array([[age, bmi, hairfall, acne,
                            irregular_cycle, weight_gain,
                            hirsutism, lh, fsh, amh, cycle_length]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ PCOS Detected")
        st.write(f"Risk Probability: {probability*100:.2f}%")
    else:
        st.success("✅ No PCOS Detected")
        st.write(f"Risk Probability: {(1-probability)*100:.2f}%")

# ----------------------------
# Feature Importance
# ----------------------------
st.header("Model Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))
st.bar_chart(importance.set_index("Feature"))
# ---------------- Disclaimer ----------------
st.markdown("⚠️ This tool is for educational purposes only and not a medical diagnosis.")


# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>👩‍💻 Developed by <b>Eshmita Saha</b> | "
    "<a href='https://github.com/eshmitasaha2004-cpu/AI-Based-PCOS-Risk-Prediction' target='_blank'>GitHub</a></div>",
    unsafe_allow_html=True
)

