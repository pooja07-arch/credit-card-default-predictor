import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("💳 Credit Card Default Predictor")
st.write(
    "Predict whether a customer is likely to default "
    "on their next month's credit card payment."
)

# -----------------------------
# CREATE SAMPLE DATASET
# -----------------------------
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(21, 70, n),
    "Credit_Limit": np.random.randint(10000, 500000, n),
    "Bill_Amount": np.random.randint(1000, 300000, n),
    "Payment_Amount": np.random.randint(500, 200000, n),
    "Previous_Payment_Status": np.random.randint(-2, 4, n)
})

# Create target using simple financial conditions
data["Default"] = (
    (
        (data["Payment_Amount"] < data["Bill_Amount"] * 0.2)
        & (data["Previous_Payment_Status"] > 1)
    )
    |
    (
        data["Credit_Limit"] < data["Bill_Amount"] * 1.2
    )
).astype(int)

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data[
    [
        "Age",
        "Credit_Limit",
        "Bill_Amount",
        "Payment_Amount",
        "Previous_Payment_Status"
    ]
]

y = data["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📊 Model Information")

st.sidebar.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.write("Model Used:")
st.sidebar.write("🌲 Random Forest")

# -----------------------------
# CUSTOMER INPUT
# -----------------------------
st.header("👤 Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

credit_limit = st.number_input(
    "Credit Limit",
    min_value=1000,
    max_value=1000000,
    value=50000
)

bill_amount = st.number_input(
    "Current Bill Amount",
    min_value=0,
    max_value=1000000,
    value=30000
)

payment_amount = st.number_input(
    "Payment Amount",
    min_value=0,
    max_value=1000000,
    value=5000
)

previous_status = st.selectbox(
    "Previous Payment Status",
    [-2, -1, 0, 1, 2, 3],
    index=4
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Default"):

    customer = pd.DataFrame({
        "Age": [age],
        "Credit_Limit": [credit_limit],
        "Bill_Amount": [bill_amount],
        "Payment_Amount": [payment_amount],
        "Previous_Payment_Status": [previous_status]
    })

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0]

    no_default_probability = probability[0] * 100
    default_probability = probability[1] * 100

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "No Default",
            f"{no_default_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Default",
            f"{default_probability:.2f}%"
        )

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to DEFAULT")
    else:
        st.success("✅ Prediction: Customer is likely to NOT DEFAULT")

    st.progress(int(default_probability))

    st.write(
        f"**Default Probability:** "
        f"{default_probability:.2f}%"
    )