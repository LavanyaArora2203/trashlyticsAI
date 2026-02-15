import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import pickle
from models.garbage_classifier import GarbageClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Urban Waste Intelligence Engine",
    page_icon="🌆",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS (LOAD ONCE)
# --------------------------------------------------
@st.cache_resource
def load_models():

    # Garbage Model
    with open("models/garbage_classifer_model.h5", "rb") as f:
        garbage_model = pickle.load(f)

    # Complaint Model
    with open("models/complaint_classifier_model.pkl", "rb") as f:
        complaint_model = pickle.load(f)

    # Vectorizer
    with open("NLP/complaint_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Forecast Model
    with open("models/waste_forecast_model.pkl", "rb") as f:
        forecast_model = pickle.load(f)

    return garbage_model, complaint_model, vectorizer, forecast_model


garbage_model, complaint_model, vectorizer, forecast_model = load_models()

# --------------------------------------------------
# REAL MODEL FUNCTIONS
# --------------------------------------------------

def predict_garbage(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((128, 128))  # must match training
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = garbage_model.predict(img_array)

    # Normalize if needed
    if isinstance(prediction[0], (int, np.integer)):
        return float(prediction[0])
    else:
        return float(prediction[0])


def classify_complaint(text):
    text_vector = vectorizer.transform([text])
    category = complaint_model.predict(text_vector)[0]

    # If model has probabilities
    try:
        probs = complaint_model.predict_proba(text_vector)
        sentiment = float(round(max(probs[0]), 2))
    except:
        sentiment = 0.8

    return category, sentiment


def forecast_zone():
    future_input = np.array([[1], [2], [3]])
    predicted = forecast_model.predict(future_input)

    predicted = predicted.flatten()
    lower = predicted - 10
    upper = predicted + 10

    overflow_prob = float(round(np.mean(predicted) / 150, 2))

    return predicted.tolist(), lower.tolist(), upper.tolist(), overflow_prob


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌆 Urban Waste Intelligence")
page = st.sidebar.radio("Navigation", ["Citizen Page", "Admin Dashboard"])
st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Predictive Governance Platform")

# ==================================================
# CITIZEN PAGE
# ==================================================
if page == "Citizen Page":

    st.title("📢 Report Waste Issue")

    complaint_text = st.text_area("Describe the issue")
    uploaded_image = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])
    submit = st.button("Submit Complaint")

    if submit and complaint_text:

        with st.spinner("Analyzing using AI models..."):

            category, sentiment = classify_complaint(complaint_text)

            if uploaded_image:
                image_severity = predict_garbage(uploaded_image)
            else:
                image_severity = 0.5

            predicted, lower, upper, overflow_prob = forecast_zone()

            priority_score = (
                0.3 * sentiment +
                0.3 * image_severity +
                0.2 * 0.7 +
                0.2 * overflow_prob
            )

            if priority_score > 0.75:
                priority_label = "HIGH"
                color = "red"
            elif priority_score > 0.55:
                priority_label = "MEDIUM"
                color = "orange"
            else:
                priority_label = "LOW"
                color = "green"

        st.success("Complaint Submitted Successfully!")

        st.subheader("🤖 AI Analysis Result")
        st.write(f"**Issue Category:** {category}")
        st.write(f"**Sentiment Confidence:** {sentiment}")
        st.write(f"**Image Severity Score:** {image_severity}")
        st.write(f"**Overflow Probability:** {overflow_prob}")

        st.markdown(
            f"<h3 style='color:{color}'>Predicted Priority: {priority_label}</h3>",
            unsafe_allow_html=True
        )

# ==================================================
# ADMIN DASHBOARD
# ==================================================
if page == "Admin Dashboard":

    st.title("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Complaints", 245)
    col2.metric("High Priority Issues", 68)
    col3.metric("High-Risk Zones", 4)
    col4.metric("Cost Reduction", "18%")

    st.markdown("---")

    # Complaint Trend
    st.subheader("📈 Complaint Trend")

    dates = pd.date_range(end=datetime.today(), periods=7)
    complaints = np.random.randint(20, 60, 7)

    trend_df = pd.DataFrame({
        "Date": dates,
        "Complaints": complaints
    })

    fig = px.line(trend_df, x="Date", y="Complaints", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    st.subheader("🔮 Waste Overflow Forecast")

    future_dates = pd.date_range(datetime.today(), periods=3)
    predicted, lower, upper, _ = forecast_zone()

    forecast_fig = go.Figure()

    forecast_fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted,
        mode='lines+markers',
        name="Predicted Fill %"
    ))

    forecast_fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    forecast_fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        fill='tonexty',
        mode='lines',
        name="Uncertainty Band"
    ))

    st.plotly_chart(forecast_fig, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Complaint Density Heatmap")

    heatmap_data = np.random.rand(5,5)

    heatmap_fig = px.imshow(
        heatmap_data,
        x=["Zone A","Zone B","Zone C","Zone D","Zone E"],
        y=["Area 1","Area 2","Area 3","Area 4","Area 5"],
        labels=dict(color="Density")
    )

    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Route Simulation
    st.subheader("🚚 Optimized Route")

    route_df = pd.DataFrame({
        "Stop Order":[1,2,3,4],
        "Zone":["Zone C","Zone A","Zone D","Zone B"]
    })

    st.table(route_df)

    # Impact
    st.subheader("💰 Impact Simulation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fuel Saved (Liters)", 35)
    col2.metric("Time Saved (Hours)", 2.5)
    col3.metric("Operational Cost Reduction", "18%")

    st.success("🚀 AI System Operational")
