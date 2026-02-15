import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Urban Waste Intelligence Engine",
    page_icon="🌆",
    layout="wide"
)

# --------------------------------------------------
# CLEAN CSS FOR PRODUCT LOOK
# --------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size:28px !important;
    font-weight:600;
}
.section-title {
    font-size:20px !important;
    font-weight:600;
    margin-top:20px;
}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIMULATED AI FUNCTIONS
# --------------------------------------------------

def predict_garbage(image):
    return round(random.uniform(0.5, 0.95), 2)

def classify_complaint(text):
    category = random.choice(["Overflow", "Missed Pickup", "Illegal Dumping"])
    sentiment = round(random.uniform(0.6, 0.95), 2)
    return category, sentiment

def forecast_zone():
    predicted = [82, 95, 108]
    lower = [75, 88, 98]
    upper = [90, 105, 120]
    overflow_prob = 0.66
    return predicted, lower, upper, overflow_prob

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

    st.markdown('<div class="big-title">📢 Report Waste Issue</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:
        complaint_text = st.text_area("Describe the issue")
        uploaded_image = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])
        submit = st.button("Submit Complaint")

    if submit and complaint_text:

        with st.spinner("Analyzing using AI models..."):
            category, sentiment = classify_complaint(complaint_text)

            if uploaded_image:
                image = Image.open(uploaded_image)
                image_severity = predict_garbage(image)
            else:
                image_severity = round(random.uniform(0.4, 0.8), 2)

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

        st.markdown("### 🤖 AI Analysis Result")
        st.write(f"**Issue Category:** {category}")
        st.write(f"**Sentiment Score:** {sentiment}")
        st.write(f"**Image Severity:** {image_severity}")
        st.write(f"**Overflow Probability:** {overflow_prob}")

        st.markdown(
            f"<h3 style='color:{color}'>Predicted Priority: {priority_label}</h3>",
            unsafe_allow_html=True
        )

# ==================================================
# ADMIN DASHBOARD
# ==================================================
if page == "Admin Dashboard":

    st.markdown('<div class="big-title">📊Dashboard</div>', unsafe_allow_html=True)

    # ---------------- KPI CARDS ----------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Complaints", 245)
    col2.metric("High Priority Issues", 68)
    col3.metric("High-Risk Zones", 4)
    col4.metric("Cost Reduction", "18%")

    st.markdown("---")

    # ---------------- Complaint Trend ----------------
    st.markdown('<div class="section-title">📈 Complaint Trend</div>', unsafe_allow_html=True)

    dates = pd.date_range(end=datetime.today(), periods=7)
    complaints = np.random.randint(20, 60, 7)

    trend_df = pd.DataFrame({
        "Date": dates,
        "Complaints": complaints
    })

    fig = px.line(trend_df, x="Date", y="Complaints", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Forecast Graph ----------------
    st.markdown('<div class="section-title">🔮 Waste Overflow Forecast</div>', unsafe_allow_html=True)

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

    # ---------------- Heatmap ----------------
    st.markdown('<div class="section-title">🔥 Complaint Density Heatmap</div>', unsafe_allow_html=True)

    heatmap_data = np.random.rand(5,5)

    heatmap_fig = px.imshow(
        heatmap_data,
        x=["Zone A","Zone B","Zone C","Zone D","Zone E"],
        y=["Area 1","Area 2","Area 3","Area 4","Area 5"],
        labels=dict(color="Density")
    )

    st.plotly_chart(heatmap_fig, use_container_width=True)

    # ---------------- Route UI ----------------
    st.markdown('<div class="section-title">🚚 Optimized Route (Simulation)</div>', unsafe_allow_html=True)

    route_df = pd.DataFrame({
        "Stop Order":[1,2,3,4],
        "Zone":["Zone C","Zone A","Zone D","Zone B"]
    })

    st.table(route_df)

    # ---------------- Impact Simulation ----------------
    st.markdown('<div class="section-title">💰 Impact Simulation</div>', unsafe_allow_html=True)

    fuel_saved = 35
    time_saved = 2.5
    cost_saved = 18

    col1, col2, col3 = st.columns(3)
    col1.metric("Fuel Saved (Liters)", fuel_saved)
    col2.metric("Time Saved (Hours)", time_saved)
    col3.metric("Operational Cost Reduction", f"{cost_saved}%")

    st.success("🚀 System reduces operational cost by 18%")
