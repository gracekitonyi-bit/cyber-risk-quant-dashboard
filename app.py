import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from risk_model import simulate_annual_losses, compute_risk_metrics

st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")

st.title("🛡️ Cyber Risk Quantification Dashboard")

st.sidebar.header("Simulation Inputs")

trials = st.sidebar.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)

lam = st.sidebar.slider("Threat Frequency (events/year)", 0.0, 20.0, 5.0)

p_vuln = st.sidebar.slider("Vulnerability Probability", 0.0, 1.0, 0.3)

asset_value = st.sidebar.number_input("Asset Value ($)", value=1000000)

exposure_factor = st.sidebar.slider("Exposure Factor (0-1)", 0.1, 1.0, 0.5)

sev_mu = st.sidebar.slider("Severity Mean (lognormal μ)", 0.0, 2.0, 0.5)

sev_sigma = st.sidebar.slider("Severity Std (lognormal σ)", 0.1, 2.0, 1.0)

if st.button("Run Simulation"):

    losses = simulate_annual_losses(
        trials,
        lam,
        p_vuln,
        asset_value,
        exposure_factor,
        sev_mu,
        sev_sigma
    )

    metrics = compute_risk_metrics(losses)

    st.subheader("📊 Risk Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Expected Annual Loss", f"${metrics['Expected Annual Loss']:,.0f}")
    col2.metric("Probability of Breach-Year", f"{metrics['Probability of Breach-Year']:.2%}")
    col3.metric("VaR 95%", f"${metrics['VaR 95%']:,.0f}")

    st.subheader("📈 Annual Loss Distribution")

    fig, ax = plt.subplots()
    ax.hist(losses, bins=50)
    ax.axvline(metrics["VaR 95%"], linestyle="dashed")
    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
