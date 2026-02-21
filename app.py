import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from risk_model import simulate_annual_losses, compute_risk_metrics

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")

st.title("🛡️ Cyber Risk Quantification Dashboard")

# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.sidebar.header("Simulation Inputs")

trials = st.sidebar.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)

lam = st.sidebar.slider("Threat Frequency (events/year)", 0.0, 20.0, 5.0)

p_vuln = st.sidebar.slider("Vulnerability Probability", 0.0, 1.0, 0.3)

asset_value = st.sidebar.number_input("Asset Value ($)", value=1000000)

exposure_factor = st.sidebar.slider("Exposure Factor (0-1)", 0.1, 1.0, 0.5)

sev_mu = st.sidebar.slider("Severity Mean (lognormal μ)", 0.0, 2.0, 0.5)

sev_sigma = st.sidebar.slider("Severity Std (lognormal σ)", 0.1, 2.0, 1.0)

# -----------------------------
# Sidebar: Controls
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Security Controls")

use_controls = st.sidebar.checkbox("Enable Security Controls")

lambda_reduction = st.sidebar.slider("Threat Reduction (%)", 0, 90, 30)
vuln_reduction = st.sidebar.slider("Vulnerability Reduction (%)", 0, 90, 40)
severity_reduction = st.sidebar.slider("Impact Reduction (%)", 0, 90, 25)

# -----------------------------
# Run simulation
# -----------------------------
if st.button("Run Simulation"):

    # Baseline simulation
    baseline_losses = simulate_annual_losses(
        trials,
        lam,
        p_vuln,
        asset_value,
        exposure_factor,
        sev_mu,
        sev_sigma
    )

    baseline_metrics = compute_risk_metrics(baseline_losses)

    # Apply controls if enabled
    if use_controls:
        lam_control = lam * (1 - lambda_reduction / 100)
        p_control = p_vuln * (1 - vuln_reduction / 100)
        exposure_control = exposure_factor * (1 - severity_reduction / 100)

        controlled_losses = simulate_annual_losses(
            trials,
            lam_control,
            p_control,
            asset_value,
            exposure_control,
            sev_mu,
            sev_sigma
        )

        controlled_metrics = compute_risk_metrics(controlled_losses)
    else:
        controlled_losses = None
        controlled_metrics = None

    # -----------------------------
    # Display metrics
    # -----------------------------
    st.subheader("📊 Risk Metrics (Baseline)")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Expected Annual Loss",
        f"${baseline_metrics['Expected Annual Loss']:,.0f}"
    )

    col2.metric(
        "Probability of Breach-Year",
        f"{baseline_metrics['Probability of Breach-Year']:.2%}"
    )

    col3.metric(
        "VaR 95%",
        f"${baseline_metrics['VaR 95%']:,.0f}"
    )

    if controlled_metrics is not None:
        st.subheader("🛡️ Risk Metrics (With Controls)")

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Expected Annual Loss",
            f"${controlled_metrics['Expected Annual Loss']:,.0f}",
            delta=f"-${baseline_metrics['Expected Annual Loss'] - controlled_metrics['Expected Annual Loss']:,.0f}"
        )

        col5.metric(
            "Probability of Breach-Year",
            f"{controlled_metrics['Probability of Breach-Year']:.2%}"
        )

        col6.metric(
            "VaR 95%",
            f"${controlled_metrics['VaR 95%']:,.0f}"
        )

    # -----------------------------
    # Display distribution
    # -----------------------------
    st.subheader("📈 Annual Loss Distribution")

    fig, ax = plt.subplots()

    ax.hist(baseline_losses, bins=50, alpha=0.6, label="Baseline")

    if controlled_losses is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.6, label="Controlled")

    ax.axvline(baseline_metrics["VaR 95%"], linestyle="dashed")

    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("Set parameters on the left, then click **Run Simulation**.")
