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

asset_value = st.sidebar.number_input("Asset Value ($)", value=1000000, step=10000)
exposure_factor = st.sidebar.slider("Exposure Factor (0-1)", 0.1, 1.0, 0.5)

sev_mu = st.sidebar.slider("Severity Mean (lognormal μ)", 0.0, 2.0, 0.5)
sev_sigma = st.sidebar.slider("Severity Std (lognormal σ)", 0.1, 2.0, 1.0)

# -----------------------------
# Sidebar: Controls
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Security Controls")

use_controls = st.sidebar.checkbox("Enable Security Controls", value=True)

lambda_reduction = st.sidebar.slider("Threat Reduction (%)", 0, 90, 30)
vuln_reduction = st.sidebar.slider("Vulnerability Reduction (%)", 0, 90, 40)
severity_reduction = st.sidebar.slider("Impact Reduction (%)", 0, 90, 25)

# ROI inputs (only meaningful if controls are enabled)
st.sidebar.markdown("---")
st.sidebar.header("Investment (ROI)")

annual_control_cost = st.sidebar.number_input(
    "Annual control cost ($/year)",
    value=250000,
    step=10000,
    help="Estimated annual cost of the selected controls (licenses, staff time, training, etc.)"
)

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
    controlled_losses = None
    controlled_metrics = None

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

    # -----------------------------
    # Layout: Metrics
    # -----------------------------
    st.subheader("📊 Risk Metrics")

    if controlled_metrics is None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Loss", f"${baseline_metrics['Expected Annual Loss']:,.0f}")
        col2.metric("Probability of Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
        col3.metric("VaR 95%", f"${baseline_metrics['VaR 95%']:,.0f}")

    else:
        # Side-by-side comparison
        left, right = st.columns(2)

        with left:
            st.markdown("### Baseline")
            col1, col2, col3 = st.columns(3)
            col1.metric("EAL", f"${baseline_metrics['Expected Annual Loss']:,.0f}")
            col2.metric("Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
            col3.metric("VaR 95%", f"${baseline_metrics['VaR 95%']:,.0f}")

        with right:
            st.markdown("### With Controls")
            col4, col5, col6 = st.columns(3)
            col4.metric(
                "EAL",
                f"${controlled_metrics['Expected Annual Loss']:,.0f}",
                delta=f"-${baseline_metrics['Expected Annual Loss'] - controlled_metrics['Expected Annual Loss']:,.0f}"
            )
            col5.metric("Breach-Year", f"{controlled_metrics['Probability of Breach-Year']:.2%}")
            col6.metric("VaR 95%", f"${controlled_metrics['VaR 95%']:,.0f}")

        # -----------------------------
        # ROI / ROSI
        # -----------------------------
        st.subheader("💰 ROI / ROSI (Investment Value)")

        risk_reduction = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
        net_benefit = risk_reduction - annual_control_cost

        if annual_control_cost > 0:
            rosi = net_benefit / annual_control_cost
        else:
            rosi = np.nan

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Annual Risk Reduction", f"${risk_reduction:,.0f}")
        colB.metric("Annual Control Cost", f"${annual_control_cost:,.0f}")
        colC.metric("Net Benefit", f"${net_benefit:,.0f}")
        colD.metric("ROSI", f"{rosi:.2%}" if not np.isnan(rosi) else "N/A")

        # Decision message
        if net_benefit > 0:
            st.success("✅ Controls look cost-effective under these assumptions (net benefit > 0).")
        else:
            st.warning("⚠️ Controls may not be cost-effective under these assumptions (net benefit ≤ 0).")
        # -----------------------------
        # Executive Summary (plain language)
st.subheader("🧾 Executive Summary (Plain Language)")

st.write(f"""
Baseline risk:
The model estimates about {breach_base:.0%} chance of at least one successful breach in a year.
Expected annual loss is approximately ${eal_base:,.0f}.
In a bad year (worst 5%), losses may exceed ${var_base:,.0f}.

With security controls:
Breach-year likelihood drops to about {breach_ctrl:.0%}.
Expected annual loss drops to about ${eal_ctrl:,.0f}.
Bad-year threshold (VaR 95%) drops to ${var_ctrl:,.0f}.

Estimated impact of controls:
• Breach probability decreases by {breach_drop:.0%}
• Expected annual loss decreases by ${eal_drop:,.0f}
• VaR 95% decreases by ${var_drop:,.0f}

Investment view:
If controls cost ${annual_control_cost:,.0f} per year,
the estimated net benefit is ${net_benefit:,.0f} per year,
with a ROSI of {rosi:.0%}.
""")

st.caption("This is a scenario-based Monte Carlo model. Results depend on assumptions.")
    # -----------------------------
    # Distribution plot
    # -----------------------------
    st.subheader("📈 Annual Loss Distribution")

    fig, ax = plt.subplots()
    ax.hist(baseline_losses, bins=50, alpha=0.6, label="Baseline")

    # Baseline VaR line
    ax.axvline(baseline_metrics["VaR 95%"], linestyle="dashed", label="Baseline VaR 95%")

    if controlled_losses is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.6, label="Controlled")
        ax.axvline(controlled_metrics["VaR 95%"], linestyle="dotted", label="Controlled VaR 95%")

    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Set parameters on the left, then click **Run Simulation**.")
