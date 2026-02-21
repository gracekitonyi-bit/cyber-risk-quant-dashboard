import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from risk_model import simulate_annual_losses, compute_risk_metrics

from fpdf import FPDF
import tempfile
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")
st.title("🛡️ Cyber Risk Quantification Dashboard")
st.caption("Monte Carlo cyber risk model • Controls ROI/ROSI • Stress testing • Sensitivity • PDF export")

# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.sidebar.header("Simulation Inputs")

trials = st.sidebar.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)
lam = st.sidebar.slider("Threat Frequency (events/year)", 0.0, 20.0, 5.0)
p_vuln = st.sidebar.slider("Vulnerability Probability", 0.0, 1.0, 0.3)

asset_value = st.sidebar.number_input("Asset Value ($)", value=1000000, step=10000)
exposure_factor = st.sidebar.slider("Exposure Factor (0–1)", 0.1, 1.0, 0.5)

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

# -----------------------------
# Sidebar: ROI Inputs
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Investment (ROI)")

annual_control_cost = st.sidebar.number_input(
    "Annual control cost ($/year)",
    value=250000,
    step=10000,
    help="Estimated annual cost of the selected controls (licenses, staff time, training, etc.)"
)

# -----------------------------
# Sidebar: Sensitivity
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Sensitivity Analysis")
run_sensitivity = st.sidebar.checkbox("Run sensitivity analysis (slower)", value=False)
sens_pct = st.sidebar.slider("Sensitivity change (%)", 5, 50, 20)

# -----------------------------
# Sidebar: Stress Testing
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("Stress Testing")

stress_mode = st.sidebar.selectbox(
    "Stress Scenario",
    ["Normal Year", "Elevated Threat Year", "Ransomware Wave", "Custom"]
)

custom_shock = 1.0
if stress_mode == "Custom":
    custom_shock = st.sidebar.slider("Custom Threat Multiplier", 1.0, 5.0, 2.0)


# -----------------------------
# Helper: compute EAL quickly
# -----------------------------
def compute_eal(trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_):
    losses_ = simulate_annual_losses(
        trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_
    )
    metrics_ = compute_risk_metrics(losses_)
    return metrics_["Expected Annual Loss"]


# -----------------------------
# Helper: PDF generator
# -----------------------------
def build_pdf_report(
    stress_mode,
    lam_effective,
    baseline_metrics,
    baseline_summary_lines,
    dist_plot_path,
    controlled_metrics=None,
    controlled_summary_lines=None,
    dist_plot_control_path=None,
    rosi=None,
    annual_control_cost=None,
    tornado_plot_path=None,
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Cyber Risk Quantification - Executive Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Stress scenario: {stress_mode}", ln=True)
    pdf.cell(0, 8, f"Effective threat frequency (lambda): {lam_effective:.2f} events/year", ln=True)
    pdf.ln(2)

    # Baseline Metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Baseline Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "\n".join([
        f"EAL (Expected Annual Loss): ${baseline_metrics['Expected Annual Loss']:,.0f}",
        f"Probability of Breach-Year: {baseline_metrics['Probability of Breach-Year']:.2%}",
        f"VaR 95%: ${baseline_metrics['VaR 95%']:,.0f}",
    ]))
    pdf.ln(2)

    # Baseline Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Plain-Language Summary (Baseline)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "\n".join(baseline_summary_lines))
    pdf.ln(2)

    # Controlled section if present
    if controlled_metrics is not None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "With Controls", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "\n".join([
            f"EAL: ${controlled_metrics['Expected Annual Loss']:,.0f}",
            f"Probability of Breach-Year: {controlled_metrics['Probability of Breach-Year']:.2%}",
            f"VaR 95%: ${controlled_metrics['VaR 95%']:,.0f}",
        ]))
        pdf.ln(1)

        if annual_control_cost is not None and rosi is not None:
            pdf.multi_cell(0, 6, "\n".join([
                f"Annual control cost: ${annual_control_cost:,.0f}",
                f"ROSI: {rosi:.0%}",
            ]))
            pdf.ln(1)

        if controlled_summary_lines is not None:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Plain-Language Summary (With Controls)", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, "\n".join(controlled_summary_lines))
            pdf.ln(2)

    # Distribution Plot(s)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Annual Loss Distribution", ln=True)
    pdf.ln(1)

    # Insert baseline dist plot image
    if dist_plot_path and os.path.exists(dist_plot_path):
        pdf.image(dist_plot_path, w=180)
        pdf.ln(4)

    # Insert second plot if provided (combined plot still fine; but we keep optional)
    if dist_plot_control_path and os.path.exists(dist_plot_control_path):
        pdf.image(dist_plot_control_path, w=180)
        pdf.ln(4)

    # Tornado plot if present
    if tornado_plot_path and os.path.exists(tornado_plot_path):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Sensitivity Analysis (Tornado Plot)", ln=True)
        pdf.ln(2)
        pdf.image(tornado_plot_path, w=180)

    # Save to bytes
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# -----------------------------
# Run simulation
# -----------------------------
if st.button("Run Simulation"):

    # Apply stress multiplier
    if stress_mode == "Normal Year":
        shock = 1.0
    elif stress_mode == "Elevated Threat Year":
        shock = 1.5
    elif stress_mode == "Ransomware Wave":
        shock = 2.5
    else:
        shock = custom_shock

    lam_effective = lam * shock
    st.info(f"Stress scenario: **{stress_mode}** → threat frequency λ becomes **{lam_effective:.2f} events/year**")

    # -----------------------------
    # Baseline simulation
    # -----------------------------
    baseline_losses = simulate_annual_losses(
        trials, lam_effective, p_vuln, asset_value, exposure_factor, sev_mu, sev_sigma
    )
    baseline_metrics = compute_risk_metrics(baseline_losses)

    # -----------------------------
    # Controlled simulation (if enabled)
    # -----------------------------
    controlled_losses = None
    controlled_metrics = None

    if use_controls:
        lam_control = lam_effective * (1 - lambda_reduction / 100)  # stress-adjusted
        p_control = p_vuln * (1 - vuln_reduction / 100)
        exposure_control = exposure_factor * (1 - severity_reduction / 100)

        controlled_losses = simulate_annual_losses(
            trials, lam_control, p_control, asset_value, exposure_control, sev_mu, sev_sigma
        )
        controlled_metrics = compute_risk_metrics(controlled_losses)

    # -----------------------------
    # Risk metrics section
    # -----------------------------
    st.subheader("📊 Risk Metrics")

    rosi = None
    net_benefit = None

    if controlled_metrics is None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Loss (EAL)", f"${baseline_metrics['Expected Annual Loss']:,.0f}")
        col2.metric("Probability of Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
        col3.metric("VaR 95%", f"${baseline_metrics['VaR 95%']:,.0f}")

    else:
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
            eal_delta = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
            col4.metric(
                "EAL",
                f"${controlled_metrics['Expected Annual Loss']:,.0f}",
                delta=f"-${eal_delta:,.0f}"
            )
            col5.metric("Breach-Year", f"{controlled_metrics['Probability of Breach-Year']:.2%}")
            col6.metric("VaR 95%", f"${controlled_metrics['VaR 95%']:,.0f}")

        # ROI / ROSI
        st.subheader("💰 ROI / ROSI (Investment Value)")

        risk_reduction = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
        net_benefit = risk_reduction - annual_control_cost
        rosi = (net_benefit / annual_control_cost) if annual_control_cost > 0 else np.nan

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Annual Risk Reduction", f"${risk_reduction:,.0f}")
        colB.metric("Annual Control Cost", f"${annual_control_cost:,.0f}")
        colC.metric("Net Benefit", f"${net_benefit:,.0f}")
        colD.metric("ROSI", f"{rosi:.2%}" if not np.isnan(rosi) else "N/A")

        if net_benefit > 0:
            st.success("✅ Controls look cost-effective under these assumptions (net benefit > 0).")
        else:
            st.warning("⚠️ Controls may not be cost-effective under these assumptions (net benefit ≤ 0).")

    # -----------------------------
    # Executive Summary (Plain Language)
    # -----------------------------
    st.subheader("🧾 Executive Summary (Plain Language)")

    breach_base = baseline_metrics["Probability of Breach-Year"]
    eal_base = baseline_metrics["Expected Annual Loss"]
    var_base = baseline_metrics["VaR 95%"]

    baseline_summary_lines = [
        f"Baseline risk: About {breach_base:.0%} chance of at least one successful breach in a year.",
        f"Expected annual loss is approximately ${eal_base:,.0f}.",
        f"In a bad year (worst 5%), losses may exceed ${var_base:,.0f}.",
    ]

    controlled_summary_lines = None

    if controlled_metrics is not None:
        breach_ctrl = controlled_metrics["Probability of Breach-Year"]
        eal_ctrl = controlled_metrics["Expected Annual Loss"]
        var_ctrl = controlled_metrics["VaR 95%"]

        breach_drop = breach_base - breach_ctrl
        eal_drop = eal_base - eal_ctrl
        var_drop = var_base - var_ctrl

        controlled_summary_lines = [
            f"With security controls: breach-year likelihood drops to about {breach_ctrl:.0%}.",
            f"Expected annual loss drops to about ${eal_ctrl:,.0f}.",
            f"Bad-year threshold (VaR 95%) drops to ${var_ctrl:,.0f}.",
            "",
            f"Estimated impact: breach probability decreases by {breach_drop:.0%};",
            f"EAL decreases by ${eal_drop:,.0f}; VaR 95% decreases by ${var_drop:,.0f}.",
        ]

        st.write(
            f"""
Baseline risk: The model estimates about **{breach_base:.0%}** chance of at least one successful breach in a year.
Expected annual loss is approximately **${eal_base:,.0f}**.
In a bad year (worst 5%), losses may exceed **${var_base:,.0f}**.

With security controls: Breach-year likelihood drops to about **{breach_ctrl:.0%}**.
Expected annual loss drops to about **${eal_ctrl:,.0f}**.
Bad-year threshold (VaR 95%) drops to **${var_ctrl:,.0f}**.

Estimated impact of controls:
• Breach probability decreases by **{breach_drop:.0%}**
• Expected annual loss decreases by **${eal_drop:,.0f}**
• VaR 95% decreases by **${var_drop:,.0f}**
"""
        )

        if rosi is not None and net_benefit is not None:
            st.write(
                f"""
Investment view: If controls cost **${annual_control_cost:,.0f} per year**,
the estimated net benefit is **${net_benefit:,.0f} per year**,
with a ROSI of **{rosi:.0%}**.
"""
            )

    else:
        st.write(
            f"""
Baseline risk: The model estimates about **{breach_base:.0%}** chance of at least one successful breach in a year.
Expected annual loss is approximately **${eal_base:,.0f}**.
In a bad year (worst 5%), losses may exceed **${var_base:,.0f}**.
"""
        )

    st.caption("Scenario-based Monte Carlo model. Results depend on the assumptions you select.")

    # -----------------------------
    # Distribution plot
    # -----------------------------
    st.subheader("📈 Annual Loss Distribution")

    fig, ax = plt.subplots()
    ax.hist(baseline_losses, bins=50, alpha=0.6, label="Baseline")
    ax.axvline(baseline_metrics["VaR 95%"], linestyle="dashed", label="Baseline VaR 95%")

    if controlled_losses is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.6, label="Controlled")
        ax.axvline(controlled_metrics["VaR 95%"], linestyle="dotted", label="Controlled VaR 95%")

    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Save distribution plot to temp for PDF
    dist_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(dist_tmp.name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tornado_path = None

    # -----------------------------
    # Sensitivity Analysis
    # -----------------------------
    if run_sensitivity:
        st.subheader("🧪 Sensitivity Analysis (What Drives Risk Most?)")
        st.write(f"We vary each input by ±{sens_pct}% and measure how Expected Annual Loss (EAL) changes.")

        base_eal = baseline_metrics["Expected Annual Loss"]
        delta = sens_pct / 100

        params = [
            ("Threat frequency (λ)", "lam"),
            ("Vulnerability probability (p)", "p_vuln"),
            ("Asset value", "asset_value"),
            ("Exposure factor", "exposure_factor"),
            ("Severity mean (μ)", "sev_mu"),
            ("Severity std (σ)", "sev_sigma"),
        ]

        results = []

        for label, name in params:
            lam_low, lam_high = lam_effective, lam_effective
            p_low, p_high = p_vuln, p_vuln
            av_low, av_high = float(asset_value), float(asset_value)
            ef_low, ef_high = exposure_factor, exposure_factor
            mu_low, mu_high = sev_mu, sev_mu
            sg_low, sg_high = sev_sigma, sev_sigma

            if name == "lam":
                lam_low = lam_effective * (1 - delta)
                lam_high = lam_effective * (1 + delta)
            elif name == "p_vuln":
                p_low = max(0.0, p_vuln * (1 - delta))
                p_high = min(1.0, p_vuln * (1 + delta))
            elif name == "asset_value":
                av_low = av_low * (1 - delta)
                av_high = av_high * (1 + delta)
            elif name == "exposure_factor":
                ef_low = max(0.0, exposure_factor * (1 - delta))
                ef_high = min(1.0, exposure_factor * (1 + delta))
            elif name == "sev_mu":
                mu_low = max(0.0, sev_mu * (1 - delta))
                mu_high = sev_mu * (1 + delta)
            elif name == "sev_sigma":
                sg_low = max(0.1, sev_sigma * (1 - delta))
                sg_high = sev_sigma * (1 + delta)

            eal_low = compute_eal(trials, lam_low, p_low, av_low, ef_low, mu_low, sg_low)
            eal_high = compute_eal(trials, lam_high, p_high, av_high, ef_high, mu_high, sg_high)

            impact_low = eal_low - base_eal
            impact_high = eal_high - base_eal
            span = max(abs(impact_low), abs(impact_high))

            results.append((label, impact_low, impact_high, span))

        results.sort(key=lambda x: x[3], reverse=True)

        labels = [r[0] for r in results]
        lows = [r[1] for r in results]
        highs = [r[2] for r in results]

        fig2, ax2 = plt.subplots()
        y = np.arange(len(labels))

        ax2.barh(y, highs, alpha=0.7, label=f"+{sens_pct}%")
        ax2.barh(y, lows, alpha=0.7, label=f"-{sens_pct}%")
        ax2.axvline(0, linestyle="dashed")

        ax2.set_yticks(y)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel("Change in Expected Annual Loss ($)")
        ax2.set_title("Sensitivity of EAL to each input (Tornado-style)")
        ax2.legend()

        st.pyplot(fig2)

        # Save tornado plot for PDF
        tornado_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig2.savefig(tornado_tmp.name, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        tornado_path = tornado_tmp.name

    # -----------------------------
    # PDF Export
    # -----------------------------
    st.subheader("📄 Export Executive PDF")

    if st.button("Generate PDF Report"):
        pdf_path = build_pdf_report(
            stress_mode=stress_mode,
            lam_effective=lam_effective,
            baseline_metrics=baseline_metrics,
            baseline_summary_lines=baseline_summary_lines,
            dist_plot_path=dist_tmp.name,
            controlled_metrics=controlled_metrics,
            controlled_summary_lines=controlled_summary_lines,
            dist_plot_control_path=None,
            rosi=rosi,
            annual_control_cost=annual_control_cost if controlled_metrics is not None else None,
            tornado_plot_path=tornado_path
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Executive Report (PDF)",
                data=f.read(),
                file_name="Cyber_Risk_Executive_Report.pdf",
                mime="application/pdf"
            )

        # Clean up pdf file (images are cleaned by OS later; safe to leave)
        try:
            os.remove(pdf_path)
        except Exception:
            pass

else:
    st.info("Set parameters on the left, then click **Run Simulation**.")
