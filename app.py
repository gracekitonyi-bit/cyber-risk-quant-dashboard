import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from risk_model import simulate_annual_losses, compute_risk_metrics

import os
import io
from datetime import datetime
import tempfile
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")
st.title("🛡️ Cyber Risk Quantification Dashboard")
st.caption("Monte Carlo cyber risk model • Controls ROI/ROSI • Stress testing • Sensitivity • PDF export")

# -----------------------------
# Small UI styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem;}
      .big-card {padding: 1rem 1.25rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03);}
      .muted {color: rgba(255,255,255,0.65);}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helper: safe logo loader
# -----------------------------
def load_logo_image():
    candidates = [
        "assets/aims_logo.png",
        "assets/logo.png",
        "assets/aimslogo.png",
        "assetslogo.png",     # (your screenshot shows this exists)
        "logo.png",
        "aims_logo.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return plt.imread(p)
            except Exception:
                pass
    return None

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
    help="Estimated annual cost of the selected controls (licenses, staff time, training, staff time, etc.)"
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
    losses_ = simulate_annual_losses(trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_)
    metrics_ = compute_risk_metrics(losses_)
    return metrics_["Expected Annual Loss"]

# -----------------------------
# Helper: make distribution figure
# -----------------------------
def make_distribution_figure(baseline_losses, baseline_metrics, controlled_losses=None, controlled_metrics=None):
    fig, ax = plt.subplots()
    ax.hist(baseline_losses, bins=50, alpha=0.6, label="Baseline")
    ax.axvline(baseline_metrics["VaR 95%"], linestyle="dashed", label="Baseline VaR 95%")

    if controlled_losses is not None and controlled_metrics is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.6, label="Controlled")
        ax.axvline(controlled_metrics["VaR 95%"], linestyle="dotted", label="Controlled VaR 95%")

    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_title("Annual Loss Distribution")
    return fig

# -----------------------------
# Helper: build tornado plot figure
# -----------------------------
def make_tornado_figure(labels, lows, highs, sens_pct):
    fig, ax = plt.subplots()
    y = np.arange(len(labels))
    ax.barh(y, highs, alpha=0.7, label=f"+{sens_pct}%")
    ax.barh(y, lows, alpha=0.7, label=f"-{sens_pct}%")
    ax.axvline(0, linestyle="dashed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in Expected Annual Loss ($)")
    ax.set_title("Sensitivity of EAL to each input (Tornado-style)")
    ax.legend()
    return fig

# -----------------------------
# Helper: PDF Export using Matplotlib only (NO extra libraries)
# -----------------------------
def export_pdf_report(
    logo_img,
    meta_title,
    stress_mode,
    lam_effective,
    baseline_metrics,
    controlled_metrics,
    use_controls,
    lambda_reduction,
    vuln_reduction,
    severity_reduction,
    annual_control_cost,
    rosi,
    net_benefit,
    exec_lines,
    dist_fig,
    tornado_fig=None
):
    # Build PDF into bytes (in-memory)
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Cover
        cover = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches-ish
        cover.clf()
        ax = cover.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # Logo
        if logo_img is not None:
            ax_logo = cover.add_axes([0.08, 0.86, 0.18, 0.10])
            ax_logo.axis("off")
            ax_logo.imshow(logo_img)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ax.text(0.08, 0.80, meta_title, fontsize=20, fontweight="bold")
        ax.text(0.08, 0.765, "Professional Executive Report", fontsize=12)
        ax.text(0.08, 0.735, f"Generated: {now}", fontsize=10)
        ax.text(0.08, 0.715, f"Stress scenario: {stress_mode}", fontsize=10)
        ax.text(0.08, 0.695, f"Effective threat frequency (λ): {lam_effective:.2f} events/year", fontsize=10)

        # Key metrics table (text)
        y0 = 0.63
        ax.text(0.08, y0, "Baseline Metrics", fontsize=13, fontweight="bold")
        ax.text(0.08, y0 - 0.03, f"EAL: ${baseline_metrics['Expected Annual Loss']:,.0f}", fontsize=11)
        ax.text(0.08, y0 - 0.055, f"Breach-Year: {baseline_metrics['Probability of Breach-Year']:.2%}", fontsize=11)
        ax.text(0.08, y0 - 0.08, f"VaR 95%: ${baseline_metrics['VaR 95%']:,.0f}", fontsize=11)

        if controlled_metrics is not None:
            ax.text(0.52, y0, "With Controls", fontsize=13, fontweight="bold")
            ax.text(0.52, y0 - 0.03, f"EAL: ${controlled_metrics['Expected Annual Loss']:,.0f}", fontsize=11)
            ax.text(0.52, y0 - 0.055, f"Breach-Year: {controlled_metrics['Probability of Breach-Year']:.2%}", fontsize=11)
            ax.text(0.52, y0 - 0.08, f"VaR 95%: ${controlled_metrics['VaR 95%']:,.0f}", fontsize=11)

        # Executive summary
        ax.text(0.08, 0.50, "Executive Summary (Plain Language)", fontsize=13, fontweight="bold")
        yy = 0.47
        for line in exec_lines:
            ax.text(0.08, yy, line, fontsize=10)
            yy -= 0.025

        ax.text(
            0.08, 0.06,
            "Note: This is a scenario-based Monte Carlo model. Results depend on your assumptions and inputs.",
            fontsize=9, alpha=0.8
        )

        pdf.savefig(cover, dpi=200, bbox_inches="tight")
        plt.close(cover)

        # Page 2: Controls Summary
        controls = plt.figure(figsize=(8.27, 11.69))
        controls.clf()
        ax = controls.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.08, 0.92, "Controls Summary", fontsize=18, fontweight="bold")

        if use_controls and controlled_metrics is not None:
            ax.text(0.08, 0.86, "Selected Control Effects", fontsize=13, fontweight="bold")
            ax.text(0.08, 0.83, f"Threat Reduction: {lambda_reduction:.0f}%", fontsize=11)
            ax.text(0.08, 0.805, f"Vulnerability Reduction: {vuln_reduction:.0f}%", fontsize=11)
            ax.text(0.08, 0.78, f"Impact Reduction: {severity_reduction:.0f}%", fontsize=11)

            ax.text(0.08, 0.73, "Investment (ROI/ROSI)", fontsize=13, fontweight="bold")
            ax.text(0.08, 0.70, f"Annual control cost: ${annual_control_cost:,.0f}", fontsize=11)

            if rosi is not None and not np.isnan(rosi):
                ax.text(0.08, 0.675, f"ROSI: {rosi:.2%}", fontsize=11)
            else:
                ax.text(0.08, 0.675, "ROSI: N/A", fontsize=11)

            if net_benefit is not None:
                ax.text(0.08, 0.65, f"Net benefit: ${net_benefit:,.0f} per year", fontsize=11)

        else:
            ax.text(0.08, 0.86, "Controls are OFF for this run.", fontsize=11)

        pdf.savefig(controls, dpi=200, bbox_inches="tight")
        plt.close(controls)

        # Page 3: Distribution Plot
        pdf.savefig(dist_fig, dpi=200, bbox_inches="tight")

        # Page 4: Tornado plot (optional)
        if tornado_fig is not None:
            pdf.savefig(tornado_fig, dpi=200, bbox_inches="tight")

    buffer.seek(0)
    return buffer.getvalue()

# -----------------------------
# Run simulation
# -----------------------------
run = st.button("Run Simulation")

# We store latest outputs in session_state so the PDF button can work after simulation
if "latest" not in st.session_state:
    st.session_state.latest = {}

if run:
    # Stress multiplier
    if stress_mode == "Normal Year":
        shock = 1.0
    elif stress_mode == "Elevated Threat Year":
        shock = 1.5
    elif stress_mode == "Ransomware Wave":
        shock = 2.5
    else:
        shock = custom_shock

    lam_effective = lam * shock
    st.info(f"Stress scenario: {stress_mode} → threat frequency λ = {lam_effective:.2f} events/year")

    # Baseline simulation
    baseline_losses = simulate_annual_losses(trials, lam_effective, p_vuln, asset_value, exposure_factor, sev_mu, sev_sigma)
    baseline_metrics = compute_risk_metrics(baseline_losses)

    # Controls simulation
    controlled_losses = None
    controlled_metrics = None

    if use_controls:
        lam_control = lam_effective * (1 - lambda_reduction / 100.0)
        p_control = p_vuln * (1 - vuln_reduction / 100.0)
        exposure_control = exposure_factor * (1 - severity_reduction / 100.0)

        controlled_losses = simulate_annual_losses(trials, lam_control, p_control, asset_value, exposure_control, sev_mu, sev_sigma)
        controlled_metrics = compute_risk_metrics(controlled_losses)

    # -----------------------------
    # Risk metrics
    # -----------------------------
    st.subheader("📊 Risk Metrics")

    rosi = None
    net_benefit = None

    if controlled_metrics is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Annual Loss (EAL)", f"${baseline_metrics['Expected Annual Loss']:,.0f}")
        c2.metric("Probability of Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
        c3.metric("VaR 95%", f"${baseline_metrics['VaR 95%']:,.0f}")
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Baseline")
            c1, c2, c3 = st.columns(3)
            c1.metric("EAL", f"${baseline_metrics['Expected Annual Loss']:,.0f}")
            c2.metric("Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
            c3.metric("VaR 95%", f"${baseline_metrics['VaR 95%']:,.0f}")

        with right:
            st.markdown("#### With Controls")
            c4, c5, c6 = st.columns(3)
            eal_delta = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
            c4.metric("EAL", f"${controlled_metrics['Expected Annual Loss']:,.0f}", delta=f"-${eal_delta:,.0f}")
            c5.metric("Breach-Year", f"{controlled_metrics['Probability of Breach-Year']:.2%}")
            c6.metric("VaR 95%", f"${controlled_metrics['VaR 95%']:,.0f}")

        # ROI / ROSI
        st.subheader("💰 ROI / ROSI (Investment Value)")
        risk_reduction = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
        net_benefit = risk_reduction - annual_control_cost
        rosi = (net_benefit / annual_control_cost) if annual_control_cost > 0 else np.nan

        a, b, c, d = st.columns(4)
        a.metric("Annual Risk Reduction", f"${risk_reduction:,.0f}")
        b.metric("Annual Control Cost", f"${annual_control_cost:,.0f}")
        c.metric("Net Benefit", f"${net_benefit:,.0f}")
        d.metric("ROSI", f"{rosi:.2%}" if not np.isnan(rosi) else "N/A")

        if net_benefit > 0:
            st.success("✅ Controls look cost-effective under these assumptions (net benefit > 0).")
        else:
            st.warning("⚠️ Controls may not be cost-effective under these assumptions (net benefit ≤ 0).")

    # -----------------------------
    # Executive Summary (NO markdown stars)
    # -----------------------------
    st.subheader("🧾 Executive Summary (Plain Language)")

    breach_base = baseline_metrics["Probability of Breach-Year"]
    eal_base = baseline_metrics["Expected Annual Loss"]
    var_base = baseline_metrics["VaR 95%"]

    exec_lines = [
        f"Baseline risk: About {breach_base:.0%} chance of at least one successful breach in a year.",
        f"Expected annual loss is about ${eal_base:,.0f}.",
        f"In a bad year (worst 5%), losses may exceed ${var_base:,.0f}.",
    ]

    if controlled_metrics is not None:
        breach_ctrl = controlled_metrics["Probability of Breach-Year"]
        eal_ctrl = controlled_metrics["Expected Annual Loss"]
        var_ctrl = controlled_metrics["VaR 95%"]

        breach_drop = breach_base - breach_ctrl
        eal_drop = eal_base - eal_ctrl
        var_drop = var_base - var_ctrl

        exec_lines += [
            "",
            f"With security controls: breach-year likelihood drops to about {breach_ctrl:.0%}.",
            f"Expected annual loss drops to about ${eal_ctrl:,.0f}.",
            f"Bad-year threshold (VaR 95%) drops to about ${var_ctrl:,.0f}.",
            "",
            f"Estimated impact: breach probability decreases by {breach_drop:.0%}.",
            f"Expected annual loss decreases by ${eal_drop:,.0f}.",
            f"VaR 95% decreases by ${var_drop:,.0f}.",
        ]

        if rosi is not None and net_benefit is not None and not np.isnan(rosi):
            exec_lines += [
                "",
                f"Investment view: controls cost ${annual_control_cost:,.0f} per year.",
                f"Estimated net benefit is ${net_benefit:,.0f} per year (ROSI {rosi:.0%}).",
            ]

    # Display summary cleanly
    st.markdown('<div class="big-card">', unsafe_allow_html=True)
    for line in exec_lines:
        if line.strip() == "":
            st.write("")
        else:
            st.write(line)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Scenario-based Monte Carlo model. Results depend on the assumptions you select.")

    # -----------------------------
    # Distribution plot
    # -----------------------------
    st.subheader("📈 Annual Loss Distribution")
    dist_fig = make_distribution_figure(baseline_losses, baseline_metrics, controlled_losses, controlled_metrics)
    st.pyplot(dist_fig)

    # -----------------------------
    # Sensitivity Analysis
    # -----------------------------
    tornado_fig = None
    if run_sensitivity:
        st.subheader("🧪 Sensitivity Analysis (What Drives Risk Most?)")
        st.write(f"We vary each input by ±{sens_pct}% and measure how Expected Annual Loss (EAL) changes.")

        base_eal = baseline_metrics["Expected Annual Loss"]
        delta = sens_pct / 100.0

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

        tornado_fig = make_tornado_figure(labels, lows, highs, sens_pct)
        st.pyplot(tornado_fig)

    # -----------------------------
    # Save outputs for PDF download
    # -----------------------------
    st.session_state.latest = {
        "stress_mode": stress_mode,
        "lam_effective": lam_effective,
        "baseline_metrics": baseline_metrics,
        "controlled_metrics": controlled_metrics,
        "use_controls": use_controls,
        "lambda_reduction": lambda_reduction,
        "vuln_reduction": vuln_reduction,
        "severity_reduction": severity_reduction,
        "annual_control_cost": annual_control_cost,
        "rosi": rosi,
        "net_benefit": net_benefit,
        "exec_lines": exec_lines,
        "dist_fig": dist_fig,
        "tornado_fig": tornado_fig,
    }

# -----------------------------
# PDF Export (works after a run)
# -----------------------------
st.markdown("---")
st.subheader("📄 Export Executive PDF (Professional)")

if not st.session_state.latest:
    st.info("Run the simulation first, then export the PDF report.")
else:
    if st.button("Generate PDF Report"):
        logo_img = load_logo_image()
        data = st.session_state.latest

        pdf_bytes = export_pdf_report(
            logo_img=logo_img,
            meta_title="Cyber Risk Quantification Dashboard",
            stress_mode=data["stress_mode"],
            lam_effective=data["lam_effective"],
            baseline_metrics=data["baseline_metrics"],
            controlled_metrics=data["controlled_metrics"],
            use_controls=data["use_controls"],
            lambda_reduction=data["lambda_reduction"],
            vuln_reduction=data["vuln_reduction"],
            severity_reduction=data["severity_reduction"],
            annual_control_cost=data["annual_control_cost"],
            rosi=data["rosi"],
            net_benefit=data["net_benefit"],
            exec_lines=data["exec_lines"],
            dist_fig=data["dist_fig"],
            tornado_fig=data["tornado_fig"],
        )

        st.download_button(
            label="⬇️ Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name="Cyber_Risk_Executive_Report.pdf",
            mime="application/pdf"
        )

st.caption("Tip: Put your AIMS logo in `assets/aims_logo.png` for the PDF cover page.")
