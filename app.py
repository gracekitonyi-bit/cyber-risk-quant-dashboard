import os
import io
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from risk_model import simulate_annual_losses, compute_risk_metrics


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")

st.title("🛡️ Cyber Risk Quantification Dashboard")
st.caption("Monte Carlo cyber risk model • Controls ROI/ROSI • Stress testing • Sensitivity • PDF export")


# =========================================================
# Helpers
# =========================================================
def find_logo_path() -> str | None:
    """Try to locate the logo in common places/names (including accidental names)."""
    candidates = [
        "assets/aims_logo.png",
        "assets/aims_logo.jpg",
        "assets/aims_logo.jpeg",
        "assets/logo.png",
        "assets/logo.jpg",
        "assets/logo.jpeg",
        # Your screenshot shows weird names like these:
        "assets/assetsaims_logo.png",
        "assets/assetsaims_logo.jpg",
        "assets/assetslogo.png",
        "assetslogo.png",
        "assetslogo.jpg",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_eal(trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_):
    losses_ = simulate_annual_losses(
        trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_
    )
    metrics_ = compute_risk_metrics(losses_)
    return metrics_["Expected Annual Loss"]


def make_distribution_plot(baseline_losses, baseline_var95, controlled_losses=None, controlled_var95=None):
    fig, ax = plt.subplots()
    ax.hist(baseline_losses, bins=50, alpha=0.6, label="Baseline")
    ax.axvline(baseline_var95, linestyle="dashed", label="Baseline VaR 95%")

    if controlled_losses is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.6, label="Controlled")
        ax.axvline(controlled_var95, linestyle="dotted", label="Controlled VaR 95%")

    ax.set_xlabel("Annual Loss ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


def try_build_pdf_report_reportlab(payload: dict) -> bytes | None:
    """
    Build a professional PDF using reportlab.
    Returns bytes if reportlab is available, otherwise None.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None

    # Unpack
    logo_path = payload.get("logo_path")
    created_at = payload["created_at"]
    stress_mode = payload["stress_mode"]
    shock = payload["shock"]
    lam_effective = payload["lam_effective"]

    inputs = payload["inputs"]
    baseline_metrics = payload["baseline_metrics"]
    baseline_summary_lines = payload["baseline_summary_lines"]

    controlled_metrics = payload.get("controlled_metrics")
    controlled_summary_lines = payload.get("controlled_summary_lines")

    controls_params = payload.get("controls_params")
    roi = payload.get("roi")

    dist_png_bytes = payload.get("dist_png_bytes")
    tornado_png_bytes = payload.get("tornado_png_bytes")

    # Create PDF in memory
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def header(title: str, subtitle: str | None = None):
        # Logo
        if logo_path and os.path.exists(logo_path):
            try:
                img = ImageReader(logo_path)
                c.drawImage(img, 1.5 * cm, h - 3.0 * cm, width=5.5 * cm, height=2.0 * cm, mask="auto")
            except Exception:
                pass

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1.5 * cm, h - 3.6 * cm, title)

        if subtitle:
            c.setFont("Helvetica", 11)
            c.drawString(1.5 * cm, h - 4.3 * cm, subtitle)

        # Date/time
        c.setFont("Helvetica", 9)
        c.drawRightString(w - 1.5 * cm, h - 2.0 * cm, f"Generated: {created_at}")

        # Line
        c.line(1.5 * cm, h - 4.7 * cm, w - 1.5 * cm, h - 4.7 * cm)

    def write_block(x, y, lines, font="Helvetica", size=10, leading=13):
        c.setFont(font, size)
        cur_y = y
        for line in lines:
            if cur_y < 2.0 * cm:
                c.showPage()
                header("Cyber Risk Quantification Report", "Continued")
                cur_y = h - 5.5 * cm
                c.setFont(font, size)
            c.drawString(x, cur_y, line)
            cur_y -= leading
        return cur_y

    # ---------------- Cover/Executive page
    header(
        "Cyber Risk Quantification Report",
        "Monte Carlo simulation • Loss distribution • Controls ROI/ROSI • Plain-language summary"
    )

    y = h - 5.5 * cm
    y = write_block(
        1.5 * cm,
        y,
        [
            f"Scenario: {stress_mode}  (Threat shock multiplier = {shock:.2f})",
            f"Effective threat frequency (λ): {lam_effective:.2f} events/year",
            "",
            "INPUTS",
            f"• Trials: {inputs['trials']:,}",
            f"• Base threat frequency (λ): {inputs['lam']:.2f}",
            f"• Vulnerability probability (p): {inputs['p_vuln']:.2f}",
            f"• Asset value: {fmt_money(inputs['asset_value'])}",
            f"• Exposure factor: {inputs['exposure_factor']:.2f}",
            f"• Severity (lognormal) μ: {inputs['sev_mu']:.2f}, σ: {inputs['sev_sigma']:.2f}",
            "",
            "BASELINE METRICS",
            f"• Expected Annual Loss (EAL): {fmt_money(baseline_metrics['Expected Annual Loss'])}",
            f"• Probability of breach-year: {baseline_metrics['Probability of Breach-Year']:.2%}",
            f"• VaR 95%: {fmt_money(baseline_metrics['VaR 95%'])}",
        ],
        size=10
    )

    if controlled_metrics:
        y -= 5
        y = write_block(
            1.5 * cm,
            y,
            [
                "",
                "WITH CONTROLS",
                f"• Expected Annual Loss (EAL): {fmt_money(controlled_metrics['Expected Annual Loss'])}",
                f"• Probability of breach-year: {controlled_metrics['Probability of Breach-Year']:.2%}",
                f"• VaR 95%: {fmt_money(controlled_metrics['VaR 95%'])}",
            ],
            size=10
        )

        if roi:
            y -= 5
            y = write_block(
                1.5 * cm,
                y,
                [
                    "",
                    "ROI / ROSI SUMMARY",
                    f"• Annual risk reduction: {fmt_money(roi['risk_reduction'])}",
                    f"• Annual control cost: {fmt_money(roi['annual_control_cost'])}",
                    f"• Net benefit: {fmt_money(roi['net_benefit'])}",
                    f"• ROSI: {roi['rosi']:.2%}",
                ],
                size=10
            )

    # Plain language summary
    y -= 10
    y = write_block(1.5 * cm, y, ["", "PLAIN-LANGUAGE SUMMARY"], font="Helvetica-Bold", size=11)
    y = write_block(1.7 * cm, y, [f"- {s}" for s in baseline_summary_lines], size=10)

    if controlled_summary_lines:
        y = write_block(1.7 * cm, y - 5, [f"- {s}" for s in controlled_summary_lines if s.strip()], size=10)

    # ---------------- Distribution plot page
    c.showPage()
    header("Annual Loss Distribution", "Histogram of simulated annual loss outcomes")

    if dist_png_bytes:
        try:
            img = ImageReader(io.BytesIO(dist_png_bytes))
            c.drawImage(img, 1.5 * cm, 4.0 * cm, width=w - 3.0 * cm, height=h - 7.5 * cm, preserveAspectRatio=True)
        except Exception:
            pass

    # ---------------- Controls Summary page
    if controlled_metrics and controls_params:
        c.showPage()
        header("Controls Summary", "What controls were assumed, and their modeled effect")

        lines = [
            "Controls configuration (as modeled):",
            f"• Threat reduction: {controls_params['lambda_reduction']}%",
            f"• Vulnerability reduction: {controls_params['vuln_reduction']}%",
            f"• Impact reduction: {controls_params['severity_reduction']}%",
            "",
            "Interpretation:",
            "• Threat reduction lowers how often attacks occur (frequency).",
            "• Vulnerability reduction lowers the chance an attack succeeds.",
            "• Impact reduction lowers loss size when an incident happens.",
        ]
        _ = write_block(1.5 * cm, h - 5.5 * cm, lines, size=10)

    # ---------------- Sensitivity page (optional)
    if tornado_png_bytes:
        c.showPage()
        header("Sensitivity Analysis", "Which inputs drive Expected Annual Loss the most?")

        try:
            img = ImageReader(io.BytesIO(tornado_png_bytes))
            c.drawImage(img, 1.5 * cm, 4.0 * cm, width=w - 3.0 * cm, height=h - 7.5 * cm, preserveAspectRatio=True)
        except Exception:
            pass

    c.save()
    return buf.getvalue()


# =========================================================
# Show logo on the app (optional)
# =========================================================
logo_path = find_logo_path()
if logo_path:
    st.image(logo_path, width=260)
else:
    st.info("Tip: Put your logo in `assets/aims_logo.png` (or `assets/logo.png`). I’ll auto-detect it.")


# =========================================================
# Sidebar: Inputs
# =========================================================
st.sidebar.header("Simulation Inputs")

trials = st.sidebar.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)
lam = st.sidebar.slider("Threat Frequency (events/year)", 0.0, 20.0, 5.0)
p_vuln = st.sidebar.slider("Vulnerability Probability", 0.0, 1.0, 0.3)

asset_value = st.sidebar.number_input("Asset Value ($)", value=1000000, step=10000)
exposure_factor = st.sidebar.slider("Exposure Factor (0–1)", 0.1, 1.0, 0.5)

sev_mu = st.sidebar.slider("Severity Mean (lognormal μ)", 0.0, 2.0, 0.5)
sev_sigma = st.sidebar.slider("Severity Std (lognormal σ)", 0.1, 2.0, 1.0)

# Sidebar: Controls
st.sidebar.markdown("---")
st.sidebar.header("Security Controls")

use_controls = st.sidebar.checkbox("Enable Security Controls", value=True)
lambda_reduction = st.sidebar.slider("Threat Reduction (%)", 0, 90, 30)
vuln_reduction = st.sidebar.slider("Vulnerability Reduction (%)", 0, 90, 40)
severity_reduction = st.sidebar.slider("Impact Reduction (%)", 0, 90, 25)

# Sidebar: ROI
st.sidebar.markdown("---")
st.sidebar.header("Investment (ROI)")
annual_control_cost = st.sidebar.number_input(
    "Annual control cost ($/year)",
    value=250000,
    step=10000,
    help="Estimated annual cost of the selected controls (licenses, staff time, training, etc.)"
)

# Sidebar: Sensitivity
st.sidebar.markdown("---")
st.sidebar.header("Sensitivity Analysis")
run_sensitivity = st.sidebar.checkbox("Run sensitivity analysis (slower)", value=False)
sens_pct = st.sidebar.slider("Sensitivity change (%)", 5, 50, 20)

# Sidebar: Stress Testing
st.sidebar.markdown("---")
st.sidebar.header("Stress Testing")
stress_mode = st.sidebar.selectbox(
    "Stress Scenario",
    ["Normal Year", "Elevated Threat Year", "Ransomware Wave", "Custom"]
)
custom_shock = 1.0
if stress_mode == "Custom":
    custom_shock = st.sidebar.slider("Custom Threat Multiplier", 1.0, 5.0, 2.0)


# =========================================================
# Run simulation
# =========================================================
run = st.button("Run Simulation", type="primary")

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
    st.info(f"Stress scenario: **{stress_mode}** → effective threat frequency λ = **{lam_effective:.2f} events/year**")

    # Baseline simulation
    baseline_losses = simulate_annual_losses(
        trials, lam_effective, p_vuln, asset_value, exposure_factor, sev_mu, sev_sigma
    )
    baseline_metrics = compute_risk_metrics(baseline_losses)

    # Controls simulation
    controlled_losses = None
    controlled_metrics = None

    if use_controls:
        lam_control = lam_effective * (1 - lambda_reduction / 100.0)  # stress-adjusted
        p_control = p_vuln * (1 - vuln_reduction / 100.0)
        exposure_control = exposure_factor * (1 - severity_reduction / 100.0)

        controlled_losses = simulate_annual_losses(
            trials, lam_control, p_control, asset_value, exposure_control, sev_mu, sev_sigma
        )
        controlled_metrics = compute_risk_metrics(controlled_losses)

    # =============================
    # Risk Metrics
    # =============================
    st.subheader("📊 Risk Metrics")

    rosi = None
    roi_pack = None

    if controlled_metrics is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Annual Loss (EAL)", fmt_money(baseline_metrics["Expected Annual Loss"]))
        c2.metric("Probability of Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
        c3.metric("VaR 95%", fmt_money(baseline_metrics["VaR 95%"]))
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("### Baseline")
            c1, c2, c3 = st.columns(3)
            c1.metric("EAL", fmt_money(baseline_metrics["Expected Annual Loss"]))
            c2.metric("Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
            c3.metric("VaR 95%", fmt_money(baseline_metrics["VaR 95%"]))

        with right:
            st.markdown("### With Controls")
            c4, c5, c6 = st.columns(3)
            eal_delta = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
            c4.metric("EAL", fmt_money(controlled_metrics["Expected Annual Loss"]), delta=f"-{fmt_money(eal_delta)}")
            c5.metric("Breach-Year", f"{controlled_metrics['Probability of Breach-Year']:.2%}")
            c6.metric("VaR 95%", fmt_money(controlled_metrics["VaR 95%"]))

        # ROI / ROSI
        st.subheader("💰 ROI / ROSI (Investment Value)")

        risk_reduction = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
        net_benefit = risk_reduction - annual_control_cost
        rosi = (net_benefit / annual_control_cost) if annual_control_cost > 0 else float("nan")

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Annual Risk Reduction", fmt_money(risk_reduction))
        cb.metric("Annual Control Cost", fmt_money(annual_control_cost))
        cc.metric("Net Benefit", fmt_money(net_benefit))
        cd.metric("ROSI", f"{rosi:.2%}" if not np.isnan(rosi) else "N/A")

        if net_benefit > 0:
            st.success("✅ Controls look cost-effective under these assumptions (net benefit > 0).")
        else:
            st.warning("⚠️ Controls may not be cost-effective under these assumptions (net benefit ≤ 0).")

        roi_pack = {
            "risk_reduction": safe_float(risk_reduction),
            "annual_control_cost": safe_float(annual_control_cost),
            "net_benefit": safe_float(net_benefit),
            "rosi": safe_float(rosi),
        }

    # =============================
    # Executive Summary (NO STARS)
    # =============================
    st.subheader("🧾 Executive Summary (Plain Language)")

    breach_base = baseline_metrics["Probability of Breach-Year"]
    eal_base = baseline_metrics["Expected Annual Loss"]
    var_base = baseline_metrics["VaR 95%"]

    baseline_summary_lines = [
        f"Baseline risk: About {breach_base:.0%} chance of at least one successful breach in a year.",
        f"Expected annual loss is about {fmt_money(eal_base)}.",
        f"In a bad year (worst 5%), losses may exceed {fmt_money(var_base)}.",
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
            f"With controls: breach-year likelihood drops to about {breach_ctrl:.0%}.",
            f"Expected annual loss drops to about {fmt_money(eal_ctrl)}.",
            f"Bad-year threshold (VaR 95%) drops to {fmt_money(var_ctrl)}.",
            f"Impact: breach probability decreases by {breach_drop:.0%}, EAL decreases by {fmt_money(eal_drop)}, VaR 95% decreases by {fmt_money(var_drop)}.",
        ]

        # Use st.markdown without **bold** to avoid any star artifacts
        st.markdown(
            "\n".join([
                f"- {baseline_summary_lines[0]}",
                f"- {baseline_summary_lines[1]}",
                f"- {baseline_summary_lines[2]}",
                "",
                f"- {controlled_summary_lines[0]}",
                f"- {controlled_summary_lines[1]}",
                f"- {controlled_summary_lines[2]}",
                f"- {controlled_summary_lines[3]}",
            ])
        )

        if roi_pack:
            st.markdown(
                "\n".join([
                    "",
                    f"- Investment view: if controls cost {fmt_money(annual_control_cost)} per year, estimated net benefit is {fmt_money(roi_pack['net_benefit'])} per year, with ROSI ≈ {roi_pack['rosi']:.0%}.",
                ])
            )
    else:
        st.markdown("\n".join([f"- {s}" for s in baseline_summary_lines]))

    st.caption("Scenario-based Monte Carlo model. Results depend on the assumptions you select.")

    # =============================
    # Distribution plot
    # =============================
    st.subheader("📈 Annual Loss Distribution")

    fig = make_distribution_plot(
        baseline_losses=baseline_losses,
        baseline_var95=baseline_metrics["VaR 95%"],
        controlled_losses=controlled_losses,
        controlled_var95=(controlled_metrics["VaR 95%"] if controlled_metrics else None),
    )
    st.pyplot(fig)

    # Save distribution fig to bytes (for PDF)
    dist_buf = io.BytesIO()
    fig.savefig(dist_buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    dist_png_bytes = dist_buf.getvalue()

    # =============================
    # Sensitivity Analysis
    # =============================
    tornado_png_bytes = None

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

        # Save tornado to bytes
        tor_buf = io.BytesIO()
        fig2.savefig(tor_buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        tornado_png_bytes = tor_buf.getvalue()

    # =============================
    # Store results for PDF export (session_state)
    # =============================
    st.session_state["last_run_payload"] = {
        "logo_path": logo_path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stress_mode": stress_mode,
        "shock": shock,
        "lam_effective": lam_effective,
        "inputs": {
            "trials": trials,
            "lam": lam,
            "p_vuln": p_vuln,
            "asset_value": float(asset_value),
            "exposure_factor": exposure_factor,
            "sev_mu": sev_mu,
            "sev_sigma": sev_sigma,
        },
        "baseline_metrics": baseline_metrics,
        "baseline_summary_lines": baseline_summary_lines,
        "controlled_metrics": controlled_metrics,
        "controlled_summary_lines": controlled_summary_lines,
        "controls_params": (None if not controlled_metrics else {
            "lambda_reduction": lambda_reduction,
            "vuln_reduction": vuln_reduction,
            "severity_reduction": severity_reduction,
        }),
        "roi": roi_pack,
        "dist_png_bytes": dist_png_bytes,
        "tornado_png_bytes": tornado_png_bytes,
    }

    st.divider()
    st.subheader("📄 Export Executive PDF (Professional)")

    st.caption("Tip: For best results, rename your logo to `assets/aims_logo.png` (optional — auto-detect works too).")

    if st.button("Generate PDF Report"):
        payload = st.session_state.get("last_run_payload")
        if not payload:
            st.warning("Run the simulation first, then generate the PDF.")
        else:
            pdf_bytes = try_build_pdf_report_reportlab(payload)
            if pdf_bytes is None:
                st.error(
                    "PDF library not available on your Streamlit environment.\n\n"
                    "Fix: add `reportlab` to `requirements.txt`, redeploy, then try again."
                )
            else:
                st.download_button(
                    "⬇️ Download Executive Report (PDF)",
                    data=pdf_bytes,
                    file_name="Cyber_Risk_Executive_Report.pdf",
                    mime="application/pdf",
                )

else:
    st.info("Set parameters on the left, then click **Run Simulation**.")
