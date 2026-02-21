import os
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

from risk_model import simulate_annual_losses, compute_risk_metrics


# =============================
# Page config
# =============================
st.set_page_config(page_title="Cyber Risk Quant Dashboard", layout="wide")


# =============================
# Helpers
# =============================
def safe_show_logo(logo_path: str, width: int = 240):
    """Show logo in Streamlit without crashing if image/path has issues."""
    if not logo_path or not os.path.exists(logo_path):
        return
    try:
        with open(logo_path, "rb") as f:
            st.image(f.read(), width=width)
    except Exception:
        pass


def stress_multiplier(stress_mode: str, custom_shock: float) -> float:
    if stress_mode == "Normal Year":
        return 1.0
    if stress_mode == "Elevated Threat Year":
        return 1.5
    if stress_mode == "Ransomware Wave":
        return 2.5
    return float(custom_shock)


def compute_eal(trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_):
    losses_ = simulate_annual_losses(
        trials_, lam_, p_vuln_, asset_value_, exposure_factor_, sev_mu_, sev_sigma_
    )
    metrics_ = compute_risk_metrics(losses_)
    return float(metrics_["Expected Annual Loss"])


def _fmt_money(x, pos=None):
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.1f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"


def make_distribution_figure(baseline_losses, baseline_metrics, controlled_losses=None, controlled_metrics=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(baseline_losses, bins=50, alpha=0.65, label="Baseline")
    ax.axvline(baseline_metrics["VaR 95%"], linestyle="dashed", linewidth=2, label="Baseline VaR 95%")

    if controlled_losses is not None and controlled_metrics is not None:
        ax.hist(controlled_losses, bins=50, alpha=0.65, label="With Controls")
        ax.axvline(controlled_metrics["VaR 95%"], linestyle="dotted", linewidth=2, label="Controls VaR 95%")

    ax.set_title("Annual Loss Distribution", pad=10)
    ax.set_xlabel("Annual Loss")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_money))
    ax.ticklabel_format(style="plain", axis="x")  # no 1e8
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def make_tornado_figure(results, sens_pct):
    labels = [r[0] for r in results]
    lows = [r[1] for r in results]
    highs = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))

    ax.barh(y, highs, alpha=0.75, label=f"+{sens_pct}%")
    ax.barh(y, lows, alpha=0.75, label=f"-{sens_pct}%")
    ax.axvline(0, linestyle="dashed", linewidth=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in Expected Annual Loss (EAL)")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_money))
    ax.ticklabel_format(style="plain", axis="x")
    ax.set_title("Sensitivity Analysis (Tornado Plot)", pad=10)
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def _footer(fig):
    fig.text(
        0.5, 0.02,
        "Prepared by Grace Nzambali Kitonyi | African Institute for Mathematical Sciences (AIMS Rwanda)",
        ha="center", va="bottom", fontsize=8, color="gray"
    )


def export_pdf_report(
    out_path: str,
    logo_path: str,
    created_at: str,
    stress_mode: str,
    shock: float,
    lam_effective: float,
    inputs: dict,
    controls: dict,
    baseline_metrics: dict,
    controlled_metrics: dict | None,
    rosi: float | None,
    net_benefit: float | None,
    dist_fig,
    tornado_fig=None,
):
    """
    Professional multi-page PDF using Matplotlib PdfPages (no extra dependencies).
    """
    with PdfPages(out_path) as pdf:

        # =========================
        # PAGE 1 — Cover + Key Results
        # =========================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor("white")

        # Logo strip
        if logo_path and os.path.exists(logo_path):
            try:
                img = mpimg.imread(logo_path)
                ax_logo = fig.add_axes([0.08, 0.88, 0.84, 0.10])
                ax_logo.imshow(img)
                ax_logo.axis("off")
            except Exception:
                pass

        ax = fig.add_axes([0.08, 0.08, 0.84, 0.78])
        ax.axis("off")

        ax.text(0.0, 0.98, "CYBER RISK QUANTIFICATION", fontsize=22, fontweight="bold", va="top")
        ax.text(0.0, 0.94, "Executive Risk Assessment Report", fontsize=12, color="gray", va="top")
        ax.add_line(plt.Line2D([0, 1], [0.915, 0.915], linewidth=1, color="#D0D0D0"))

        ax.text(0.0, 0.89, f"Generated: {created_at}", fontsize=10, va="top")
        ax.text(0.0, 0.86, f"Stress scenario: {stress_mode}  (multiplier = {shock:.2f})", fontsize=10, va="top")
        ax.text(0.0, 0.83, f"Effective threat frequency λ: {lam_effective:.2f} events/year", fontsize=10, va="top")

        # Key metrics "cards"
        y0 = 0.74
        card_h = 0.12
        ax.add_patch(Rectangle((0.0, y0 - card_h), 1.0, card_h, facecolor="#F7F7F7", edgecolor="#E0E0E0"))
        ax.text(0.02, y0 - 0.02, "Baseline", fontsize=12, fontweight="bold", va="top")
        ax.text(0.02, y0 - 0.07, f"EAL: {_fmt_money(baseline_metrics['Expected Annual Loss'])}", fontsize=11, va="top")
        ax.text(0.35, y0 - 0.07, f"Breach-Year: {baseline_metrics['Probability of Breach-Year']:.2%}", fontsize=11, va="top")
        ax.text(0.70, y0 - 0.07, f"VaR 95%: {_fmt_money(baseline_metrics['VaR 95%'])}", fontsize=11, va="top")

        y1 = 0.58
        if controlled_metrics is not None:
            ax.add_patch(Rectangle((0.0, y1 - card_h), 1.0, card_h, facecolor="#F0F7FF", edgecolor="#D7E7FF"))
            ax.text(0.02, y1 - 0.02, "With Controls", fontsize=12, fontweight="bold", va="top")
            ax.text(0.02, y1 - 0.07, f"EAL: {_fmt_money(controlled_metrics['Expected Annual Loss'])}", fontsize=11, va="top")
            ax.text(0.35, y1 - 0.07, f"Breach-Year: {controlled_metrics['Probability of Breach-Year']:.2%}", fontsize=11, va="top")
            ax.text(0.70, y1 - 0.07, f"VaR 95%: {_fmt_money(controlled_metrics['VaR 95%'])}", fontsize=11, va="top")

            if rosi is not None and net_benefit is not None and controls.get("annual_control_cost", None) is not None:
                ax.text(0.0, 0.41, "Investment Summary", fontsize=13, fontweight="bold", va="top")
                ax.text(
                    0.0, 0.36,
                    "\n".join([
                        f"Annual control cost: {_fmt_money(controls['annual_control_cost'])}",
                        f"Estimated net benefit: {_fmt_money(net_benefit)} per year",
                        f"ROSI: {rosi:.0%}",
                    ]),
                    fontsize=11, va="top"
                )

                if net_benefit > 0:
                    ax.text(
                        0.0, 0.22,
                        "Strategic Interpretation: Controls materially reduce expected losses and are financially justified under current assumptions.",
                        fontsize=10, color="black", va="top"
                    )
                else:
                    ax.text(
                        0.0, 0.22,
                        "Strategic Interpretation: Controls reduce risk but may not be cost-effective under current assumptions.",
                        fontsize=10, color="black", va="top"
                    )

        ax.text(
            0.0, 0.10,
            "Note: Scenario-based Monte Carlo model. Results depend on selected assumptions.",
            fontsize=9, color="gray", va="top"
        )

        _footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================
        # PAGE 2 — Controls + Assumptions (clean layout)
        # =========================
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.86])
        ax.axis("off")

        ax.text(0.0, 0.98, "Controls Summary", fontsize=18, fontweight="bold", va="top")
        ax.add_line(plt.Line2D([0, 1], [0.95, 0.95], linewidth=1, color="#D0D0D0"))

        left_x = 0.0
        right_x = 0.52

        ax.text(left_x, 0.90, "Baseline Inputs", fontsize=13, fontweight="bold", va="top")
        baseline_lines = [
            f"Trials: {inputs['trials']:,}",
            f"Threat frequency λ (base): {inputs['lam']:.2f} / year",
            f"Vulnerability probability p: {inputs['p_vuln']:.2f}",
            f"Asset value: {_fmt_money(inputs['asset_value'])}",
            f"Exposure factor: {inputs['exposure_factor']:.2f}",
            f"Severity (lognormal μ, σ): ({inputs['sev_mu']:.2f}, {inputs['sev_sigma']:.2f})",
        ]
        ax.text(left_x, 0.86, "\n".join(baseline_lines), fontsize=11, va="top")

        ax.text(right_x, 0.90, "Controls Configuration", fontsize=13, fontweight="bold", va="top")
        control_lines = [
            f"Enabled: {controls['use_controls']}",
            f"Threat reduction: {controls['lambda_reduction']}%",
            f"Vulnerability reduction: {controls['vuln_reduction']}%",
            f"Impact reduction: {controls['severity_reduction']}%",
            f"Annual control cost: {_fmt_money(controls['annual_control_cost'])}",
        ]
        ax.text(right_x, 0.86, "\n".join(control_lines), fontsize=11, va="top")

        ax.text(0.0, 0.62, "How to interpret", fontsize=13, fontweight="bold", va="top")
        ax.text(
            0.0, 0.58,
            "\n".join([
                "• EAL estimates the expected annual loss under selected inputs.",
                "• Breach-Year is the probability of ≥1 successful breach within a year.",
                "• VaR 95% indicates a severe (worst 5%) loss threshold.",
                "• ROSI compares risk reduction vs. annual control cost (not a guarantee).",
            ]),
            fontsize=11, va="top"
        )

        _footer(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =========================
        # PAGE 3 — Distribution Plot
        # =========================
        _footer(dist_fig)
        pdf.savefig(dist_fig, bbox_inches="tight")

        # =========================
        # PAGE 4 — Tornado Plot (optional)
        # =========================
        if tornado_fig is not None:
            _footer(tornado_fig)
            pdf.savefig(tornado_fig, bbox_inches="tight")


# =============================
# Header (logo + title)
# =============================
logo_path = os.path.join("assets", "aims_logo.png")

h1, h2 = st.columns([1, 3], vertical_alignment="center")
with h1:
    safe_show_logo(logo_path, width=160)
with h2:
    st.title("🛡️ Cyber Risk Quantification Dashboard")
    st.caption("Monte Carlo cyber risk model • Controls ROI/ROSI • Stress testing • Sensitivity • PDF export")


# =============================
# Sidebar: Inputs
# =============================
st.sidebar.header("Simulation Inputs")

trials = st.sidebar.slider("Monte Carlo Trials", 1000, 20000, 10000, step=1000)
lam = st.sidebar.slider("Threat Frequency (events/year)", 0.0, 20.0, 5.0)
p_vuln = st.sidebar.slider("Vulnerability Probability", 0.0, 1.0, 0.30)

asset_value = st.sidebar.number_input("Asset Value ($)", value=1_000_000, step=10_000)
exposure_factor = st.sidebar.slider("Exposure Factor (0–1)", 0.10, 1.00, 0.50)

sev_mu = st.sidebar.slider("Severity Mean (lognormal μ)", 0.0, 2.0, 0.50)
sev_sigma = st.sidebar.slider("Severity Std (lognormal σ)", 0.10, 2.0, 1.00)


# =============================
# Sidebar: Controls
# =============================
st.sidebar.markdown("---")
st.sidebar.header("Security Controls")

use_controls = st.sidebar.checkbox("Enable Security Controls", value=True)
lambda_reduction = st.sidebar.slider("Threat Reduction (%)", 0, 90, 30)
vuln_reduction = st.sidebar.slider("Vulnerability Reduction (%)", 0, 90, 40)
severity_reduction = st.sidebar.slider("Impact Reduction (%)", 0, 90, 25)


# =============================
# Sidebar: ROI
# =============================
st.sidebar.markdown("---")
st.sidebar.header("Investment (ROI)")

annual_control_cost = st.sidebar.number_input(
    "Annual control cost ($/year)",
    value=250_000,
    step=10_000,
    help="Estimated annual cost of the selected controls (licenses, staff time, training, etc.)"
)


# =============================
# Sidebar: Sensitivity
# =============================
st.sidebar.markdown("---")
st.sidebar.header("Sensitivity Analysis")
run_sensitivity = st.sidebar.checkbox("Run sensitivity analysis (slower)", value=False)
sens_pct = st.sidebar.slider("Sensitivity change (%)", 5, 50, 20)


# =============================
# Sidebar: Stress Testing
# =============================
st.sidebar.markdown("---")
st.sidebar.header("Stress Testing")

stress_mode = st.sidebar.selectbox(
    "Stress Scenario",
    ["Normal Year", "Elevated Threat Year", "Ransomware Wave", "Custom"]
)

custom_shock = 2.0
if stress_mode == "Custom":
    custom_shock = st.sidebar.slider("Custom Threat Multiplier", 1.0, 5.0, 2.0)


# =============================
# Session state
# =============================
if "last" not in st.session_state:
    st.session_state.last = None


# =============================
# Run simulation
# =============================
run = st.sidebar.button("Run Simulation")
if run:
    shock = stress_multiplier(stress_mode, custom_shock)
    lam_effective = lam * shock
    st.info(f"Stress scenario selected: **{stress_mode}** → λ becomes **{lam_effective:.2f} events/year**")

    baseline_losses = simulate_annual_losses(
        trials, lam_effective, p_vuln, asset_value, exposure_factor, sev_mu, sev_sigma
    )
    baseline_metrics = compute_risk_metrics(baseline_losses)

    controlled_losses = None
    controlled_metrics = None

    if use_controls:
        lam_control = lam_effective * (1 - lambda_reduction / 100.0)
        p_control = p_vuln * (1 - vuln_reduction / 100.0)
        exposure_control = exposure_factor * (1 - severity_reduction / 100.0)

        controlled_losses = simulate_annual_losses(
            trials, lam_control, p_control, asset_value, exposure_control, sev_mu, sev_sigma
        )
        controlled_metrics = compute_risk_metrics(controlled_losses)

    rosi = None
    net_benefit = None
    risk_reduction = None

    if controlled_metrics is not None:
        risk_reduction = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
        net_benefit = risk_reduction - annual_control_cost
        rosi = (net_benefit / annual_control_cost) if annual_control_cost > 0 else np.nan

    tornado_fig = None
    if run_sensitivity:
        base_eal = float(baseline_metrics["Expected Annual Loss"])
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
        tornado_fig = make_tornado_figure(results, sens_pct)

    dist_fig = make_distribution_figure(baseline_losses, baseline_metrics, controlled_losses, controlled_metrics)

    st.session_state.last = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stress_mode": stress_mode,
        "shock": shock,
        "lam_effective": lam_effective,
        "inputs": {
            "trials": trials, "lam": lam, "p_vuln": p_vuln,
            "asset_value": float(asset_value), "exposure_factor": exposure_factor,
            "sev_mu": sev_mu, "sev_sigma": sev_sigma,
        },
        "controls": {
            "use_controls": use_controls,
            "lambda_reduction": int(lambda_reduction),
            "vuln_reduction": int(vuln_reduction),
            "severity_reduction": int(severity_reduction),
            "annual_control_cost": float(annual_control_cost),
        },
        "baseline_metrics": baseline_metrics,
        "controlled_metrics": controlled_metrics,
        "rosi": rosi,
        "net_benefit": net_benefit,
        "risk_reduction": risk_reduction,
        "fig_dist": dist_fig,
        "fig_tornado": tornado_fig,
    }


# =============================
# Render results
# =============================
last = st.session_state.last

if last is None:
    st.info("Set parameters on the left, then click **Run Simulation**.")
else:
    baseline_metrics = last["baseline_metrics"]
    controlled_metrics = last["controlled_metrics"]

    st.markdown("## 📊 Risk Metrics")

    if controlled_metrics is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Annual Loss (EAL)", f"{_fmt_money(baseline_metrics['Expected Annual Loss'])}")
        c2.metric("Probability of Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
        c3.metric("VaR 95%", f"{_fmt_money(baseline_metrics['VaR 95%'])}")
    else:
        left, right = st.columns(2)

        with left:
            st.markdown("### Baseline")
            c1, c2, c3 = st.columns(3)
            c1.metric("EAL", f"{_fmt_money(baseline_metrics['Expected Annual Loss'])}")
            c2.metric("Breach-Year", f"{baseline_metrics['Probability of Breach-Year']:.2%}")
            c3.metric("VaR 95%", f"{_fmt_money(baseline_metrics['VaR 95%'])}")

        with right:
            st.markdown("### With Controls")
            c4, c5, c6 = st.columns(3)
            eal_delta = baseline_metrics["Expected Annual Loss"] - controlled_metrics["Expected Annual Loss"]
            c4.metric("EAL", f"{_fmt_money(controlled_metrics['Expected Annual Loss'])}", delta=f"-{_fmt_money(eal_delta)}")
            c5.metric("Breach-Year", f"{controlled_metrics['Probability of Breach-Year']:.2%}")
            c6.metric("VaR 95%", f"{_fmt_money(controlled_metrics['VaR 95%'])}")

        st.subheader("💰 ROI / ROSI (Investment Value)")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Annual Risk Reduction", f"{_fmt_money(last['risk_reduction'])}")
        cB.metric("Annual Control Cost", f"{_fmt_money(last['controls']['annual_control_cost'])}")
        cC.metric("Net Benefit", f"{_fmt_money(last['net_benefit'])}")
        rosi = last["rosi"]
        cD.metric("ROSI", f"{rosi:.2%}" if (rosi is not None and not np.isnan(rosi)) else "N/A")

        if last["net_benefit"] is not None and last["net_benefit"] > 0:
            st.success("Controls look cost-effective under these assumptions (net benefit > 0).")
        else:
            st.warning("Controls may not be cost-effective under these assumptions (net benefit ≤ 0).")

    st.subheader("🧾 Executive Summary (Plain Language)")

    breach_base = baseline_metrics["Probability of Breach-Year"]
    eal_base = baseline_metrics["Expected Annual Loss"]
    var_base = baseline_metrics["VaR 95%"]

    if controlled_metrics is None:
        st.markdown(
            f"- Baseline breach-year probability: **{breach_base:.0%}**\n"
            f"- Expected annual loss (EAL): **{_fmt_money(eal_base)}**\n"
            f"- Severe year threshold (VaR 95%): **{_fmt_money(var_base)}**"
        )
    else:
        breach_ctrl = controlled_metrics["Probability of Breach-Year"]
        eal_ctrl = controlled_metrics["Expected Annual Loss"]
        var_ctrl = controlled_metrics["VaR 95%"]

        breach_drop = breach_base - breach_ctrl
        eal_drop = eal_base - eal_ctrl
        var_drop = var_base - var_ctrl

        st.markdown(
            f"**Baseline**\n"
            f"- Breach-year probability: **{breach_base:.0%}**\n"
            f"- EAL: **{_fmt_money(eal_base)}**\n"
            f"- VaR 95%: **{_fmt_money(var_base)}**\n\n"
            f"**With controls**\n"
            f"- Breach-year probability: **{breach_ctrl:.0%}**\n"
            f"- EAL: **{_fmt_money(eal_ctrl)}**\n"
            f"- VaR 95%: **{_fmt_money(var_ctrl)}**\n\n"
            f"**Impact**\n"
            f"- Breach probability change: **-{breach_drop:.0%}**\n"
            f"- EAL reduction: **{_fmt_money(eal_drop)}**\n"
            f"- VaR 95% reduction: **{_fmt_money(var_drop)}**\n\n"
            f"**Investment view**\n"
            f"- Annual control cost: **{_fmt_money(last['controls']['annual_control_cost'])}**\n"
            f"- Net benefit: **{_fmt_money(last['net_benefit'])}**\n"
            f"- ROSI: **{last['rosi']:.0%}**"
        )

    st.caption("Scenario-based Monte Carlo model. Results depend on the assumptions you select.")

    st.subheader("📈 Annual Loss Distribution")
    st.pyplot(last["fig_dist"])

    if last["fig_tornado"] is not None:
        st.subheader("🧪 Sensitivity Analysis (Tornado Plot)")
        st.pyplot(last["fig_tornado"])

    st.subheader("📄 Export Executive PDF (Professional)")
    st.caption("Logo path used: assets/aims_logo.png")

    if st.button("Generate PDF Report"):
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.close()

        export_pdf_report(
            out_path=tmp_pdf.name,
            logo_path=logo_path,
            created_at=last["created_at"],
            stress_mode=last["stress_mode"],
            shock=last["shock"],
            lam_effective=last["lam_effective"],
            inputs=last["inputs"],
            controls=last["controls"],
            baseline_metrics=last["baseline_metrics"],
            controlled_metrics=last["controlled_metrics"],
            rosi=last["rosi"],
            net_benefit=last["net_benefit"],
            dist_fig=last["fig_dist"],
            tornado_fig=last["fig_tornado"],
        )

        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                "⬇️ Download Executive Report (PDF)",
                data=f.read(),
                file_name="Cyber_Risk_Executive_Report.pdf",
                mime="application/pdf",
            )

        try:
            os.remove(tmp_pdf.name)
        except Exception:
            pass
