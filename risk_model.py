import numpy as np

def simulate_annual_losses(
    trials,
    lam,              # Threat frequency (events per year)
    p_vuln,           # Probability breach occurs per event
    asset_value,      # Asset value in dollars
    exposure_factor,  # % of asset lost per breach
    sev_mu,           # Lognormal mean
    sev_sigma         # Lognormal std
):
    rng = np.random.default_rng()

    annual_losses = np.zeros(trials)

    for t in range(trials):

        # Step 1: number of attack attempts
        num_events = rng.poisson(lam)

        total_loss = 0

        for _ in range(num_events):

            # Step 2: does breach succeed?
            breach = rng.binomial(1, p_vuln)

            if breach == 1:

                # Step 3: loss severity
                severity = rng.lognormal(sev_mu, sev_sigma)

                loss = asset_value * exposure_factor * severity
                total_loss += loss

        annual_losses[t] = total_loss

    return annual_losses


def compute_risk_metrics(losses):

    mean_loss = np.mean(losses)
    breach_prob = np.mean(losses > 0)

    var_95 = np.quantile(losses, 0.95)

    tail_losses = losses[losses >= var_95]
    cvar_95 = np.mean(tail_losses) if len(tail_losses) > 0 else var_95

    return {
        "Expected Annual Loss": mean_loss,
        "Probability of Breach-Year": breach_prob,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95
    }
