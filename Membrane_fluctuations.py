# -*- coding: utf-8 -*-
"""
Active membrane height fluctuations: simulation + multi-parameter fit.

Fits:
  - correlation amplitude A0
  - activity strength alpha
  - q_center of the active band
  - width of the active band

on the UNNORMALIZED correlation C(t).
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================================
# 1. Relaxation rate T(q, h)
# =========================================

def fluctuation_coeff(q, h, eta, sigma, gamma, kappa=20 * 4.11e-21):
    """
    Relaxation rate T(q, h) [1/s] for each Fourier mode q,
    in the same spirit as your MATLAB FluctuationCoeff.
    """
    q = np.asarray(q)

    # Base (no-wall) contribution
    f1 = (kappa * q**4 + sigma * q**2 + gamma) / (4.0 * eta * q)

    # Wall correction factor
    fac = h * q
    sinh_fac = np.sinh(fac)
    cosh_fac = np.cosh(fac)
    fac2 = fac**2

    numerator = 2.0 * (sinh_fac**2 - fac2)
    denominator = (
        sinh_fac**2
        - fac2
        + sinh_fac * cosh_fac
        + fac
    )

    f2 = numerator / denominator
    f2 = np.where(np.isnan(f2), np.inf, f2)

    return f1 * f2


# =========================================
# 2. Active profile A(q)
# =========================================

def active_profile(q, q_center, width, alpha):
    """
    Dimensionless activity profile:

        A(q) = alpha * exp(- (q - q_center)**2 / (2 width**2))

    alpha controls the amplitude of activity;
    q_center and width set which band of modes is most active.
    """
    return alpha * np.exp(- (q - q_center)**2 / (2.0 * width**2))


# =========================================
# 3. Exact OU simulation with q-dependent variance
# =========================================

def simulate_height_timeseries(
    T_total,
    dt_guess,
    q,
    h_avg,
    eta,
    sigma,
    gamma,
    alpha,
    q_center,
    width,
    w=280e-9,
    T_kelvin=294.0,
    kappa=20 * 4.11e-21,
    noise_rel=0.03,
    seed=None,
):
    """
    Simulate height h(t) at x=0 for an active membrane with:

      <|h_q|^2> ∝ [1 + A(q)] / (kappa q^4 + sigma q^2 + gamma),

    where A(q) = active_profile(q, q_center, width, alpha).

    Dynamics of each mode is OU with rate T(q) = fluctuation_coeff(q, ...),
    and stationary variance var_eq(q) as above.

    We use the exact OU time step:

      h_q(t+dt) = h_q(t) * exp(-T dt)
                  + sqrt(var_eq * (1 - exp(-2 T dt))) * N(0,1).

    This guarantees the correct variance and avoids numerical instability.
    """
    if seed is not None:
        np.random.seed(seed)

    kB = 1.380649e-23
    kBT = kB * T_kelvin

    q = np.asarray(q)
    Nq = len(q)

    # Relaxation rates for each mode
    Tq = fluctuation_coeff(q, h_avg, eta, sigma, gamma, kappa=kappa)
    Tq = np.maximum(Tq, 0.0)

    # Choose dt safely: ensure T_max * dt <= 0.05
    Tmax = np.max(Tq)
    if Tmax > 0:
        dt_safety = 0.05 / Tmax
        dt = min(dt_guess, dt_safety)
    else:
        dt = dt_guess

    # override if you want fixed dt
    dt = 1e-4
    N = int(round(T_total / dt))

    # Mode stiffness K(q) = kappa q^4 + sigma q^2 + gamma
    Kq = kappa * q**4 + sigma * q**2 + gamma

    # Activity profile A(q) (dimensionless)
    A_q = active_profile(q, q_center, width, alpha)

    # Stationary variance per mode: var_q ∝ [kBT (1 + A(q))] / K(q)
    var_eq = kBT * (1.0 + A_q) / Kq

    # Exact OU coefficients
    expfac = np.exp(-Tq * dt)
    one_minus_exp2 = -np.expm1(-2.0 * Tq * dt)    # = 1 - exp(-2 Tq dt)
    noise_std = np.sqrt(var_eq * one_minus_exp2)
    noise_std = np.where(np.isfinite(noise_std), noise_std, 0.0)

    # Initialize modes
    h_q = np.zeros(Nq)

    # Gaussian detection weighting
    gaussian_weight = np.exp(-w**2 * q**2 / 4.0)

    # Time series for h(t)
    h = np.zeros(N)

    for i in range(N):
        # Exact OU update for each mode
        h_q = h_q * expfac + noise_std * np.random.randn(Nq)

        # Height at x=0
        h[i] = np.dot(gaussian_weight, h_q) + h_avg

    # Add measurement noise (white)
    std_h = np.std(h)
    h_meas = h + noise_rel * std_h * np.random.randn(N)

    time = np.arange(N) * dt
    return time, h_meas, Tq, var_eq, dt


# =========================================
# 4. Autocorrelation via FFT (UNNORMALIZED)
# =========================================

def autocorr_fft_raw(x, dt):
    """
    Unnormalized autocorrelation of x(t) using FFT.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1)))

    X = np.fft.fft(x, nfft)
    S = X * np.conjugate(X)
    corr = np.fft.ifft(S).real
    corr = corr[:n]

    t = np.arange(n) * dt
    return t, corr


# =========================================
# 5. Analytic correlation model (UNNORMALIZED)
# =========================================

def corr_model_unscaled(
    t,
    q,
    h_avg,
    eta,
    sigma,
    gamma,
    alpha,
    q_center,
    width,
    w=280e-9,
    kappa=20 * 4.11e-21,
):
    """
    Unnormalized analytic correlation C(t) with SAME weighting as simulation:

      h(t) = sum_q w(q) h_q(t),
      C(t) ∝ sum_q w(q)^2 * [1 + A(q)] / K(q) * exp(-T(q) t),

    where w(q) = exp(-w^2 q^2 / 4),
          K(q) = kappa q^4 + sigma q^2 + gamma,
          A(q) = active_profile(q, q_center, width, alpha),
          T(q) = fluctuation_coeff(q, h_avg, eta, sigma, gamma).
    """
    t = np.asarray(t)
    q = np.asarray(q)

    # relaxation rates
    Tq = fluctuation_coeff(q, h_avg, eta, sigma, gamma, kappa=kappa)
    Tq = np.maximum(Tq, 0.0)

    # stiffness
    Kq = kappa * q**4 + sigma * q**2 + gamma

    # activity profile
    A_q = active_profile(q, q_center, width, alpha)

    # Gaussian detection weighting
    w_q = np.exp(-w**2 * q**2 / 4.0)

    # effective weights (up to a global factor)
    Wq = (w_q**2) * (1.0 + A_q) / Kq

    exp_term = np.exp(-np.outer(t, Tq))    # shape (Nt, Nq)
    Ct = exp_term @ Wq                     # (Nt,)
    return Ct


def corr_model_normalized(*args, **kwargs):
    """
    Convenience wrapper: return normalized g(t) = C(t)/C(0).
    """
    Ct = corr_model_unscaled(*args, **kwargs)
    return Ct / Ct[0] if Ct[0] != 0 else Ct


# =========================================
# 6. Fit A0, alpha, q_center, width
# =========================================

def fit_active_params(
    t_data,
    C_data,
    q,
    h_avg,
    eta,
    sigma,
    gamma,
    alpha_grid,
    q_center_rel_grid,
    width_rel_grid,
    w=280e-9,
    kappa=20 * 4.11e-21,
):
    """
    Fit the unnormalized correlation by grid search over alpha, q_center, width,
    and analytic best amplitude A0.

    Parameterization:
      q_center = q_min + q_center_rel*(q_max - q_min)
      width    = width_rel*(q_max - q_min)
    """
    t_data = np.asarray(t_data)
    C_data = np.asarray(C_data)

    q_min = q[0]
    q_max = q[-1]

    best_err = np.inf
    best_params = None
    best_model = None

    for alpha in alpha_grid:
        for qc_rel in q_center_rel_grid:
            q_center = q_min + qc_rel * (q_max - q_min)
            for w_rel in width_rel_grid:
                width = w_rel * (q_max - q_min)

                Ct = corr_model_unscaled(
                    t_data,
                    q,
                    h_avg,
                    eta,
                    sigma,
                    gamma,
                    alpha,
                    q_center,
                    width,
                    w=w,
                    kappa=kappa,
                )

                # best A0 in least-squares sense
                num = np.dot(Ct, C_data)
                den = np.dot(Ct, Ct) if np.dot(Ct, Ct) > 0 else 1e-30
                A0 = max(num / den, 0.0)

                model = A0 * Ct
                err = np.sum((C_data - model) ** 2)

                if err < best_err:
                    best_err = err
                    best_params = (A0, alpha, q_center, width)
                    best_model = model

    return best_params, best_model, best_err


# =========================================
# 7. Demo: simulate & fit
# =========================================

def run_demo():
    # --- physical parameters ---
    eta = 1.2e-3
    sigma = 2e-5
    gamma = 1e7
    kappa = 20 * 4.11e-21
    h_avg = 40e-9
    w_beam = 280e-9

    q_min = (gamma / kappa)**0.25
    q_max = 1.0 / h_avg
    Nq = 400
    q = np.linspace(q_min, q_max, Nq)

    # true active parameters
    q_center_true = q_min * 2.5
    width_true = 0.05 * (q_max - q_min)
    alpha_true = 20.0

    T_total = 10.0
    dt_guess = 1e-3
    noise_rel = 0.05

    # simulate
    time, h_meas, Tq, var_eq, dt_used = simulate_height_timeseries(
        T_total,
        dt_guess,
        q,
        h_avg,
        eta,
        sigma,
        gamma,
        alpha_true,
        q_center_true,
        width_true,
        w=w_beam,
        T_kelvin=294.0,
        kappa=kappa,
        noise_rel=noise_rel,
        seed=1,
    )

    # unnormalized correlation
    t_corr, C_corr = autocorr_fft_raw(h_meas, dt_used)

    # fit window
    mask = (t_corr > 1e-3) & (t_corr < 2.0)
    t_fit = t_corr[mask]
    C_fit = C_corr[mask]

    # grid ranges (tune as needed)
    alpha_grid = np.linspace(0.0, 40.0, 41)
    q_center_rel_grid = np.linspace(0.0, 0.5, 11)   # first half of q-range
    width_rel_grid = np.linspace(0.02, 0.2, 9)      # 2%..20% of q-range

    best_params, best_model, best_err = fit_active_params(
        t_fit,
        C_fit,
        q,
        h_avg,
        eta,
        sigma,
        gamma,
        alpha_grid,
        q_center_rel_grid,
        width_rel_grid,
        w=w_beam,
        kappa=kappa,
    )

    A0_fit, alpha_fit, q_center_fit, width_fit = best_params

    print(f"True alpha       = {alpha_true:.2f}")
    print(f"Fitted alpha     = {alpha_fit:.2f}")
    print(f"True q_center    = {q_center_true:.3e}")
    print(f"Fitted q_center  = {q_center_fit:.3e}")
    print(f"True width       = {width_true:.3e}")
    print(f"Fitted width     = {width_fit:.3e}")
    print(f"Fitted amplitude = {A0_fit:.3e}")
    print(f"SSE              = {best_err:.3e}")

    # purely passive model for comparison
    Ct_passive = corr_model_unscaled(
        t_fit,
        q,
        h_avg,
        eta,
        sigma,
        gamma,
        alpha=0.0,
        q_center=q_center_true,
        width=width_true,
        w=w_beam,
        kappa=kappa,
    )
    num_p = np.dot(Ct_passive, C_fit)
    den_p = np.dot(Ct_passive, Ct_passive) if np.dot(Ct_passive, Ct_passive) > 0 else 1e-30
    A0_passive = max(num_p / den_p, 0.0)
    model_passive = A0_passive * Ct_passive

    # normalized curves for visual comparison
    g_data = C_fit / C_fit[0]
    g_fit_active = best_model / best_model[0]
    g_fit_passive = model_passive / model_passive[0]

    # ---------- plotting ----------
    fig, axs = plt.subplots(3, 1, figsize=(7, 9))

    # height trace
    axs[0].plot(time, h_meas * 1e9)
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("height [nm]")
    axs[0].set_title("Simulated height at x=0 (active membrane)")

    # unnormalized correlation and fits
    axs[1].loglog(t_fit, C_fit, "k.", label="simulation")
    axs[1].loglog(t_fit, model_passive, "C0--", label="passive fit (alpha=0)")
    axs[1].loglog(t_fit, best_model, "C3-", label=f"active fit (alpha={alpha_fit:.2f})")
    axs[1].set_xlabel("lag time t [s]")
    axs[1].set_ylabel("C(t) (arb. units)")
    axs[1].legend()
    axs[1].set_title("Unnormalized correlation and fits")

    # normalized version (shape only)
    axs[2].semilogx(t_fit, g_data, "k.", label="simulation (norm)")
    axs[2].semilogx(t_fit, g_fit_passive, "C0--", label="passive model (norm)")
    axs[2].semilogx(t_fit, g_fit_active, "C3-", label="active model (norm)")
    axs[2].set_xlabel("lag time t [s]")
    axs[2].set_ylabel("normalized g(t)")
    axs[2].legend()
    axs[2].set_title("Normalized shapes")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
