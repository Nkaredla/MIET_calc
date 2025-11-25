import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 1. Relaxation rate T(q, h)
# -------------------------------

def fluctuation_coeff(q, h, eta, sigma, gamma, kappa=20 * 4.11e-21):
    """
    Relaxation rate T(q, h) [1/s] for each Fourier mode q,
    matching the structure of your FluctuationCoeff in MATLAB + LaTeX.
    """
    q = np.asarray(q)

    # base (no-wall) part
    f1 = (kappa * q**4 + sigma * q**2 + gamma) / (4.0 * eta * q)

    # wall correction factor
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

def fluctuation_coeff_two_walls(q, h_bot, h_top, eta, sigma, gamma, kappa=20*4.11e-21):
    q = np.asarray(q)
    Kq = kappa * q**4 + sigma * q**2 + gamma

    def f_single(x):
        sinhx = np.sinh(x)
        coshx = np.cosh(x)
        x2 = x**2
        num = 2.0 * (sinhx**2 - x2)
        den = sinhx**2 - x2 + sinhx*coshx + x
        out = num / den
        return np.where(np.isnan(out), np.inf, out)

    f_bot = f_single(h_bot * q)
    f_top = f_single(h_top * q)

    F = 0.5 * (f_bot + f_top)

    return (Kq / (4.0 * eta * q)) * F


# -------------------------------
# 2. Height correlation from q-sum
# -------------------------------

def height_corr_from_weights(t, q, Tq, Wq):
    """
    Compute normalized height correlation g(t) = C(t)/C(0) given:

      Tq : relaxation rates T(q) [1/s]
      Wq : weights W(q)

    C(t) ∝ sum_q W(q) e^{-T(q) t}  (Riemann sum).
    """
    t = np.asarray(t)
    Tq = np.asarray(Tq)
    Wq = np.asarray(Wq)

    exp_term = np.exp(-np.outer(t, Tq))   # shape (Nt, Nq)
    Ct = exp_term @ Wq                   # (Nt,)

    if Ct[0] != 0:
        g = Ct / Ct[0]
    else:
        g = Ct
    return g


# -------------------------------
# 3. Activity profile A(q)
# -------------------------------

def active_profile(q, q_center, width, alpha):
    """
    A(q) = alpha * exp(- (q - q_center)^2 / (2 width^2))

    Dimensionless "activity field" that multiplies kBT in the weights.
    alpha controls the amplitude of activity.
    """
    return alpha * np.exp(- (q - q_center)**2 / (2.0 * width**2))


# -------------------------------
# 4. Driver: show how g(t) changes with alpha
# -------------------------------

def run_active_weight_demo():
    # Effective plasma-membrane-like parameters
    eta = 1.2e-3      # Pa·s
    sigma = 2e-5      # J/m^2
    gamma = 1e7       # J/m^4
    kappa = 20 * 4.11e-21  # ~20 kBT
    h_avg = 40e-9     # 40 nm

    # q-range as in your analysis
    q_min = (gamma / kappa)**0.25
    q_max = 1.0 / h_avg
    Nq = 400
    q = np.linspace(q_min, q_max, Nq)

    # Relaxation rates T(q)
    Tq = fluctuation_coeff(q, h_avg, eta, sigma, gamma, kappa=kappa)
    Tq = np.maximum(Tq, 0.0)

    # Denominator K(q) = kappa q^4 + sigma q^2 + gamma
    Kq = kappa * q**4 + sigma * q**2 + gamma

    # Gaussian focus (beam waist)
    w_beam = 280e-9                       # detection beam waist [m]
    w_q = np.exp(-w_beam**2 * q**2 / 4.0) # Fourier-space detection profile

    # Choose an "actively driven" band of modes (adjust as you like)
    q_center = q_min * 2.5
    width = (q_max - q_min) * 0.05

    # Time axis for correlation (log-spaced to mimic your plots)
    t = np.logspace(-4, 1, 300)   # 10^-4 to 10^1 s

    # Different activity amplitudes
    alphas = [0.0, 1.0, 5.0, 20.0]

    plt.figure(figsize=(7, 6))

    for alpha in alphas:
        # A(q) from active theory: kBT -> kBT [1 + A(q)]
        A_q = active_profile(q, q_center, width, alpha)

        # Weights including detection and activity:
        # W(q) ∝ w(q)^2 [1 + A(q)] / K(q)
        Wq = (w_q**2) * (1.0 + A_q) / Kq

        # Compute normalized correlation
        g_t = height_corr_from_weights(t, q, Tq, Wq)

        label = f"alpha = {alpha:g}"
        plt.semilogx(t, g_t, label=label)

    plt.xlabel("lag time t [s]")
    plt.ylabel("normalized height corr. g(t)")
    plt.title("Effect of q-dependent active noise on hACF shape (with Gaussian focus)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_active_weight_demo()
