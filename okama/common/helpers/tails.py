import numpy as np
from scipy.stats import norm, lognorm, t


# Theoretical
def var_normal(alpha: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """
    Value-at-Risk for X ~ Normal(mu, sigma^2) at level alpha.

    Definition:
        VaR_alpha = F^{-1}(alpha), i.e., the (upper) alpha-quantile.

    Parameters:
        alpha (float): Tail probability level, must be in (0, 1).
        loc (float): Mean of the normal distribution.
        scale (float): Standard deviation (>0) of the normal distribution.

    Returns:
        float: VaR at level alpha.

    Notes:
        For loss modeling with returns R, one often sets X = -R and applies this formula.
    """
    z = norm.ppf(alpha)
    return loc + scale * z

def cvar_normal(alpha: float, loc: float, scale: float) -> float:
    """
    Compute left-tail CVaR (Conditional VaR, Expected Shortfall) at level alpha
    for a Normal(mu, sigma^2) distribution.

    Closed-form:
      Let z = Phi^{-1}(alpha), phi = standard normal pdf, Phi = standard normal cdf.
      VaR_alpha = mu + sigma * z
      CVaR_alpha = E[X | X <= VaR_alpha] = mu - sigma * phi(z) / alpha

    Args:
        alpha: Left-tail probability (e.g., 0.05).
        loc: Mean of the normal distribution.
        scale: Standard deviation (> 0).

    Returns:
        Left-tail CVaR at level alpha.
    """
    if scale <= 0:
        return np.nan
    z = norm.ppf(alpha)
    phi = norm.pdf(z)
    return loc - scale * (phi / alpha)

def var_t(alpha: float, v: float, loc: float = 0.0, scale: float = 1.0) -> float:
    q = t.ppf(alpha, v)
    return loc + scale * q

def cvar_t(alpha: float, v: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """
    Compute left-tail CVaR (Conditional VaR, Expected Shortfall) at level alpha
    for a Student's t distribution with degrees of freedom df, location loc, and scale scale.

    Parameterization (SciPy):
      If X ~ t(df, loc, scale), then X = loc + scale * T, where T ~ t(df) (standardized).
      VaR_alpha = loc + scale * q, where q = t_ppf(alpha; df).
      Left-tail CVaR (alpha) = E[X | X <= VaR_alpha]
                             = loc + scale * E[T | T <= q].

    Closed-form for standard t (df > 1):
      E[T | T <= q] = - ((df + q^2) / ((df - 1) * alpha)) * f(q),
      where q = t_ppf(alpha; df), f(q) = t_pdf(q; df).

    Args:
        alpha: Left-tail probability (e.g., 0.05).
        v: Degrees of freedom (> 1 required for finite CVaR).
        loc: Location parameter.
        scale: Scale parameter (> 0).

    Returns:
        Left-tail CVaR at level alpha. Returns NaN if df <= 1 or on numerical issues.
    """
    if v <= 1 or scale <= 0:
        return np.nan
    q = t.ppf(alpha, v)
    f = t.pdf(q, v)
    es_std = - ((v + q ** 2) / ((v - 1) * alpha)) * f
    return loc + scale * es_std

def var_lognorm(alpha: float, shape: float, loc: float = 0.0, scale: float = 1.0) -> float:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if shape <= 0:
        raise ValueError("shape must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return lognorm.ppf(alpha, shape, loc=loc, scale=scale)


def cvar_lognorm(alpha: float, shape: float, loc: float, scale: float) -> float:
    """
    Compute left-tail CVaR (Conditional VaR, Expected Shortfall) at level alpha
    for a lognormal RV in SciPy's parameterization: lognorm(s, loc, scale).

    Parameterization:
      X ~ lognorm(s, loc, scale) means: X = loc + Y, where ln(Y) ~ N(mu, sigma^2),
      sigma = s and mu = ln(scale). Support is X > loc.

    CVaR at level alpha is E[X | X <= q], where q = VaR_X(alpha).
    Let y_cap = q - loc (so y_cap > 0). Then:
      E[Y | Y <= y_cap] = exp(mu + 0.5*sigma^2)
                          * Phi((ln(y_cap) - mu - sigma^2)/sigma)
                          / Phi((ln(y_cap) - mu)/sigma),
      where Phi is the standard normal CDF.

    Returns loc + E[Y | Y <= y_cap]. If the conditioning set is degenerate (e.g., y_cap <= 0),
    returns NaN.

    Args:
        alpha: Left-tail probability (e.g., 0.05).
        shape: Shape (sigma > 0).
        loc: Location (shift). For returns modeled as r = G - 1, use loc = -1.
        scale: Scale (= exp(mu)).

    Returns:
        Theoretical left-tail CVaR at level alpha.
    """
    # VaR (quantile) for X
    q = lognorm.ppf(alpha, shape, loc=loc, scale=scale)
    y_cap = q - loc
    if not np.isfinite(y_cap) or y_cap <= 0:
        return np.nan

    mu = np.log(scale)
    sigma = shape

    num = norm.cdf((np.log(y_cap) - mu - sigma**2) / sigma)
    den = norm.cdf((np.log(y_cap) - mu) / sigma)
    if den <= 0:
        return np.nan

    e_y_cond = np.exp(mu + 0.5 * sigma**2) * (num / den)
    return loc + e_y_cond


def var_theoretical(distr: str, alpha: float, args: tuple) -> float:
    match distr:
        case "norm":
            return var_normal(alpha, *args)
        case "lognorm":
            return var_lognorm(alpha, *args)
        case "t":
            return var_t(alpha, *args)
        case _:
            raise ValueError("Unknown distribution: " + distr)

def cvar_theoretical(distr: str, alpha: float, args: tuple) -> float:
    match distr:
        case "norm":
            return cvar_normal(alpha, *args)
        case "lognorm":
            return cvar_lognorm(alpha, *args)
        case "t":
            return cvar_t(alpha, *args)
        case _:
            raise ValueError("Unknown distribution: " + distr)




# Empiric
def cvar_of_sample(arr: np.ndarray, alpha: float):
    """
    Compute the empirical left-tail CVaR (Conditional VaR, Expected Shortfall) at level alpha
    for a 1D sample.

    Definition:
        CVaR_alpha = mean{x_i : x_i <= q_alpha}, where q_alpha is the empirical alpha-quantile.

    Args:
        arr: 1D array-like of observations (will be converted to float and flattened).
        alpha: Left-tail probability level in (0, 1), e.g., 0.05.

    Returns:
        Empirical CVaR as a float. Returns NaN if the input array is empty.

    Notes:
        - Uses the “historical” estimator based on the mean of observations in the left tail.
        - If due to numerical issues no element is <= the computed quantile, the function
          falls back to returning the minimum observation (ensures at least one point).
    """
    q = np.quantile(arr, alpha)
    return arr[arr <= q].mean()


