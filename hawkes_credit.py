"""Multivariate Hawkes credit default simulator, CDS calibrator, and Monte Carlo k-th-to-default pricer."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hawkes_cache")
logging.getLogger("fontTools").setLevel(logging.ERROR)

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class CreditBasketMarketData:
    """Market data container for an N-entity corporate credit basket."""

    entity_names: list[str]
    sectors: list[str]
    cds_spreads_5y: np.ndarray
    lgd: np.ndarray
    correlation_matrix: np.ndarray
    risk_free_rate: float = 0.04
    maturity: float = 5.0

    @property
    def cds_spreads_bps(self) -> np.ndarray:
        """Return 5-year par CDS spreads in basis points."""
        return self.cds_spreads_5y * 10000.0

    @property
    def recovery_rates(self) -> np.ndarray:
        """Return implied recovery rates R_i = 1 - LGD_i."""
        return 1.0 - self.lgd


@dataclass(frozen=True)
class CalibratedHawkesParameters(Mapping[str, Any]):
    """Calibrated multivariate Hawkes parameters and market metadata for a credit basket."""

    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    lgd: np.ndarray
    entity_names: list[str]
    sectors: list[str]
    cds_spreads_5y: np.ndarray
    correlation_matrix: np.ndarray
    risk_free_rate: float = 0.04
    maturity: float = 5.0
    absorbing_defaults: bool = True

    _SIMULATOR_KEYS: Tuple[str, ...] = (
        "mu",
        "alpha",
        "beta",
        "lgd",
        "entity_names",
        "absorbing_defaults",
    )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._SIMULATOR_KEYS)

    def __len__(self) -> int:
        return len(self._SIMULATOR_KEYS)

    @property
    def recovery_rates(self) -> np.ndarray:
        """Return implied recovery rates R_i = 1 - LGD_i."""
        return 1.0 - self.lgd

    @property
    def branching_matrix(self) -> np.ndarray:
        """Return the integrated excitation matrix Gamma = alpha / beta."""
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        """Return the spectral radius rho(alpha / beta)."""
        return float(np.max(np.abs(np.linalg.eigvals(self.branching_matrix))))

    def to_simulator_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by MultivariateHawkesCreditSimulator."""
        return {
            "mu": self.mu.copy(),
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "lgd": self.lgd.copy(),
            "entity_names": list(self.entity_names),
            "absorbing_defaults": self.absorbing_defaults,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of all calibrated parameters and market inputs."""
        return {
            "mu": self.mu.copy(),
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "lgd": self.lgd.copy(),
            "recovery_rates": self.recovery_rates,
            "entity_names": list(self.entity_names),
            "sectors": list(self.sectors),
            "cds_spreads_5y": self.cds_spreads_5y.copy(),
            "correlation_matrix": self.correlation_matrix.copy(),
            "risk_free_rate": self.risk_free_rate,
            "maturity": self.maturity,
            "absorbing_defaults": self.absorbing_defaults,
            "spectral_radius": self.spectral_radius,
        }

    def create_simulator(self) -> MultivariateHawkesCreditSimulator:
        """Instantiate a MultivariateHawkesCreditSimulator from these calibrated parameters."""
        return MultivariateHawkesCreditSimulator(**self.to_simulator_kwargs())


@dataclass(frozen=True)
class DefaultEvent:
    """Record of a single simulated default event in the credit basket."""

    event_index: int
    timestamp: float
    entity_id: int
    entity_name: str
    pre_jump_intensities: np.ndarray
    post_jump_intensities: np.ndarray
    portfolio_intensity_pre: float
    portfolio_intensity_post: float
    thinning_upper_bound: float
    lgd: float = 0.60


@dataclass
class HawkesSimulationResult:
    """Output container for a single multivariate Hawkes default trajectory."""

    horizon: float
    entity_names: list[str]
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    event_log: list[DefaultEvent]
    event_times: np.ndarray
    event_entities: np.ndarray
    pre_jump_intensities: np.ndarray
    post_jump_intensities: np.ndarray
    portfolio_intensity_pre: np.ndarray
    portfolio_intensity_post: np.ndarray
    thinning_upper_bounds: np.ndarray
    times: np.ndarray
    intensities: np.ndarray
    portfolio_intensity: np.ndarray
    n_proposals: int
    n_accepted: int
    n_rejected: int
    absorbing_defaults: bool = True
    lgd: np.ndarray = field(default_factory=lambda: np.full(5, 0.60, dtype=np.float64))
    event_lgds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    _event_log_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _timeseries_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def acceptance_rate(self) -> float:
        """Return the ratio of accepted defaults to total Ogata thinning proposals."""
        if self.n_proposals == 0:
            return 0.0
        return float(self.n_accepted) / float(self.n_proposals)

    @property
    def first_to_default(self) -> Optional[DefaultEvent]:
        """Return the first default event on the trajectory, or None if no default occurred."""
        if not self.event_log:
            return None
        return self.event_log[0]

    @property
    def total_portfolio_loss(self) -> float:
        """Return cumulative Loss Given Default realized across all accepted defaults."""
        return float(np.sum(self.event_lgds))

    @property
    def event_log_df(self) -> pd.DataFrame:
        """Return the simulated default event log as a pandas DataFrame."""
        if self._event_log_df is None:
            names_arr = np.asarray(self.entity_names, dtype=str)
            pre_cols = np.char.add("lambda_pre_", names_arr).tolist()
            post_cols = np.char.add("lambda_post_", names_arr).tolist()
            base_cols = [
                "event_index",
                "timestamp",
                "entity_id",
                "entity_name",
                "lgd",
                "portfolio_intensity_pre",
                "portfolio_intensity_post",
                "thinning_upper_bound",
            ]
            all_cols = base_cols + pre_cols + post_cols

            k = self.event_times.shape[0]
            if k == 0:
                self._event_log_df = pd.DataFrame(columns=all_cols)
            else:
                numeric_cols = (
                    [
                        "lgd",
                        "portfolio_intensity_pre",
                        "portfolio_intensity_post",
                        "thinning_upper_bound",
                    ]
                    + pre_cols
                    + post_cols
                )
                numeric_mat = np.column_stack(
                    [
                        self.event_lgds,
                        self.portfolio_intensity_pre,
                        self.portfolio_intensity_post,
                        self.thinning_upper_bounds,
                        self.pre_jump_intensities,
                        self.post_jump_intensities,
                    ]
                )
                df = pd.DataFrame(numeric_mat, columns=numeric_cols)
                df.insert(0, "entity_name", names_arr[self.event_entities])
                df.insert(0, "entity_id", self.event_entities)
                df.insert(0, "timestamp", self.event_times)
                df.insert(0, "event_index", np.arange(k, dtype=np.int64))
                self._event_log_df = df
        return self._event_log_df

    @property
    def timeseries_df(self) -> pd.DataFrame:
        """Return the hazard rate timeseries lambda_i(t) and Lambda(t) as a pandas DataFrame."""
        if self._timeseries_df is None:
            names_arr = np.asarray(self.entity_names, dtype=str)
            lambda_cols = np.char.add("lambda_", names_arr).tolist()
            cols = ["time"] + lambda_cols + ["portfolio_intensity"]
            mat = np.column_stack([self.times, self.intensities, self.portfolio_intensity])
            self._timeseries_df = pd.DataFrame(mat, columns=cols)
        return self._timeseries_df


@dataclass
class FTDSwapPricingResult:
    """Output container for Monte Carlo First-to-Default and k-th-to-Default swap valuation."""

    fair_spread: float
    first_default_times: np.ndarray
    first_default_entities: np.ndarray
    first_default_lgds: np.ndarray
    defaulted_mask: np.ndarray
    protection_leg_pvs: np.ndarray
    premium_leg_pvs: np.ndarray
    risky_annuities: np.ndarray
    expected_protection_pv: float
    expected_premium_pv: float
    expected_risky_annuity: float
    net_present_value: float
    default_probability: float
    entity_ftd_counts: np.ndarray
    entity_ftd_probabilities: np.ndarray
    spread_std_error: float
    n_paths: int
    horizon: float
    risk_free_rate: float
    notional: float
    entity_names: list[str]
    kth_default_times: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=np.float64))
    kth_default_entities: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=np.int64))
    kth_default_lgds: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=np.float64))
    kth_defaulted_mask: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=bool))
    kth_fair_spreads: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_spread_std_errors: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_expected_protection_pvs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_expected_premium_pvs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_expected_risky_annuities: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_default_probabilities: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    kth_expected_stopping_times: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    _path_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _kth_summary_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __iter__(self) -> Iterator[Any]:
        """Yield fair_spread and first_default_times for 2-tuple unpacking."""
        yield self.fair_spread
        yield self.first_default_times

    def __getitem__(self, key: Union[int, str]) -> Any:
        if isinstance(key, int):
            if key == 0:
                return self.fair_spread
            if key == 1:
                return self.first_default_times
            raise IndexError(key)
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(key)

    @property
    def fair_spread_bps(self) -> float:
        """Return the fair 1st-to-Default spread in basis points."""
        return self.fair_spread * 10000.0

    @property
    def spread_std_error_bps(self) -> float:
        """Return the Monte Carlo standard error of the 1st-to-Default spread in basis points."""
        return self.spread_std_error * 10000.0

    @property
    def kth_fair_spreads_bps(self) -> np.ndarray:
        """Return fair k-th-to-Default spreads (k = 1..N) in basis points."""
        return self.kth_fair_spreads * 10000.0

    @property
    def kth_spread_std_errors_bps(self) -> np.ndarray:
        """Return Monte Carlo standard errors for k-th-to-Default spreads in basis points."""
        return self.kth_spread_std_errors * 10000.0

    @property
    def protection_leg_pv(self) -> float:
        """Return expected present value of the 1st-to-Default protection leg."""
        return self.expected_protection_pv

    @property
    def premium_leg_pv(self) -> float:
        """Return expected present value of the 1st-to-Default premium leg."""
        return self.expected_premium_pv

    @property
    def realized_first_default_times(self) -> np.ndarray:
        """Return finite first default timestamps tau_1 <= T."""
        return self.first_default_times[self.defaulted_mask]

    def realized_kth_default_times(self, k: int) -> np.ndarray:
        """Return finite k-th default timestamps tau_k <= T for tranche order k in 1..N."""
        n_tranches = self.kth_default_times.shape[1]
        if k < 1 or k > n_tranches:
            raise ValueError(f"Tranche index `k` must lie in 1..{n_tranches}, got {k}.")
        col = k - 1
        return self.kth_default_times[self.kth_defaulted_mask[:, col], col]

    @property
    def mean_first_default_time(self) -> float:
        """Return conditional mean first default time across paths defaulting in [0, T]."""
        realized = self.realized_first_default_times
        if realized.size == 0:
            return float("nan")
        return float(np.mean(realized))

    @property
    def effective_stopping_times(self) -> np.ndarray:
        """Return effective contract stopping times min(tau_1, T) across all paths."""
        return np.minimum(self.first_default_times, self.horizon)

    @property
    def kth_summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame of k-th-to-Default pricing metrics across k = 1..N."""
        if self._kth_summary_df is None:
            n_tranches = self.kth_fair_spreads.shape[0]
            cols = [
                "k",
                "default_probability",
                "expected_stopping_time",
                "expected_protection_pv",
                "expected_premium_pv",
                "expected_risky_annuity",
                "fair_spread",
                "fair_spread_bps",
                "spread_std_error_bps",
            ]
            mat = np.column_stack(
                [
                    np.arange(1, n_tranches + 1, dtype=np.int64),
                    self.kth_default_probabilities,
                    self.kth_expected_stopping_times,
                    self.kth_expected_protection_pvs,
                    self.kth_expected_premium_pvs,
                    self.kth_expected_risky_annuities,
                    self.kth_fair_spreads,
                    self.kth_fair_spreads_bps,
                    self.kth_spread_std_errors_bps,
                ]
            )
            df = pd.DataFrame(mat, columns=cols)
            df["k"] = df["k"].astype(np.int64)
            self._kth_summary_df = df
        return self._kth_summary_df

    @property
    def path_df(self) -> pd.DataFrame:
        """Return a path-by-path DataFrame of simulated 1st-to-Default swap outcomes."""
        if self._path_df is None:
            cols = [
                "path_index",
                "first_default_time",
                "effective_stopping_time",
                "defaulted",
                "first_default_entity",
                "first_default_lgd",
                "protection_leg_pv",
                "premium_leg_pv",
                "risky_annuity",
            ]
            mat = np.column_stack(
                [
                    np.arange(self.n_paths, dtype=np.int64),
                    self.first_default_times,
                    self.effective_stopping_times,
                    self.defaulted_mask.astype(np.int64),
                    self.first_default_entities,
                    self.first_default_lgds,
                    self.protection_leg_pvs,
                    self.premium_leg_pvs,
                    self.risky_annuities,
                ]
            )
            self._path_df = pd.DataFrame(mat, columns=cols)
        return self._path_df

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of all pricing outputs, stopping times, and leg present values."""
        return {
            "fair_spread": self.fair_spread,
            "fair_spread_bps": self.fair_spread_bps,
            "spread_std_error": self.spread_std_error,
            "spread_std_error_bps": self.spread_std_error_bps,
            "first_default_times": self.first_default_times.copy(),
            "realized_first_default_times": self.realized_first_default_times.copy(),
            "mean_first_default_time": self.mean_first_default_time,
            "effective_stopping_times": self.effective_stopping_times.copy(),
            "first_default_entities": self.first_default_entities.copy(),
            "first_default_lgds": self.first_default_lgds.copy(),
            "defaulted_mask": self.defaulted_mask.copy(),
            "protection_leg_pvs": self.protection_leg_pvs.copy(),
            "premium_leg_pvs": self.premium_leg_pvs.copy(),
            "risky_annuities": self.risky_annuities.copy(),
            "expected_protection_pv": self.expected_protection_pv,
            "expected_premium_pv": self.expected_premium_pv,
            "expected_risky_annuity": self.expected_risky_annuity,
            "net_present_value": self.net_present_value,
            "default_probability": self.default_probability,
            "entity_ftd_counts": self.entity_ftd_counts.copy(),
            "entity_ftd_probabilities": self.entity_ftd_probabilities.copy(),
            "kth_default_times": self.kth_default_times.copy(),
            "kth_default_entities": self.kth_default_entities.copy(),
            "kth_default_lgds": self.kth_default_lgds.copy(),
            "kth_defaulted_mask": self.kth_defaulted_mask.copy(),
            "kth_fair_spreads": self.kth_fair_spreads.copy(),
            "kth_fair_spreads_bps": self.kth_fair_spreads_bps,
            "kth_spread_std_errors": self.kth_spread_std_errors.copy(),
            "kth_spread_std_errors_bps": self.kth_spread_std_errors_bps,
            "kth_expected_protection_pvs": self.kth_expected_protection_pvs.copy(),
            "kth_expected_premium_pvs": self.kth_expected_premium_pvs.copy(),
            "kth_expected_risky_annuities": self.kth_expected_risky_annuities.copy(),
            "kth_default_probabilities": self.kth_default_probabilities.copy(),
            "kth_expected_stopping_times": self.kth_expected_stopping_times.copy(),
            "n_paths": self.n_paths,
            "horizon": self.horizon,
            "risk_free_rate": self.risk_free_rate,
            "notional": self.notional,
            "entity_names": list(self.entity_names),
        }


@dataclass
class ContagionComparisonResult:
    """Comparison of k-th-to-Default swap pricing between Hawkes contagion and independent Poisson models."""

    hawkes_result: FTDSwapPricingResult
    baseline_result: FTDSwapPricingResult
    contagion_premium_bps: np.ndarray
    _comparison_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def comparison_df(self) -> pd.DataFrame:
        """Return a DataFrame comparing Hawkes and independent baseline metrics across k = 1..N."""
        if self._comparison_df is None:
            n_tranches = self.contagion_premium_bps.shape[0]
            cols = [
                "k",
                "hawkes_default_prob",
                "baseline_default_prob",
                "hawkes_expected_tau",
                "baseline_expected_tau",
                "hawkes_spread_bps",
                "hawkes_se_bps",
                "baseline_spread_bps",
                "baseline_se_bps",
                "contagion_premium_bps",
            ]
            mat = np.column_stack(
                [
                    np.arange(1, n_tranches + 1, dtype=np.int64),
                    self.hawkes_result.kth_default_probabilities,
                    self.baseline_result.kth_default_probabilities,
                    self.hawkes_result.kth_expected_stopping_times,
                    self.baseline_result.kth_expected_stopping_times,
                    self.hawkes_result.kth_fair_spreads_bps,
                    self.hawkes_result.kth_spread_std_errors_bps,
                    self.baseline_result.kth_fair_spreads_bps,
                    self.baseline_result.kth_spread_std_errors_bps,
                    self.contagion_premium_bps,
                ]
            )
            df = pd.DataFrame(mat, columns=cols)
            df["k"] = df["k"].astype(np.int64)
            self._comparison_df = df
        return self._comparison_df


def generate_synthetic_market_data(
    seed: Optional[Union[int, np.random.Generator]] = None,
    risk_free_rate: float = 0.04,
    maturity: float = 5.0,
) -> CreditBasketMarketData:
    """Generate synthetic 5-name credit basket market data with sector correlation structure."""
    entity_names = [
        "FIN_Bank_Alpha",
        "FIN_Bank_Beta",
        "ENR_Oil_Gamma",
        "ENR_Gas_Delta",
        "IND_Aero_Epsilon",
    ]
    sectors = [
        "Financials",
        "Financials",
        "Energy",
        "Energy",
        "Industrial",
    ]
    cds_spreads_5y = np.array([0.0320, 0.0360, 0.0420, 0.0460, 0.0290], dtype=np.float64)
    lgd = np.array([0.60, 0.60, 0.55, 0.55, 0.65], dtype=np.float64)
    correlation_matrix = np.array(
        [
            [1.00, 0.82, 0.26, 0.24, 0.30],
            [0.82, 1.00, 0.25, 0.28, 0.32],
            [0.26, 0.25, 1.00, 0.78, 0.29],
            [0.24, 0.28, 0.78, 1.00, 0.22],
            [0.30, 0.32, 0.29, 0.22, 1.00],
        ],
        dtype=np.float64,
    )

    if seed is not None:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        spread_shock = rng.uniform(0.92, 1.08, size=5)
        cds_spreads_5y = np.round(cds_spreads_5y * spread_shock, 6)

    return CreditBasketMarketData(
        entity_names=entity_names,
        sectors=sectors,
        cds_spreads_5y=cds_spreads_5y,
        lgd=lgd,
        correlation_matrix=correlation_matrix,
        risk_free_rate=float(risk_free_rate),
        maturity=float(maturity),
    )


def extract_baseline_hazard_rates_from_cds(
    cds_spreads_5y: Union[np.ndarray, Sequence[float]],
    lgd: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.04,
    maturity: float = 5.0,
    payment_frequency: int = 4,
    method: str = "par_bootstrap",
) -> np.ndarray:
    """Bootstrap baseline hazard rates mu_i from 5-year par CDS spreads and entity LGDs."""
    spreads = np.asarray(cds_spreads_5y, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)

    if spreads.ndim != 1 or spreads.size == 0:
        raise ValueError("`cds_spreads_5y` must be a non-empty 1D array.")
    if lgd_arr.shape != spreads.shape:
        raise ValueError("`lgd` must have the same shape as `cds_spreads_5y`.")
    if np.any(spreads <= 0.0) or not np.all(np.isfinite(spreads)):
        raise ValueError("All 5-year CDS spreads must be finite and strictly positive.")
    if np.any(lgd_arr <= 0.0) or np.any(lgd_arr > 1.0) or not np.all(np.isfinite(lgd_arr)):
        raise ValueError("All `lgd` values must lie in (0, 1].")
    if maturity <= 0.0 or not np.isfinite(maturity):
        raise ValueError("`maturity` must be a finite positive number.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("`risk_free_rate` must be finite.")

    spreads_dec = np.where(spreads > 1.0, spreads * 1e-4, spreads)
    mu = spreads_dec / lgd_arr

    if method == "credit_triangle":
        return mu
    if method != "par_bootstrap":
        raise ValueError(f"Unknown CDS extraction method '{method}'. Use 'par_bootstrap' or 'credit_triangle'.")

    n_periods = max(1, int(round(maturity * payment_frequency)))
    dt = float(maturity) / float(n_periods)
    t_m = np.linspace(dt, maturity, n_periods, dtype=np.float64)[np.newaxis, :]
    t_prev = t_m - dt
    df_m = np.exp(-risk_free_rate * t_m)

    for _ in range(50):
        mu_col = mu[:, np.newaxis]
        surv_m = np.exp(-mu_col * t_m)
        surv_prev = np.exp(-mu_col * t_prev)
        rpv01 = np.sum(dt * df_m * 0.5 * (surv_prev + surv_m), axis=1)

        r_plus_mu = risk_free_rate + mu
        denom = np.where(
            np.abs(r_plus_mu) > 1e-12,
            (1.0 - np.exp(-r_plus_mu * maturity)) / r_plus_mu,
            maturity,
        )

        # Par CDS identity: LGD_i * mu_i * int_0^T e^{-(r + mu_i) t} dt = S_i * RPV01_i(mu_i)
        mu_next = (spreads_dec / lgd_arr) * (rpv01 / denom)
        max_err = float(np.max(np.abs(mu_next - mu)))
        mu = mu_next
        if max_err <= 1e-14:
            break

    return mu


def reprice_par_cds_spreads_bps(
    mu: Union[np.ndarray, Sequence[float]],
    lgd: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.04,
    maturity: float = 5.0,
    payment_frequency: int = 4,
) -> np.ndarray:
    """Reprice single-name par CDS spreads in basis points from hazard rates mu_i and LGDs."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)
    n_periods = max(1, int(round(maturity * payment_frequency)))
    dt = float(maturity) / float(n_periods)
    t_m = np.linspace(dt, maturity, n_periods, dtype=np.float64)[np.newaxis, :]
    t_prev = t_m - dt
    df_m = np.exp(-risk_free_rate * t_m)

    mu_col = mu_arr[:, np.newaxis]
    rpv01 = np.sum(
        dt * df_m * 0.5 * (np.exp(-mu_col * t_prev) + np.exp(-mu_col * t_m)),
        axis=1,
    )
    pv_prot = (
        lgd_arr
        * (mu_arr / (risk_free_rate + mu_arr))
        * (1.0 - np.exp(-(risk_free_rate + mu_arr) * maturity))
    )
    return (pv_prot / rpv01) * 10000.0


def map_correlation_to_contagion_matrix(
    correlation_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
    sectors: Sequence[str],
    lgd: Union[np.ndarray, Sequence[float]],
    beta: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    intra_sector_multiplier: float = 2.40,
    inter_sector_multiplier: float = 0.65,
    contagion_scale: float = 0.30,
    self_excitation_scale: float = 0.0,
    correlation_exponent: float = 1.25,
    target_spectral_radius: Optional[float] = 0.68,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a correlation matrix and sector topology to zero-diagonal contagion and decay matrices."""
    corr = np.asarray(correlation_matrix, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)
    sectors_arr = np.asarray(sectors, dtype=str)
    n = lgd_arr.shape[0]

    if corr.shape != (n, n):
        raise ValueError(f"`correlation_matrix` must have shape ({n}, {n}), got {corr.shape}.")
    if sectors_arr.shape != (n,):
        raise ValueError(f"`sectors` must have length {n}, got {sectors_arr.shape}.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("`correlation_matrix` must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
        raise ValueError("Diagonal entries of `correlation_matrix` must equal 1.0.")
    if intra_sector_multiplier <= inter_sector_multiplier:
        raise ValueError(
            "`intra_sector_multiplier` must exceed `inter_sector_multiplier` so same-sector "
            "entities have higher excitation coefficients."
        )

    same_sector_mask = sectors_arr[:, np.newaxis] == sectors_arr[np.newaxis, :]
    eye_mask = np.eye(n, dtype=bool)
    intra_off_diag_mask = same_sector_mask & (~eye_mask)

    sector_weights = np.where(
        eye_mask,
        0.0,
        np.where(intra_off_diag_mask, intra_sector_multiplier, inter_sector_multiplier),
    )

    pos_corr = np.clip(corr, a_min=0.0, a_max=1.0) ** correlation_exponent
    lgd_severity = lgd_arr[np.newaxis, :] / np.mean(lgd_arr)

    alpha = contagion_scale * pos_corr * sector_weights * lgd_severity
    np.fill_diagonal(alpha, 0.0)

    if beta is None:
        beta_mat = np.where(
            eye_mask,
            1.40,
            np.where(intra_off_diag_mask, 1.15, 1.85),
        ).astype(np.float64)
    else:
        beta_mat = np.asarray(beta, dtype=np.float64).copy()
        if beta_mat.shape != (n, n) or np.any(beta_mat <= 0.0):
            raise ValueError(f"`beta` must be a positive ({n}, {n}) matrix.")

    if target_spectral_radius is not None:
        if target_spectral_radius <= 0.0:
            raise ValueError("`target_spectral_radius` must be strictly positive.")
        current_rho = float(np.max(np.abs(np.linalg.eigvals(alpha / beta_mat))))
        if current_rho > 0.0:
            alpha = alpha * (float(target_spectral_radius) / current_rho)
            np.fill_diagonal(alpha, 0.0)

    return alpha, beta_mat


def calibrate_hawkes_parameters(
    market_data: Optional[CreditBasketMarketData] = None,
    *,
    entity_names: Optional[Sequence[str]] = None,
    sectors: Optional[Sequence[str]] = None,
    cds_spreads_5y: Optional[Union[np.ndarray, Sequence[float]]] = None,
    lgd: Optional[Union[np.ndarray, Sequence[float]]] = None,
    correlation_matrix: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    beta: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    risk_free_rate: float = 0.04,
    maturity: float = 5.0,
    cds_method: str = "par_bootstrap",
    intra_sector_multiplier: float = 2.40,
    inter_sector_multiplier: float = 0.65,
    contagion_scale: float = 0.30,
    self_excitation_scale: float = 0.0,
    correlation_exponent: float = 1.25,
    target_spectral_radius: Optional[float] = 0.68,
    absorbing_defaults: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> CalibratedHawkesParameters:
    """Calibrate multivariate Hawkes parameters (mu, alpha, beta, lgd) from credit basket market data."""
    if market_data is None:
        if (
            cds_spreads_5y is None
            and lgd is None
            and correlation_matrix is None
            and sectors is None
        ):
            market_data = generate_synthetic_market_data(
                seed=seed,
                risk_free_rate=risk_free_rate,
                maturity=maturity,
            )
        else:
            default_md = generate_synthetic_market_data(
                seed=seed,
                risk_free_rate=risk_free_rate,
                maturity=maturity,
            )
            market_data = CreditBasketMarketData(
                entity_names=list(entity_names) if entity_names is not None else default_md.entity_names,
                sectors=list(sectors) if sectors is not None else default_md.sectors,
                cds_spreads_5y=(
                    np.asarray(cds_spreads_5y, dtype=np.float64)
                    if cds_spreads_5y is not None
                    else default_md.cds_spreads_5y
                ),
                lgd=np.asarray(lgd, dtype=np.float64) if lgd is not None else default_md.lgd,
                correlation_matrix=(
                    np.asarray(correlation_matrix, dtype=np.float64)
                    if correlation_matrix is not None
                    else default_md.correlation_matrix
                ),
                risk_free_rate=float(risk_free_rate),
                maturity=float(maturity),
            )

    mu = extract_baseline_hazard_rates_from_cds(
        cds_spreads_5y=market_data.cds_spreads_5y,
        lgd=market_data.lgd,
        risk_free_rate=market_data.risk_free_rate,
        maturity=market_data.maturity,
        method=cds_method,
    )

    alpha, beta_mat = map_correlation_to_contagion_matrix(
        correlation_matrix=market_data.correlation_matrix,
        sectors=market_data.sectors,
        lgd=market_data.lgd,
        beta=beta,
        intra_sector_multiplier=intra_sector_multiplier,
        inter_sector_multiplier=inter_sector_multiplier,
        contagion_scale=contagion_scale,
        self_excitation_scale=self_excitation_scale,
        correlation_exponent=correlation_exponent,
        target_spectral_radius=target_spectral_radius,
    )

    return CalibratedHawkesParameters(
        mu=mu,
        alpha=alpha,
        beta=beta_mat,
        lgd=market_data.lgd.copy(),
        entity_names=list(market_data.entity_names),
        sectors=list(market_data.sectors),
        cds_spreads_5y=market_data.cds_spreads_5y.copy(),
        correlation_matrix=market_data.correlation_matrix.copy(),
        risk_free_rate=market_data.risk_free_rate,
        maturity=market_data.maturity,
        absorbing_defaults=absorbing_defaults,
    )


def compute_entity_intensities(
    t: Union[float, np.ndarray],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    event_times: np.ndarray,
    event_entities: np.ndarray,
    right_continuous: bool = True,
) -> np.ndarray:
    """Evaluate entity hazard rates lambda_i(t) across scalar or vector timestamps via NumPy broadcasting."""
    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    mu_arr = np.asarray(mu, dtype=np.float64)
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    ev_times = np.asarray(event_times, dtype=np.float64)
    ev_entities = np.asarray(event_entities, dtype=np.int64)

    n_entities = mu_arr.shape[0]
    m = t_arr.shape[0]
    k = ev_times.shape[0]

    if k == 0:
        lambdas = np.broadcast_to(mu_arr[np.newaxis, :], (m, n_entities)).copy()
    else:
        dt = t_arr[:, np.newaxis] - ev_times[np.newaxis, :]
        valid = (dt >= 0.0) if right_continuous else (dt > 0.0)
        dt_safe = np.where(valid, dt, 0.0)

        alpha_cols = alpha_arr[:, ev_entities].T
        beta_cols = beta_arr[:, ev_entities].T

        excitation = np.where(
            valid[:, :, np.newaxis],
            alpha_cols[np.newaxis, :, :]
            * np.exp(-beta_cols[np.newaxis, :, :] * dt_safe[:, :, np.newaxis]),
            0.0,
        )
        lambdas = mu_arr[np.newaxis, :] + np.sum(excitation, axis=1)

    if scalar_input:
        return lambdas[0]
    return lambdas


def compute_portfolio_intensity(
    t: Union[float, np.ndarray],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    event_times: np.ndarray,
    event_entities: np.ndarray,
    right_continuous: bool = True,
) -> Union[float, np.ndarray]:
    """Evaluate aggregate portfolio intensity Lambda(t) = sum_i lambda_i(t) at scalar or vector timestamps."""
    lambdas = compute_entity_intensities(
        t=t,
        mu=mu,
        alpha=alpha,
        beta=beta,
        event_times=event_times,
        event_entities=event_entities,
        right_continuous=right_continuous,
    )
    portfolio_lambda = np.sum(lambdas, axis=-1)
    if np.ndim(t) == 0:
        return float(portfolio_lambda)
    return portfolio_lambda


class MultivariateHawkesCreditSimulator:
    """Multivariate Hawkes credit default simulator using Ogata's modified thinning algorithm."""

    def __init__(
        self,
        mu: Union[np.ndarray, Sequence[float], CalibratedHawkesParameters, Mapping[str, Any]],
        alpha: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        beta: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        entity_names: Optional[Sequence[str]] = None,
        absorbing_defaults: Optional[bool] = None,
        lgd: Optional[Union[np.ndarray, Sequence[float]]] = None,
        risk_free_rate: float = 0.04,
    ) -> None:
        if isinstance(mu, (CalibratedHawkesParameters, Mapping)):
            calib = mu
            mu_val = calib["mu"]
            alpha_val = calib["alpha"] if alpha is None else alpha
            beta_val = calib["beta"] if beta is None else beta
            if entity_names is None and "entity_names" in calib:
                entity_names = calib["entity_names"]
            if lgd is None and "lgd" in calib:
                lgd = calib["lgd"]
            if absorbing_defaults is None:
                absorbing_defaults = bool(calib.get("absorbing_defaults", True))
            if "risk_free_rate" in calib:
                risk_free_rate = float(calib["risk_free_rate"])
        else:
            if alpha is None or beta is None:
                raise ValueError(
                    "Both `alpha` and `beta` must be provided when `mu` is a baseline vector."
                )
            mu_val = mu
            alpha_val = alpha
            beta_val = beta
            if absorbing_defaults is None:
                absorbing_defaults = True

        self.mu = np.asarray(mu_val, dtype=np.float64)
        self.alpha = np.asarray(alpha_val, dtype=np.float64)
        self.beta = np.asarray(beta_val, dtype=np.float64)
        self.absorbing_defaults = bool(absorbing_defaults)
        self.risk_free_rate = float(risk_free_rate)

        self.n_entities: int = int(self.mu.shape[0]) if self.mu.ndim == 1 else 0
        if lgd is None:
            self.lgd = np.full(max(1, self.n_entities), 0.60, dtype=np.float64)
        else:
            self.lgd = np.asarray(lgd, dtype=np.float64)

        self._validate_parameters()

        if entity_names is None:
            self.entity_names = np.char.add(
                "Entity_", np.arange(self.n_entities).astype(str)
            ).tolist()
        else:
            if len(entity_names) != self.n_entities:
                raise ValueError(
                    f"Length of entity_names ({len(entity_names)}) must match "
                    f"number of entities ({self.n_entities})."
                )
            self.entity_names = list(entity_names)

    @classmethod
    def from_calibration(
        cls,
        calibrated_params: Union[CalibratedHawkesParameters, Mapping[str, Any]],
    ) -> MultivariateHawkesCreditSimulator:
        """Instantiate a simulator directly from a CalibratedHawkesParameters instance or mapping."""
        return cls(calibrated_params)

    def _validate_parameters(self) -> None:
        """Validate dimensions and positivity constraints on mu, alpha, beta, and lgd."""
        if self.mu.ndim != 1 or self.mu.size == 0:
            raise ValueError("Baseline intensity vector `mu` must be a non-empty 1D array.")

        n = self.mu.shape[0]
        if self.alpha.shape != (n, n):
            raise ValueError(
                f"Contagion matrix `alpha` must have shape ({n}, {n}), got {self.alpha.shape}."
            )
        if self.beta.shape != (n, n):
            raise ValueError(
                f"Decay matrix `beta` must have shape ({n}, {n}), got {self.beta.shape}."
            )
        if self.lgd.shape != (n,):
            raise ValueError(
                f"Loss Given Default vector `lgd` must have shape ({n},), got {self.lgd.shape}."
            )

        if not np.all(np.isfinite(self.mu)) or np.any(self.mu <= 0.0):
            raise ValueError("All baseline intensities `mu_i` must be finite and strictly positive.")
        if not np.all(np.isfinite(self.alpha)) or np.any(self.alpha < 0.0):
            raise ValueError("All contagion entries `alpha_{ij}` must be finite and non-negative.")
        if not np.all(np.isfinite(self.beta)) or np.any(self.beta <= 0.0):
            raise ValueError("All exponential decay rates `beta_{ij}` must be finite and strictly positive.")
        if not np.all(np.isfinite(self.lgd)) or np.any(self.lgd <= 0.0) or np.any(self.lgd > 1.0):
            raise ValueError("All `lgd` entries must be finite and lie in (0, 1].")

    @property
    def branching_matrix(self) -> np.ndarray:
        """Return the integrated excitation matrix Gamma = alpha / beta."""
        return self.alpha / self.beta

    @property
    def spectral_radius(self) -> float:
        """Return the spectral radius rho(alpha / beta)."""
        eigenvalues = np.linalg.eigvals(self.branching_matrix)
        return float(np.max(np.abs(eigenvalues)))

    @property
    def stationary_intensities(self) -> Optional[np.ndarray]:
        """Return the asymptotic mean intensity vector (I - Gamma)^{-1} mu when subcritical."""
        if self.spectral_radius >= 1.0:
            return None
        identity = np.eye(self.n_entities, dtype=np.float64)
        return np.linalg.solve(identity - self.branching_matrix, self.mu)

    def compute_intensities_from_state(
        self,
        excitation_state: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute entity hazard rates lambda(t) and active portfolio intensity Lambda(t) from state matrix."""
        lambdas = self.mu + np.sum(excitation_state, axis=1)
        if active_mask is not None:
            effective_lambdas = np.where(active_mask, lambdas, 0.0)
            portfolio_lambda = float(np.sum(effective_lambdas))
        else:
            portfolio_lambda = float(np.sum(lambdas))
        return lambdas, portfolio_lambda

    def evaluate_intensities(
        self,
        t: Union[float, np.ndarray],
        event_times: np.ndarray,
        event_entities: np.ndarray,
        right_continuous: bool = True,
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Evaluate instantaneous hazard rates lambda_i(t) and Lambda(t) across timestamps t."""
        lambdas = compute_entity_intensities(
            t=t,
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            event_times=event_times,
            event_entities=event_entities,
            right_continuous=right_continuous,
        )
        portfolio_lambda = np.sum(lambdas, axis=-1)
        if np.ndim(t) == 0:
            return lambdas, float(portfolio_lambda)
        return lambdas, portfolio_lambda

    def build_time_grid(
        self,
        horizon: float,
        event_times: np.ndarray,
        n_grid_points: int = 2501,
        include_event_jumps: bool = True,
        pre_jump_eps: float = 1e-9,
    ) -> np.ndarray:
        """Construct a sorted time grid over [0, horizon] augmented with pre- and post-jump timestamps."""
        base_grid = np.linspace(0.0, float(horizon), int(n_grid_points), dtype=np.float64)
        if not include_event_jumps or event_times.size == 0:
            return base_grid

        pre_jump_times = np.clip(event_times - pre_jump_eps, a_min=0.0, a_max=horizon)
        combined = np.concatenate([base_grid, pre_jump_times, event_times])
        return np.unique(combined)

    def simulate(
        self,
        horizon: float = 5.0,
        seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = None,
        n_grid_points: int = 2501,
        time_grid: Optional[np.ndarray] = None,
        include_event_jumps: bool = True,
        include_timeseries: bool = True,
    ) -> HawkesSimulationResult:
        """Simulate interacting default times over [0, horizon] via Ogata's modified thinning algorithm."""
        if horizon <= 0.0 or not np.isfinite(horizon):
            raise ValueError("Simulation `horizon` must be a finite positive number.")

        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        t = 0.0
        excitation_state = np.zeros((self.n_entities, self.n_entities), dtype=np.float64)
        active_mask = (
            np.ones(self.n_entities, dtype=bool) if self.absorbing_defaults else None
        )

        event_log: list[DefaultEvent] = []
        ts_list: list[float] = []
        entity_list: list[int] = []
        lgd_list: list[float] = []
        pre_lambdas_list: list[np.ndarray] = []
        post_lambdas_list: list[np.ndarray] = []
        pre_port_list: list[float] = []
        post_port_list: list[float] = []
        bound_list: list[float] = []

        n_proposals = 0
        n_accepted = 0
        n_rejected = 0

        while t < horizon:
            if active_mask is not None and not np.any(active_mask):
                break

            _, lambda_star = self.compute_intensities_from_state(
                excitation_state, active_mask=active_mask
            )
            if lambda_star <= 0.0:
                break

            s = float(rng.exponential(scale=1.0 / lambda_star))
            t_cand = t + s

            if t_cand > horizon:
                break

            n_proposals += 1

            excitation_cand = excitation_state * np.exp(-self.beta * s)
            lambdas_pre, portfolio_lambda_pre = self.compute_intensities_from_state(
                excitation_cand, active_mask=active_mask
            )

            d = float(rng.uniform(0.0, lambda_star))
            if d <= portfolio_lambda_pre and portfolio_lambda_pre > 0.0:
                effective_pre = (
                    np.where(active_mask, lambdas_pre, 0.0)
                    if active_mask is not None
                    else lambdas_pre
                )
                cum_intensities = np.cumsum(effective_pre)
                j_star = int(np.searchsorted(cum_intensities, d, side="left"))
                if j_star >= self.n_entities:
                    j_star = self.n_entities - 1

                excitation_post = excitation_cand.copy()
                excitation_post[:, j_star] += self.alpha[:, j_star]

                if active_mask is not None:
                    active_mask[j_star] = False

                lambdas_post, portfolio_lambda_post = self.compute_intensities_from_state(
                    excitation_post, active_mask=active_mask
                )

                pre_copy = lambdas_pre.copy()
                post_copy = lambdas_post.copy()
                entity_lgd = float(self.lgd[j_star])

                event_log.append(
                    DefaultEvent(
                        event_index=n_accepted,
                        timestamp=t_cand,
                        entity_id=j_star,
                        entity_name=self.entity_names[j_star],
                        pre_jump_intensities=pre_copy,
                        post_jump_intensities=post_copy,
                        portfolio_intensity_pre=portfolio_lambda_pre,
                        portfolio_intensity_post=portfolio_lambda_post,
                        thinning_upper_bound=lambda_star,
                        lgd=entity_lgd,
                    )
                )
                ts_list.append(t_cand)
                entity_list.append(j_star)
                lgd_list.append(entity_lgd)
                pre_lambdas_list.append(pre_copy)
                post_lambdas_list.append(post_copy)
                pre_port_list.append(portfolio_lambda_pre)
                post_port_list.append(portfolio_lambda_post)
                bound_list.append(lambda_star)

                n_accepted += 1
                t = t_cand
                excitation_state = excitation_post
            else:
                n_rejected += 1
                t = t_cand
                excitation_state = excitation_cand

        event_times = np.asarray(ts_list, dtype=np.float64)
        event_entities = np.asarray(entity_list, dtype=np.int64)
        event_lgds = np.asarray(lgd_list, dtype=np.float64)
        if n_accepted > 0:
            pre_jump_arr = np.vstack(pre_lambdas_list)
            post_jump_arr = np.vstack(post_lambdas_list)
        else:
            pre_jump_arr = np.empty((0, self.n_entities), dtype=np.float64)
            post_jump_arr = np.empty((0, self.n_entities), dtype=np.float64)

        portfolio_pre_arr = np.asarray(pre_port_list, dtype=np.float64)
        portfolio_post_arr = np.asarray(post_port_list, dtype=np.float64)
        upper_bounds_arr = np.asarray(bound_list, dtype=np.float64)

        if include_timeseries:
            if time_grid is not None:
                t_grid = np.asarray(time_grid, dtype=np.float64)
                if include_event_jumps and event_times.size > 0:
                    pre_jump_times = np.clip(event_times - 1e-9, a_min=0.0, a_max=horizon)
                    t_grid = np.unique(np.concatenate([t_grid, pre_jump_times, event_times]))
            else:
                t_grid = self.build_time_grid(
                    horizon=horizon,
                    event_times=event_times,
                    n_grid_points=n_grid_points,
                    include_event_jumps=include_event_jumps,
                )

            intensities, portfolio_intensity = self.evaluate_intensities(
                t=t_grid,
                event_times=event_times,
                event_entities=event_entities,
                right_continuous=True,
            )
        else:
            t_grid = np.empty(0, dtype=np.float64)
            intensities = np.empty((0, self.n_entities), dtype=np.float64)
            portfolio_intensity = np.empty(0, dtype=np.float64)

        return HawkesSimulationResult(
            horizon=float(horizon),
            entity_names=list(self.entity_names),
            mu=self.mu.copy(),
            alpha=self.alpha.copy(),
            beta=self.beta.copy(),
            event_log=event_log,
            event_times=event_times,
            event_entities=event_entities,
            pre_jump_intensities=pre_jump_arr,
            post_jump_intensities=post_jump_arr,
            portfolio_intensity_pre=portfolio_pre_arr,
            portfolio_intensity_post=portfolio_post_arr,
            thinning_upper_bounds=upper_bounds_arr,
            times=t_grid,
            intensities=intensities,
            portfolio_intensity=portfolio_intensity,
            n_proposals=n_proposals,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            absorbing_defaults=self.absorbing_defaults,
            lgd=self.lgd.copy(),
            event_lgds=event_lgds,
        )


class MonteCarloFTDPricer:
    """Monte Carlo pricer for First-to-Default and k-th-to-Default basket swaps."""

    def __init__(
        self,
        simulator: Optional[
            Union[MultivariateHawkesCreditSimulator, CalibratedHawkesParameters, Mapping[str, Any]]
        ] = None,
        risk_free_rate: Optional[float] = None,
        horizon: float = 5.0,
        notional: float = 1.0,
    ) -> None:
        if simulator is None:
            self.simulator = create_default_5_entity_basket(absorbing_defaults=True)
        elif isinstance(simulator, MultivariateHawkesCreditSimulator):
            self.simulator = simulator
        else:
            self.simulator = MultivariateHawkesCreditSimulator(simulator)

        self.risk_free_rate: float = (
            float(risk_free_rate)
            if risk_free_rate is not None
            else float(self.simulator.risk_free_rate)
        )
        self.horizon: float = float(horizon)
        self.notional: float = float(notional)

        if self.horizon <= 0.0 or not np.isfinite(self.horizon):
            raise ValueError("`horizon` must be a finite positive number.")
        if self.notional <= 0.0 or not np.isfinite(self.notional):
            raise ValueError("`notional` must be a finite positive number.")
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("`risk_free_rate` must be finite.")

    def compute_protection_leg_pv(
        self,
        default_times: np.ndarray,
        default_lgds: np.ndarray,
    ) -> np.ndarray:
        """Compute pathwise discounted protection leg present values."""
        tau = np.asarray(default_times, dtype=np.float64)
        lgds = np.asarray(default_lgds, dtype=np.float64)
        defaulted = np.isfinite(tau) & (tau <= self.horizon)
        safe_tau = np.where(defaulted, tau, 0.0)
        return np.where(
            defaulted,
            self.notional * lgds * np.exp(-self.risk_free_rate * safe_tau),
            0.0,
        )

    def compute_risky_annuity(
        self,
        default_times: np.ndarray,
    ) -> np.ndarray:
        """Compute pathwise continuous risky annuities int_0^{min(tau_k, T)} exp(-r t) dt."""
        tau = np.asarray(default_times, dtype=np.float64)
        accrual_end = np.minimum(tau, self.horizon)
        if abs(self.risk_free_rate) > 1e-12:
            return (1.0 - np.exp(-self.risk_free_rate * accrual_end)) / self.risk_free_rate
        return accrual_end.copy()

    def compute_premium_leg_pv(
        self,
        spread: Union[float, np.ndarray],
        default_times: np.ndarray,
    ) -> np.ndarray:
        """Compute pathwise discounted continuous premium leg present values for a given spread."""
        annuities = self.compute_risky_annuity(default_times)
        return np.asarray(spread, dtype=np.float64) * self.notional * annuities

    def solve_fair_spread(
        self,
        protection_leg_pvs: np.ndarray,
        risky_annuities: np.ndarray,
    ) -> float:
        """Solve for the fair continuous swap spread equating expected leg present values."""
        expected_prot = float(np.mean(protection_leg_pvs))
        expected_annuity_notional = float(np.mean(self.notional * risky_annuities))
        if expected_annuity_notional <= 0.0:
            raise ValueError("Expected risky annuity must be strictly positive to solve for fair spread.")
        if expected_prot <= 0.0:
            return 0.0

        upper_bracket = max(1.0, 2.0 * expected_prot / expected_annuity_notional)

        def net_swap_objective(spread_candidate: float) -> float:
            return expected_prot - spread_candidate * expected_annuity_notional

        return float(brentq(net_swap_objective, 0.0, upper_bracket, xtol=1e-15, rtol=1e-15))

    def price(
        self,
        n_paths: int = 10_000,
        seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = None,
    ) -> FTDSwapPricingResult:
        """Price all k-th-to-Default tranches (k = 1..N) across n_paths using pathwise SeedSequence streams."""
        if n_paths <= 0:
            raise ValueError("`n_paths` must be a positive integer.")

        n_paths_int = int(n_paths)
        n_tranches = self.simulator.n_entities

        if isinstance(seed, np.random.SeedSequence):
            child_seeds = seed.spawn(n_paths_int)
        elif isinstance(seed, np.random.Generator):
            base_entropy = int(seed.integers(0, 2**63 - 1))
            child_seeds = np.random.SeedSequence(base_entropy).spawn(n_paths_int)
        else:
            child_seeds = np.random.SeedSequence(seed).spawn(n_paths_int)

        kth_default_times = np.full((n_paths_int, n_tranches), np.inf, dtype=np.float64)
        kth_default_entities = np.full((n_paths_int, n_tranches), -1, dtype=np.int64)
        kth_default_lgds = np.zeros((n_paths_int, n_tranches), dtype=np.float64)

        for path_idx, child_seed in enumerate(child_seeds):
            path_rng = np.random.default_rng(child_seed)
            path_res = self.simulator.simulate(
                horizon=self.horizon,
                seed=path_rng,
                include_timeseries=False,
            )
            k_events = min(path_res.n_accepted, n_tranches)
            if k_events > 0:
                kth_default_times[path_idx, :k_events] = path_res.event_times[:k_events]
                kth_default_entities[path_idx, :k_events] = path_res.event_entities[:k_events]
                kth_default_lgds[path_idx, :k_events] = path_res.event_lgds[:k_events]

        kth_defaulted_mask = np.isfinite(kth_default_times) & (kth_default_times <= self.horizon)
        kth_protection_pvs = self.compute_protection_leg_pv(
            default_times=kth_default_times,
            default_lgds=kth_default_lgds,
        )
        kth_risky_annuities = self.compute_risky_annuity(default_times=kth_default_times)

        kth_expected_protection_pvs = np.mean(kth_protection_pvs, axis=0)
        kth_expected_risky_annuities = np.mean(kth_risky_annuities, axis=0)
        kth_default_probabilities = np.mean(kth_defaulted_mask, axis=0)

        safe_taus = np.where(kth_defaulted_mask, kth_default_times, 0.0)
        default_counts = np.sum(kth_defaulted_mask, axis=0)
        kth_expected_stopping_times = np.where(
            default_counts > 0,
            np.sum(safe_taus, axis=0) / np.maximum(default_counts, 1),
            np.nan,
        )

        kth_fair_spreads = np.array(
            [
                self.solve_fair_spread(
                    protection_leg_pvs=kth_protection_pvs[:, tranche_idx],
                    risky_annuities=kth_risky_annuities[:, tranche_idx],
                )
                for tranche_idx in range(n_tranches)
            ],
            dtype=np.float64,
        )

        kth_premium_pvs = self.compute_premium_leg_pv(
            spread=kth_fair_spreads[np.newaxis, :],
            default_times=kth_default_times,
        )
        kth_expected_premium_pvs = np.mean(kth_premium_pvs, axis=0)

        if n_paths_int > 1:
            # Delta-method variance of ratio estimator S_k* = E[PV_prot] / (Notional * E[A]):
            # Var(S_k*) = Var(PV_prot - S_k* * Notional * A) / (n_paths * (Notional * E[A])^2)
            residuals = (
                kth_protection_pvs
                - kth_fair_spreads[np.newaxis, :] * self.notional * kth_risky_annuities
            )
            kth_spread_std_errors = np.std(residuals, axis=0, ddof=1) / (
                self.notional * kth_expected_risky_annuities * np.sqrt(n_paths_int)
            )
        else:
            kth_spread_std_errors = np.zeros(n_tranches, dtype=np.float64)

        first_default_times = kth_default_times[:, 0]
        first_default_entities = kth_default_entities[:, 0]
        first_default_lgds = kth_default_lgds[:, 0]
        defaulted_mask = kth_defaulted_mask[:, 0]
        protection_leg_pvs = kth_protection_pvs[:, 0]
        premium_leg_pvs = kth_premium_pvs[:, 0]
        risky_annuities = kth_risky_annuities[:, 0]

        fair_spread = float(kth_fair_spreads[0])
        expected_protection_pv = float(kth_expected_protection_pvs[0])
        expected_premium_pv = float(kth_expected_premium_pvs[0])
        expected_risky_annuity = float(kth_expected_risky_annuities[0])
        net_present_value = expected_protection_pv - expected_premium_pv
        default_probability = float(kth_default_probabilities[0])
        spread_std_error = float(kth_spread_std_errors[0])

        valid_entities = first_default_entities[defaulted_mask]
        entity_ftd_counts = np.bincount(
            valid_entities, minlength=n_tranches
        ).astype(np.int64)
        entity_ftd_probabilities = entity_ftd_counts.astype(np.float64) / float(n_paths_int)

        return FTDSwapPricingResult(
            fair_spread=fair_spread,
            first_default_times=first_default_times,
            first_default_entities=first_default_entities,
            first_default_lgds=first_default_lgds,
            defaulted_mask=defaulted_mask,
            protection_leg_pvs=protection_leg_pvs,
            premium_leg_pvs=premium_leg_pvs,
            risky_annuities=risky_annuities,
            expected_protection_pv=expected_protection_pv,
            expected_premium_pv=expected_premium_pv,
            expected_risky_annuity=expected_risky_annuity,
            net_present_value=net_present_value,
            default_probability=default_probability,
            entity_ftd_counts=entity_ftd_counts,
            entity_ftd_probabilities=entity_ftd_probabilities,
            spread_std_error=spread_std_error,
            n_paths=n_paths_int,
            horizon=self.horizon,
            risk_free_rate=self.risk_free_rate,
            notional=self.notional,
            entity_names=list(self.simulator.entity_names),
            kth_default_times=kth_default_times,
            kth_default_entities=kth_default_entities,
            kth_default_lgds=kth_default_lgds,
            kth_defaulted_mask=kth_defaulted_mask,
            kth_fair_spreads=kth_fair_spreads,
            kth_spread_std_errors=kth_spread_std_errors,
            kth_expected_protection_pvs=kth_expected_protection_pvs,
            kth_expected_premium_pvs=kth_expected_premium_pvs,
            kth_expected_risky_annuities=kth_expected_risky_annuities,
            kth_default_probabilities=kth_default_probabilities,
            kth_expected_stopping_times=kth_expected_stopping_times,
        )

    def compare_with_independent_baseline(
        self,
        n_paths: int = 10_000,
        seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = 42,
    ) -> ContagionComparisonResult:
        """Compare Hawkes contagion pricing against an independent Poisson baseline (alpha = 0) under CRN."""
        hawkes_res = self.price(n_paths=n_paths, seed=seed)

        zero_alpha = np.zeros_like(self.simulator.alpha)
        baseline_sim = MultivariateHawkesCreditSimulator(
            mu=self.simulator.mu,
            alpha=zero_alpha,
            beta=self.simulator.beta,
            entity_names=self.simulator.entity_names,
            absorbing_defaults=self.simulator.absorbing_defaults,
            lgd=self.simulator.lgd,
            risk_free_rate=self.risk_free_rate,
        )
        baseline_pricer = MonteCarloFTDPricer(
            simulator=baseline_sim,
            risk_free_rate=self.risk_free_rate,
            horizon=self.horizon,
            notional=self.notional,
        )
        baseline_res = baseline_pricer.price(n_paths=n_paths, seed=seed)

        contagion_premium_bps = (
            hawkes_res.kth_fair_spreads_bps - baseline_res.kth_fair_spreads_bps
        )
        return ContagionComparisonResult(
            hawkes_result=hawkes_res,
            baseline_result=baseline_res,
            contagion_premium_bps=contagion_premium_bps,
        )


def price_ftd_swap(
    simulator: Optional[
        Union[MultivariateHawkesCreditSimulator, CalibratedHawkesParameters, Mapping[str, Any]]
    ] = None,
    n_paths: int = 10_000,
    horizon: float = 5.0,
    risk_free_rate: float = 0.04,
    notional: float = 1.0,
    seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = 42,
) -> FTDSwapPricingResult:
    """Price First-to-Default and k-th-to-Default basket swaps over n_paths Monte Carlo paths."""
    pricer = MonteCarloFTDPricer(
        simulator=simulator,
        risk_free_rate=risk_free_rate,
        horizon=horizon,
        notional=notional,
    )
    return pricer.price(n_paths=n_paths, seed=seed)


def compare_hawkes_vs_independent_kth_default(
    simulator: Optional[
        Union[MultivariateHawkesCreditSimulator, CalibratedHawkesParameters, Mapping[str, Any]]
    ] = None,
    n_paths: int = 10_000,
    horizon: float = 5.0,
    risk_free_rate: float = 0.04,
    notional: float = 1.0,
    seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = 42,
) -> ContagionComparisonResult:
    """Compare Hawkes contagion and independent Poisson k-th-to-Default pricing using pathwise CRN."""
    pricer = MonteCarloFTDPricer(
        simulator=simulator,
        risk_free_rate=risk_free_rate,
        horizon=horizon,
        notional=notional,
    )
    return pricer.compare_with_independent_baseline(n_paths=n_paths, seed=seed)


def create_default_5_entity_basket(
    absorbing_defaults: bool = True,
) -> MultivariateHawkesCreditSimulator:
    """Create a calibrated 5-entity credit basket simulator from synthetic market data."""
    calibrated = calibrate_hawkes_parameters(absorbing_defaults=absorbing_defaults)
    return MultivariateHawkesCreditSimulator(**calibrated)


def simulate_hawkes_credit_basket(
    mu: Optional[Union[np.ndarray, Sequence[float], CalibratedHawkesParameters, Mapping[str, Any]]] = None,
    alpha: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    beta: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    horizon: float = 5.0,
    seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = None,
    n_grid_points: int = 2501,
    entity_names: Optional[Sequence[str]] = None,
    absorbing_defaults: bool = True,
    lgd: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> HawkesSimulationResult:
    """Simulate a single trajectory of the 5-entity Hawkes credit basket over [0, horizon]."""
    if isinstance(mu, (CalibratedHawkesParameters, Mapping)):
        simulator = MultivariateHawkesCreditSimulator(mu)
    elif mu is None and alpha is None and beta is None:
        simulator = create_default_5_entity_basket(absorbing_defaults=absorbing_defaults)
    elif mu is not None and alpha is not None and beta is not None:
        simulator = MultivariateHawkesCreditSimulator(
            mu=mu,
            alpha=alpha,
            beta=beta,
            entity_names=entity_names,
            absorbing_defaults=absorbing_defaults,
            lgd=lgd,
        )
    else:
        raise ValueError(
            "Provide a CalibratedHawkesParameters/dict, all three of (mu, alpha, beta), "
            "or leave all three as None."
        )

    return simulator.simulate(
        horizon=horizon,
        seed=seed,
        n_grid_points=n_grid_points,
    )


ARXIV_RC_PARAMS: dict[str, Any] = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_ACADEMIC_COLORS = (
    "#111111",
    "#004488",
    "#BB5566",
    "#228833",
    "#AA7700",
)

_ACADEMIC_LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1.5, 1, 1.5)),
)


def apply_arxiv_matplotlib_style() -> None:
    """Apply publication-style rcParams configuration to matplotlib."""
    plt.rcParams.update(ARXIV_RC_PARAMS)


def _format_arxiv_axes(ax: plt.Axes) -> None:
    """Format axes with white background, black spines on all four sides, inward ticks, and no gridlines."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        colors="black",
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)


def compute_reflected_kde(
    taus: np.ndarray,
    t_grid: np.ndarray,
    horizon: float = 5.0,
    bw_method: Optional[Union[str, float]] = "scott",
) -> np.ndarray:
    """Evaluate a double-boundary reflected Gaussian KDE on [0, horizon] scaled by 3.0."""
    clean_taus = np.asarray(taus, dtype=np.float64)
    clean_taus = clean_taus[
        np.isfinite(clean_taus) & (clean_taus >= 0.0) & (clean_taus <= float(horizon))
    ]
    t_eval = np.asarray(t_grid, dtype=np.float64)
    if clean_taus.size < 2:
        return np.zeros_like(t_eval)
    # Reflect sample across t = 0 and t = T to eliminate kernel boundary bias on [0, T]
    reflected = np.concatenate(
        [clean_taus, -clean_taus, 2.0 * float(horizon) - clean_taus]
    )
    kde = gaussian_kde(reflected, bw_method=bw_method)
    return 3.0 * kde(t_eval)


def find_cascading_default_monte_carlo_path(
    simulator: Optional[MultivariateHawkesCreditSimulator] = None,
    horizon: float = 5.0,
    start_seed: int = 1,
    max_attempts: int = 500,
) -> HawkesSimulationResult:
    """Find a representative Monte Carlo trajectory featuring a two-default intra-sector cascade."""
    sim = simulator if simulator is not None else create_default_5_entity_basket(absorbing_defaults=True)
    fallback_res: Optional[HawkesSimulationResult] = None

    for seed_val in range(int(start_seed), int(start_seed) + int(max_attempts)):
        candidate = sim.simulate(horizon=horizon, seed=seed_val, n_grid_points=801)
        if candidate.n_accepted == 2:
            t1 = float(candidate.event_times[0])
            t2 = float(candidate.event_times[1])
            e1 = int(candidate.event_entities[0])
            e2 = int(candidate.event_entities[1])
            if 0.6 <= t1 <= 2.0 and (t2 - t1) >= 0.35 and t2 <= 3.3 and float(sim.alpha[e2, e1]) > 0.30:
                return candidate
            if fallback_res is None:
                fallback_res = candidate

    if fallback_res is not None:
        return fallback_res
    return sim.simulate(horizon=horizon, seed=start_seed, n_grid_points=801)


def find_single_default_monte_carlo_path(
    simulator: Optional[MultivariateHawkesCreditSimulator] = None,
    horizon: float = 5.0,
    start_seed: int = 1,
    max_attempts: int = 500,
) -> HawkesSimulationResult:
    """Find a Monte Carlo trajectory with a single default event occurring in [0.7, 2.2] years."""
    sim = simulator if simulator is not None else create_default_5_entity_basket(absorbing_defaults=True)
    for seed_val in range(int(start_seed), int(start_seed) + int(max_attempts)):
        candidate = sim.simulate(horizon=horizon, seed=seed_val, n_grid_points=801)
        if candidate.n_accepted == 1:
            tau = float(candidate.event_times[0])
            ent_id = int(candidate.event_entities[0])
            if 0.7 <= tau <= 2.2 and float(np.max(sim.alpha[:, ent_id])) > 0.30:
                return candidate
    return sim.simulate(horizon=horizon, seed=start_seed, n_grid_points=801)


def _save_publication_figure(
    fig: plt.Figure,
    output_dir: Optional[Union[str, Path]],
    stem: str,
) -> Optional[Path]:
    if output_dir is None:
        return None
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    png_path = out_path / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    return png_path


def plot_single_path_hazard_step_chart(
    simulation_result: Optional[HawkesSimulationResult] = None,
    simulator: Optional[MultivariateHawkesCreditSimulator] = None,
    horizon: float = 5.0,
    output_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a cascading default hazard rate trajectory and optionally export fig1_cascading_hazard_path.png."""
    apply_arxiv_matplotlib_style()

    if simulation_result is None:
        simulation_result = find_cascading_default_monte_carlo_path(
            simulator=simulator,
            horizon=horizon,
        )

    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=300)
    _format_arxiv_axes(ax)

    n_entities = len(simulation_result.entity_names)
    entity_default_time = np.full(n_entities, np.inf, dtype=np.float64)
    entity_default_order = np.full(n_entities, -1, dtype=np.int64)
    for ev_idx in range(simulation_result.n_accepted):
        ent_id = int(simulation_result.event_entities[ev_idx])
        ev_time = float(simulation_result.event_times[ev_idx])
        if ev_time < entity_default_time[ent_id]:
            entity_default_time[ent_id] = ev_time
            entity_default_order[ent_id] = ev_idx + 1

    for idx in range(n_entities):
        entity_label = simulation_result.entity_names[idx]
        color = _ACADEMIC_COLORS[idx % len(_ACADEMIC_COLORS)]
        ls = _ACADEMIC_LINESTYLES[idx % len(_ACADEMIC_LINESTYLES)]
        tau_i = float(entity_default_time[idx])
        order_k = int(entity_default_order[idx])
        y_vals = simulation_result.intensities[:, idx]

        if np.isfinite(tau_i):
            mask = simulation_result.times <= tau_i
            ax.step(
                simulation_result.times[mask],
                y_vals[mask],
                where="post",
                color=color,
                linestyle=ls,
                linewidth=1.6,
                label=(
                    f"{entity_label} ("
                    + rf"$\lambda_{{{idx+1}}}(t)$, absorbed at $\tau_{{({order_k})}}={tau_i:.2f}\,\mathrm{{yr}}$)"
                ),
            )
            ev_pos = order_k - 1
            y_term = float(simulation_result.pre_jump_intensities[ev_pos, idx])
            ax.plot(
                [tau_i],
                [y_term],
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                zorder=6,
            )
        else:
            ax.step(
                simulation_result.times,
                y_vals,
                where="post",
                color=color,
                linestyle=ls,
                linewidth=1.55,
                label=f"{entity_label} (" + rf"$\lambda_{{{idx+1}}}(t)$, surviving)",
            )

    max_intensity = float(np.max(simulation_result.intensities))

    for ev_idx in range(simulation_result.n_accepted):
        tau_k = float(simulation_result.event_times[ev_idx])
        ax.axvline(
            x=tau_k,
            color="#333333" if ev_idx == 0 else "#666666",
            linestyle="--" if ev_idx == 0 else ":",
            linewidth=1.05,
            zorder=2,
        )

    if simulation_result.n_accepted > 0:
        tau_1 = float(simulation_result.event_times[0])
        j_1 = int(simulation_result.event_entities[0])
        post_0 = simulation_result.post_jump_intensities[0].copy()
        post_0[j_1] = -1.0
        target_i = int(np.argmax(post_0))
        jump_1 = float(simulation_result.alpha[target_i, j_1])
        y_peak_1 = float(simulation_result.post_jump_intensities[0, target_i])

        ax.annotate(
            rf"1st Default $\tau_{{(1)}} = {tau_1:.2f}\,\mathrm{{yr}}$ ($j^*={j_1+1}$)"
            "\n"
            rf"Jump $\Delta\lambda_{{{target_i+1}}} = \alpha_{{{target_i+1},{j_1+1}}} = {jump_1:.4f}\,\mathrm{{yr}}^{{-1}}$",
            xy=(tau_1, y_peak_1),
            xytext=(0.08, y_peak_1 * 1.04),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.85),
            fontsize=8.1,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="black", lw=0.7),
        )

        ax.text(
            0.10,
            float(np.max(simulation_result.mu)) + 0.032,
            r"Pre-FTD baseline: $\lambda_i(t) = \mu_i$",
            fontsize=8.0,
            color="#222222",
        )

    if simulation_result.n_accepted >= 2:
        tau_2 = float(simulation_result.event_times[1])
        j_2 = int(simulation_result.event_entities[1])
        y_abs_2 = float(simulation_result.pre_jump_intensities[1, j_2])
        ax.annotate(
            rf"2nd Default $\tau_{{(2)}} = {tau_2:.2f}\,\mathrm{{yr}}$ ($j^*={j_2+1}$)"
            "\n"
            r"Absorbing termination + secondary jumps",
            xy=(tau_2, y_abs_2),
            xytext=(tau_2 + 0.25, y_abs_2 * 0.95),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=0.85),
            fontsize=8.0,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="black", lw=0.7),
        )

    ax.set_xlim(0.0, float(simulation_result.horizon))
    ax.set_ylim(0.0, 1.25 * max_intensity)
    ax.set_xlabel(r"Simulation Time $t$ $(\mathrm{years})$")
    ax.set_ylabel(r"Instantaneous Hazard Rate $\lambda_i(t)$ $(\mathrm{yr}^{-1})$")
    ax.set_title(
        r"Cascading Default Trajectory with Contagion Jumps $\Delta\lambda_i = \alpha_{i,j^*}$ and Exponential Decay"
    )
    ax.legend(loc="upper right", frameon=True, fontsize=7.6)
    fig.tight_layout()

    _save_publication_figure(fig, output_dir, "fig1_cascading_hazard_path")
    return fig


def plot_contagion_matrix_heatmap(
    calibrated_params: Optional[Union[CalibratedHawkesParameters, MultivariateHawkesCreditSimulator]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot the calibrated contagion matrix alpha_{ij} and optionally export fig2_contagion_matrix.png."""
    apply_arxiv_matplotlib_style()

    if calibrated_params is None:
        calib = calibrate_hawkes_parameters()
        alpha = calib.alpha
        entity_names = calib.entity_names
        sectors = calib.sectors
    elif isinstance(calibrated_params, CalibratedHawkesParameters):
        alpha = calibrated_params.alpha
        entity_names = calibrated_params.entity_names
        sectors = calibrated_params.sectors
    else:
        alpha = calibrated_params.alpha
        entity_names = calibrated_params.entity_names
        default_md = generate_synthetic_market_data()
        sectors = (
            default_md.sectors
            if len(entity_names) == len(default_md.sectors)
            else ["Sector"] * len(entity_names)
        )

    n = alpha.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    _format_arxiv_axes(ax)

    im = ax.imshow(alpha, cmap="Blues", interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"Contagion Excitation $\alpha_{ij}$ $(\mathrm{yr}^{-1})$", fontsize=10.0)
    cbar.ax.tick_params(direction="in", width=0.8, length=3.5)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)

    tick_labels = [f"{name}\n({sector})" for name, sector in zip(entity_names, sectors)]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8.3)
    ax.set_yticklabels(tick_labels, fontsize=8.3)
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

    max_val = float(np.max(alpha)) if float(np.max(alpha)) > 0.0 else 1.0
    for row_i in range(n):
        for col_j in range(n):
            val = float(alpha[row_i, col_j])
            text_color = "white" if val > 0.50 * max_val else "black"
            if row_i == col_j:
                cell_label = f"{val:.4f}\n(Absorbing)"
            elif sectors[row_i] == sectors[col_j]:
                cell_label = f"{val:.4f}\n(Intra-Sector)"
            else:
                cell_label = f"{val:.4f}\n(Cross-Sector)"
            ax.text(
                col_j,
                row_i,
                cell_label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.9,
                fontweight="bold" if val > 0.50 * max_val else "normal",
            )

    ax.set_xlabel(r"Defaulting Source Entity $j$ (Triggering Default at $\tau^{(j)}$)")
    ax.set_ylabel(r"Impacted Target Entity $i$ (Jump $\Delta\lambda_i = \alpha_{ij}$)")
    ax.set_title(r"Calibrated Contagion Matrix $\alpha_{ij}$ ($\alpha_{ii} = 0$, $\rho(\alpha/\beta) = 0.68$)")
    fig.tight_layout()

    _save_publication_figure(fig, output_dir, "fig2_contagion_matrix")
    return fig


def plot_kth_default_densities_and_spreads(
    comparison_result: Optional[ContagionComparisonResult] = None,
    n_paths: int = 10_000,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot stopping time KDEs and k-th-to-Default spreads and optionally export fig3_kth_default_tranches.png."""
    apply_arxiv_matplotlib_style()

    if comparison_result is None:
        comparison_result = compare_hawkes_vs_independent_kth_default(
            n_paths=n_paths,
            seed=seed,
        )

    hawkes_res = comparison_result.hawkes_result
    base_res = comparison_result.baseline_result
    horizon = float(hawkes_res.horizon)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9.6, 3.8),
        dpi=300,
        gridspec_kw={"wspace": 0.25},
    )
    _format_arxiv_axes(ax1)
    _format_arxiv_axes(ax2)

    t_grid = np.linspace(0.0, horizon, 400, dtype=np.float64)
    tau1_hawkes = hawkes_res.realized_kth_default_times(1)
    tau2_hawkes = hawkes_res.realized_kth_default_times(2)
    tau2_base = base_res.realized_kth_default_times(2)

    kde_tau1 = compute_reflected_kde(tau1_hawkes, t_grid, horizon=horizon)
    kde_tau2_hawkes = compute_reflected_kde(tau2_hawkes, t_grid, horizon=horizon)
    kde_tau2_base = compute_reflected_kde(tau2_base, t_grid, horizon=horizon)

    ax1.plot(
        t_grid,
        kde_tau1,
        color="#111111",
        linestyle="-",
        linewidth=1.75,
        label=r"$\tau_{(1)}$ (FTD: Hawkes $\equiv$ Indep.)",
    )
    ax1.plot(
        t_grid,
        kde_tau2_hawkes,
        color="#004488",
        linestyle="--",
        linewidth=1.75,
        label=r"$\tau_{(2)}$ (2nd TD: Hawkes $\alpha_{ij} > 0$)",
    )
    ax1.plot(
        t_grid,
        kde_tau2_base,
        color="#BB5566",
        linestyle=":",
        linewidth=1.85,
        label=r"$\tau_{(2)}$ (2nd TD: Indep. $\alpha_{ij} = 0$)",
    )

    mean_tau1 = float(np.mean(tau1_hawkes))
    mean_tau2_h = float(np.mean(tau2_hawkes))
    mean_tau2_b = float(np.mean(tau2_base))
    max_density = float(max(np.max(kde_tau1), np.max(kde_tau2_hawkes), np.max(kde_tau2_base)))
    vline_top = max_density * 0.74
    ax1.vlines(mean_tau1, 0.0, vline_top, color="#111111", linestyle="-.", linewidth=0.9, alpha=0.80)
    ax1.vlines(mean_tau2_h, 0.0, vline_top, color="#004488", linestyle="-.", linewidth=0.9, alpha=0.80)
    ax1.vlines(mean_tau2_b, 0.0, vline_top, color="#BB5566", linestyle="-.", linewidth=0.9, alpha=0.80)

    ax1.set_xlim(0.0, horizon)
    ax1.set_ylim(0.0, max_density * 1.22)
    ax1.set_xlabel(r"Default Stopping Time $\tau_{(k)}$ $(\mathrm{years})$", fontsize=10.0)
    ax1.set_ylabel(r"Reflected Density $f(\tau_{(k)} \mid \tau_{(k)} \leq T)$", fontsize=10.0)
    ax1.set_title(r"(a) Stopping Time Densities $\tau_{(1)}$ and $\tau_{(2)}$", fontsize=10.5)
    ax1.legend(loc="upper right", fontsize=7.6)

    n_tranches = hawkes_res.kth_fair_spreads_bps.shape[0]
    k_indices = np.arange(1, n_tranches + 1, dtype=np.float64)
    bar_width = 0.36

    hawkes_bps = hawkes_res.kth_fair_spreads_bps
    base_bps = base_res.kth_fair_spreads_bps
    hawkes_se = hawkes_res.kth_spread_std_errors_bps
    base_se = base_res.kth_spread_std_errors_bps
    premiums = comparison_result.contagion_premium_bps

    ax2.bar(
        k_indices - bar_width / 2.0,
        hawkes_bps,
        width=bar_width,
        color="#1F4E78",
        edgecolor="black",
        linewidth=0.8,
        yerr=1.96 * hawkes_se,
        capsize=2.3,
        error_kw=dict(lw=0.75, ecolor="black"),
        label=r"Hawkes Contagion ($\alpha_{ij} > 0$)",
    )
    ax2.bar(
        k_indices + bar_width / 2.0,
        base_bps,
        width=bar_width,
        color="#D9E1F2",
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
        yerr=1.96 * base_se,
        capsize=2.3,
        error_kw=dict(lw=0.75, ecolor="black"),
        label=r"Independent Poisson ($\alpha_{ij} = 0$)",
    )

    for idx in range(n_tranches):
        top_y = float(max(hawkes_bps[idx] + 1.96 * hawkes_se[idx], base_bps[idx] + 1.96 * base_se[idx]))
        prem = float(premiums[idx])
        label_txt = "0.00 bps" if idx == 0 else f"+{prem:.1f} bps"
        ax2.text(
            float(k_indices[idx]),
            top_y + 42.0,
            label_txt,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
        )

    ax2.set_xticks(k_indices)
    ax2.set_xticklabels(
        ["1st TD\n(FTD)", "2nd TD", "3rd TD", "4th TD", "5th TD"],
        fontsize=8.5,
    )
    ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax2.set_ylim(0.0, float(np.max(hawkes_bps + 1.96 * hawkes_se)) * 1.24)
    ax2.set_xlabel(r"Basket Tranche Order $k$", fontsize=10.0)
    ax2.set_ylabel(r"Fair Swap Spread $S_k^*$ $(\mathrm{bps})$", fontsize=10.0)
    ax2.set_title(r"(b) Fair $k$-th-to-Default Spreads ($10,000$ CRN Paths)", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=7.8)

    _save_publication_figure(fig, output_dir, "fig3_kth_default_tranches")
    return fig


def plot_ftd_time_density(
    ftd_result: Optional[FTDSwapPricingResult] = None,
    n_paths: int = 10_000,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Backward-compatible wrapper returning Figure 3 (plot_kth_default_densities_and_spreads)."""
    return plot_kth_default_densities_and_spreads(
        n_paths=n_paths,
        seed=seed,
        output_dir=output_dir,
    )


def generate_hawkes_ftd_plots(
    output_dir: Optional[Union[str, Path]] = "figures",
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict[str, plt.Figure]:
    calibrated = calibrate_hawkes_parameters(risk_free_rate=0.04, absorbing_defaults=True)
    simulator = MultivariateHawkesCreditSimulator(**calibrated)

    cascading_path = find_cascading_default_monte_carlo_path(
        simulator=simulator,
        horizon=5.0,
        start_seed=1,
    )
    comparison = compare_hawkes_vs_independent_kth_default(
        simulator=simulator,
        n_paths=n_paths,
        horizon=5.0,
        risk_free_rate=0.04,
        seed=seed,
    )

    fig1 = plot_single_path_hazard_step_chart(
        simulation_result=cascading_path,
        output_dir=output_dir,
    )
    fig2 = plot_contagion_matrix_heatmap(
        calibrated_params=calibrated,
        output_dir=output_dir,
    )
    fig3 = plot_kth_default_densities_and_spreads(
        comparison_result=comparison,
        output_dir=output_dir,
    )

    return {
        "step_chart": fig1,
        "contagion_heatmap": fig2,
        "ftd_density": fig3,
    }


if __name__ == "__main__":
    market_data = generate_synthetic_market_data(risk_free_rate=0.04)
    calibrated = calibrate_hawkes_parameters(market_data, absorbing_defaults=True)
    simulator = MultivariateHawkesCreditSimulator(**calibrated)

    comparison = compare_hawkes_vs_independent_kth_default(
        simulator=simulator,
        n_paths=10_000,
        horizon=5.0,
        risk_free_rate=0.04,
        seed=42,
    )

    figures_dir = Path(__file__).resolve().parent / "figures"
    figs = generate_hawkes_ftd_plots(output_dir=figures_dir, n_paths=10_000, seed=42)
    plt.close("all")

    print("-" * 100)
    print("HAWKES CREDIT BASKET CALIBRATION & k-TH-TO-DEFAULT (k=1..5) MONTE CARLO PRICING (10,000 CRN PATHS)")
    print("-" * 100)
    print(f"Entities                 : {calibrated.entity_names}")
    print(f"Sectors                  : {calibrated.sectors}")
    print(f"5Y CDS Spreads (bps)     : {np.round(market_data.cds_spreads_bps, 1)}")
    print(f"Entity LGD               : {np.round(calibrated.lgd, 2)}")
    print(f"Absorbing Defaults       : {simulator.absorbing_defaults}")
    print(f"Calibrated Baseline mu_i : {np.round(calibrated.mu, 6)}")
    print("Calibrated alpha_ij (Zero Diagonal / Absorbing):")
    print(np.round(calibrated.alpha, 4))
    print(f"Spectral Radius          : {simulator.spectral_radius:.4f}")
    print("-" * 100)
    print("k-TH-TO-DEFAULT SWAP COMPARISON: HAWKES CONTAGION (alpha > 0) VS. INDEPENDENT POISSON (alpha = 0)")
    print(comparison.comparison_df.to_string(index=False))
    print("-" * 100)
