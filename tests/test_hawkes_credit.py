"""Automated verification suite for the multivariate Hawkes credit basket simulator and k-th-to-default pricer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import hawkes_credit as hc
from hawkes_credit import (
    CalibratedHawkesParameters,
    ContagionComparisonResult,
    CreditBasketMarketData,
    MultivariateHawkesCreditSimulator,
    calibrate_hawkes_parameters,
    compare_hawkes_vs_independent_kth_default,
    compute_reflected_kde,
    create_default_5_entity_basket,
    extract_baseline_hazard_rates_from_cds,
    generate_hawkes_ftd_plots,
    generate_synthetic_market_data,
    reprice_par_cds_spreads_bps,
)


def test_exact_crn_ftd_invariance_and_10k_benchmark() -> None:
    """Verify exact k=1 FTD spread invariance under CRN and benchmark contagion premiums across k=1..5."""
    comparison = compare_hawkes_vs_independent_kth_default(
        n_paths=10_000,
        horizon=5.0,
        risk_free_rate=0.04,
        seed=42,
    )
    assert isinstance(comparison, ContagionComparisonResult)

    hawkes_k1_spread = float(comparison.hawkes_result.kth_fair_spreads[0])
    indep_k1_spread = float(comparison.baseline_result.kth_fair_spreads[0])

    assert abs(hawkes_k1_spread - indep_k1_spread) < 1e-12
    assert np.allclose(
        comparison.hawkes_result.first_default_times,
        comparison.baseline_result.first_default_times,
        atol=1e-15,
        equal_nan=True,
    )
    assert np.array_equal(
        comparison.hawkes_result.first_default_entities,
        comparison.baseline_result.first_default_entities,
    )

    assert np.isclose(comparison.hawkes_result.fair_spread_bps, 1843.11, atol=0.01)
    assert np.isclose(comparison.baseline_result.fair_spread_bps, 1843.11, atol=0.01)
    expected_premiums_bps = np.array([0.00, 318.47, 218.61, 115.70, 23.78], dtype=np.float64)
    assert np.allclose(comparison.contagion_premium_bps, expected_premiums_bps, atol=0.01)


def test_zero_diagonal_self_excitation_and_spectral_radius() -> None:
    """Verify zero diagonal self-excitation in alpha and stationary spectral radius calibration."""
    calibrated = calibrate_hawkes_parameters(target_spectral_radius=0.68)
    assert isinstance(calibrated, CalibratedHawkesParameters)
    assert np.all(np.diag(calibrated.alpha) == 0.0)
    assert np.isclose(calibrated.spectral_radius, 0.68, atol=1e-12)

    simulator = MultivariateHawkesCreditSimulator(calibrated)
    assert np.all(np.diag(simulator.alpha) == 0.0)
    assert np.isclose(simulator.spectral_radius, 0.68, atol=1e-12)
    assert simulator.stationary_intensities is not None
    assert np.all(simulator.stationary_intensities > simulator.mu)


def test_absorbing_default_states() -> None:
    """Verify absorbing default states are enabled by default and prevent repeated entity defaults."""
    calibrated = calibrate_hawkes_parameters()
    assert calibrated.absorbing_defaults is True

    simulator = create_default_5_entity_basket()
    assert simulator.absorbing_defaults is True

    for seed_idx in range(1, 201):
        res = simulator.simulate(horizon=5.0, seed=seed_idx, include_timeseries=False)
        assert res.absorbing_defaults is True
        assert res.n_accepted <= simulator.n_entities
        unique_entities = np.unique(res.event_entities)
        assert unique_entities.size == res.event_entities.size


def test_single_name_cds_bootstrap_accuracy() -> None:
    """Verify bootstrapped baseline hazard rates reprice 5Y single-name par CDS spreads within 1e-10 bps."""
    market_data = generate_synthetic_market_data(risk_free_rate=0.04, maturity=5.0)
    assert isinstance(market_data, CreditBasketMarketData)
    expected_spreads_bps = np.array([320.0, 360.0, 420.0, 460.0, 290.0], dtype=np.float64)
    assert np.allclose(market_data.cds_spreads_bps, expected_spreads_bps, atol=1e-12)

    mu = extract_baseline_hazard_rates_from_cds(
        cds_spreads_5y=market_data.cds_spreads_5y,
        lgd=market_data.lgd,
        risk_free_rate=0.04,
        maturity=5.0,
        method="par_bootstrap",
    )
    repriced_bps = reprice_par_cds_spreads_bps(
        mu=mu,
        lgd=market_data.lgd,
        risk_free_rate=0.04,
        maturity=5.0,
    )
    assert np.allclose(repriced_bps, expected_spreads_bps, atol=1e-10, rtol=1e-12)


def test_positive_contagion_premiums_for_senior_tranches() -> None:
    """Verify Hawkes spreads strictly exceed independent baseline spreads for all senior tranches k >= 2."""
    comparison = compare_hawkes_vs_independent_kth_default(
        n_paths=2_000,
        horizon=5.0,
        risk_free_rate=0.04,
        seed=42,
    )
    hawkes_spreads = comparison.hawkes_result.kth_fair_spreads_bps
    indep_spreads = comparison.baseline_result.kth_fair_spreads_bps

    for k_idx in (1, 2, 3, 4):
        assert hawkes_spreads[k_idx] > indep_spreads[k_idx]
        assert comparison.contagion_premium_bps[k_idx] > 0.0

    assert np.all(np.diff(hawkes_spreads) < 0.0)
    assert np.all(np.diff(indep_spreads) < 0.0)


def test_boundary_reflected_kde_and_clean_300dpi_png_figures(tmp_path: Path) -> None:
    """Verify double-boundary reflected KDE scaling and export of 300 DPI PNG figures without Plotly."""
    rng = np.random.default_rng(42)
    sample_taus = rng.exponential(scale=1.5, size=2_000)
    sample_taus = sample_taus[sample_taus <= 5.0]
    t_grid = np.linspace(0.0, 5.0, 200)

    kde_vals = compute_reflected_kde(sample_taus, t_grid, horizon=5.0)
    reflected = np.concatenate([sample_taus, -sample_taus, 10.0 - sample_taus])
    expected_kde = 3.0 * gaussian_kde(reflected, bw_method="scott")(t_grid)
    assert np.allclose(kde_vals, expected_kde, atol=1e-14)
    assert kde_vals[0] > 0.25

    source_path = Path(hc.__file__).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    assert "plotly" not in source_text.lower()
    assert "write_html" not in source_text

    figs = generate_hawkes_ftd_plots(output_dir=tmp_path, n_paths=300, seed=42)
    for key in ("step_chart", "contagion_heatmap", "ftd_density"):
        assert isinstance(figs[key], matplotlib.figure.Figure)

    for stem in ("fig1_cascading_hazard_path", "fig2_contagion_matrix", "fig3_kth_default_tranches"):
        png_file = tmp_path / f"{stem}.png"
        assert png_file.exists() and png_file.stat().st_size > 10_000

    plt.close("all")
