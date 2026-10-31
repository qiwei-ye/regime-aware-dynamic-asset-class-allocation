# Regime-Aware Dynamic Asset-Class Allocation

A reproducible weekly research pipeline that uses causal regime probabilities and pooled return forecasts to allocate across six asset-class proxies, then compares simulated holdings with equal-weight and inverse-volatility benchmarks. It records validation selection, one locked assessment, execution accounting, and comparative figures.

## Data and sources

| Input | Source and interpretation |
| --- | --- |
| Market histories | Yahoo Finance chart JSON, requested with `requests`: `VT` global equities, `IEF` government bonds, `LQD` investment-grade credit, `VNQ` real estate, `DBC` commodities, and `BIL` cash proxy. Daily distribution- and split-adjusted closes support constructed weekly returns and execution prices. |
| Equity stress | Chicago Board Options Exchange, [CBOE Volatility Index: VIX (`VIXCLS`)](https://fred.stlouisfed.org/series/VIXCLS), daily closing index, retrieved through FRED/ALFRED, Federal Reserve Bank of St. Louis. |
| Yield-curve stress | Federal Reserve Bank of St. Louis, [10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (`T10Y2Y`)](https://fred.stlouisfed.org/series/T10Y2Y), daily percentage-point spread, retrieved through FRED/ALFRED. |
| Producer-price pressure | U.S. Bureau of Labor Statistics, [Producer Price Index by Commodity: All Commodities (`PPIACO`)](https://fred.stlouisfed.org/series/PPIACO), monthly index, 1982 = 100, retrieved through FRED/ALFRED. |

The sample contains 624 consecutive seven-day research weeks beginning 28 July 2014. Prices cover 3,028 required sessions per asset, from the 25 July 2014 initial return boundary through 10 August 2026, which completes the final four-week target. Macro coverage includes four preceding weekly cutoffs. No required price sessions or macro cutoffs are missing.

[The retrieval manifest](results/retrieval_manifest.json) records 32 exact response snapshots, source identifiers, request parameters, retrieval times, release IDs, vintage coverage, hashes, and reconstruction instructions. Original responses and prepared prices, features, and targets remain in ignored `data/`; published forecasts, probabilities, allocations, and portfolio returns are constructed research outputs. Current Yahoo adjustment history is snapshotted, rather than reconstructed from historical market-data vintages.

ALFRED supplies historical vintages but not the required intraday release/ingestion timestamps. An observation or revision becomes eligible at the next midnight in `America/New_York` after its vintage date. Values are carried forward only inside their eligible vintage intervals; unavailable observations are never filled from later revisions. See [ALFRED's vintage and release-date explanation](https://alfred.stlouisfed.org/help).

Source rights remain with their owners. VIX and the term-spread series require attribution; PPIACO is identified as public-domain data with citation requested on its source page. [FRED API access does not override series-owner restrictions](https://fred.stlouisfed.org/docs/api/terms_of_use.html). [Yahoo's general terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection and reuse; its [Finance terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/finance/index.html) also apply. The repository grants no rights to redistribute provider observations or to refresh them without the applicable permissions. Terms reviewed on 31 October 2026 are recorded in the manifest; Yahoo terms responses are preserved privately.

Software attribution: pandas, NumPy, scikit-learn, hmmlearn, CVXPY with CLARABEL, joblib, Matplotlib, requests, and exchange-calendars. The pinned environment also includes the specified yfinance and fredapi packages; actual retrieval uses direct HTTP requests, and price adjustment follows the adjusted-close/raw-close ratio used by yfinance.

## Methodology

### Calendar, predictors, and targets

Research weeks 1–52 provide warm-up history; 53–312 are initial training; 313–520 are walk-forward validation; 521–624 are locked testing. Friday's actual exchange close is the decision cutoff. A Friday holiday retains a 16:00 New York cutoff and uses the last preceding close. Monday execution moves to the next trading close when necessary. The XNYS schedule handles holidays, early closes, daylight saving time, and extraordinary closures.

For adjusted close $P^a_{c_t}$ at resolved weekly close $c_t$, total return is $r^a_t=P^a_{c_t}/P^a_{c_{t-1}}-1$, and weekly excess return is $e^a_t=r^a_t-r^{BIL}_t$. Predictors comprise $e^a_{t-1}$, trailing 4-, 13-, and 52-week compounded excess returns $\prod(1+e^a)-1$, and sample standard deviations over those windows. Trailing windows include the current known Friday return. All three macro series contribute available levels and four-week changes: level differences for VIX and the spread, and fractional percentage change for PPI.

The supervised target is the asset's adjusted total return from actual execution close $E_t$ to $H_t$, less the BIL return over the same endpoints:

$$y^a_t=(P^a_{H_t}/P^a_{E_t}-1)-(P^{BIL}_{H_t}/P^{BIL}_{E_t}-1).$$

$H_t$ is four calendar weeks after the actual execution date, advanced to the next session if closed. BIL's excess forecast is fixed at zero. Each weekly supervised refit considers the preceding 260 calendar-origin weeks, starting no earlier than week 53, and retains labels only when $H_i$ is at or before the current decision cutoff. Earlier locked-test labels enter subsequent scheduled refits only after maturity.

### Models, selection, and allocation

Three two-state univariate Gaussian HMMs use VIX level, term-spread level, and four-week fractional PPI change. Each origin fits its available history, capped at 260 weeks, with training-window mean/population-standard-deviation scaling. Initialization starts after eight observations. Diagonal variance is floored at 0.001 after every M step; `covars_prior=0`, `n_iter=200`, and `tol=0.001`. Unusable fits are exposed and retried with at most 20 named seeds. One week-17 warm-up initialization failed and was resolved by its next seed; all 312 locked-period HMM fits converged.

Forward recursion computes $q_t(s)\propto p(z_t\mid s)\sum_j A_{js}q_{t-1}(j)$, with normalization at every observation. State columns are ordered by ascending emission mean. Each historical feature row retains its own origin-time fit and filtered probability; there is no backward smoothing or later-data refit of historical rows.

A pooled `HistGradientBoostingRegressor` forecasts the five non-cash assets using the predictors, six regime probabilities, and a categorical asset identifier. Loss is squared error, with early stopping disabled. Every Cartesian combination below was fitted at all 208 validation origins: 13,750 candidates and 2,860,000 fit calls. Within an otherwise identical parameter group, the five iteration checkpoints share boosting trees; equivalence to independent fits is tested. Return settings minimize pooled validation MSE; the 25 allocation combinations then maximize validation weekly excess-return Sharpe. Exact ties select the lowest permanent Cartesian candidate ID.

| Setting | Validation grid | Frozen choice |
| --- | --- | --- |
| `learning_rate` | 0.01–0.10, step 0.01 | 0.01 |
| `max_iter` | 100–500, step 100 | 100 |
| `max_leaf_nodes` | 5–15, step 1 | 5 |
| `min_samples_leaf` | 5–25, step 5 | 25 |
| `l2_regularization` | $10^{-1},10^{-2},10^{-3},10^{-4},10^{-5}$ | $10^{-5}$ |
| Risk aversion $\lambda$ | 1–5, step 1 | 3 |
| Turnover penalty $\gamma$ | 0.001–0.005, step 0.001 | 0.005 |

Across all six assets, Ledoit–Wolf estimates covariance from the latest 260 available weekly total returns, including the current Friday; $\Sigma_{4,t}=4\Sigma_{weekly,t}$. CVXPY/CLARABEL solves

$$\max_w\ \mu_t^\top w-\frac{\lambda}{2}w^\top\Sigma_{4,t}w-\gamma\lVert w-w^{Friday}_t\rVert_1,\qquad \sum_a w_a=1,\quad 0.05\leq w_a\leq0.50.$$

$w^{Friday}_t$ is the existing holdings drifted to the decision cutoff; initial holdings are equal-weight. Targets are implemented only at the subsequent execution close. Adjusted-price units represent distribution-reinvested holdings, which drift between trades. Weekly portfolio returns span execution to next execution, rather than treating overlapping four-week forecasts as separate trades. Realized one-way turnover is $\frac12\lVert w^{executed}-w^{pretrade}\rVert_1$ at execution. The optimization penalty is not charged as a cost; transaction costs, slippage, and taxes are zero.

Equal weight rebalances to six equal targets. Inverse volatility uses 52 available Friday total returns and sample standard deviations, normalizes reciprocal volatilities, then applies Euclidean projection onto the same bounded simplex. Both benchmarks use the strategy's execution schedule. BIL is also shown as a reference series.

`MASTER_SEED=42`. `SeedSequence` children use fixed component, forecast-origin, candidate, and repeated-fit IDs, independent of call order. The frozen configuration preserves their reconstruction rule, validation HMM attempts, all grids, selected settings, code/dependency hashes, and input provenance. The original frozen file remains unchanged; the separate completed-test receipt seals the assessment outputs.

## Results

Validation settings were frozen before the first locked assessment. Selection excludes four-week labels from origins 517–520 because they mature beyond the first locked cutoff, 19 July 2024. Validation forecast MSE therefore uses 204 origins and 1,020 non-cash targets. The final validation holding interval, origin 520, ends beyond that cutoff and is also excluded from performance scoring. Its executed holdings still carry into the locked period. Locked testing scores all 104 forecasts and 104 weekly holding periods; final target prices extend to 10 August 2026.

| Evaluation | Scored weekly return interval | Weekly returns | Non-cash forecast MSE |
| --- | --- | ---: | ---: |
| Validation | 27 July 2020–15 July 2024 | 207 | 0.001645741 |
| Locked test | 22 July 2024–20 July 2026 | 104 | 0.001226289 |

MSE uses squared return fractions. Annualized return is $[\prod_{t=1}^n(1+R_t)]^{52/n}-1$; annualized volatility is $\sqrt{52}\,s(R)$; Sharpe is $\sqrt{52}\,\overline{(R-r^{BIL})}/s(R-r^{BIL})$, with sample standard deviations. Maximum drawdown includes initial wealth one. Turnover below is the arithmetic mean of weekly one-way turnover.

| Period | Portfolio | Ann. return | Ann. volatility | Excess Sharpe | Max. drawdown | Mean turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | Strategy | 15.86% | 11.93% | 1.10 | −11.96% | 3.01% |
| Validation | Equal weight | 5.63% | 8.28% | 0.43 | −13.54% | 0.54% |
| Validation | Inverse volatility | 4.15% | 4.87% | 0.40 | −8.02% | 0.58% |
| Validation | BIL cash | 2.26% | 0.34% | — | −0.15% | 0.00% |
| Locked test | Strategy | 15.52% | 6.19% | 1.70 | −3.81% | 1.66% |
| Locked test | Equal weight | 9.91% | 6.22% | 0.89 | −5.28% | 0.51% |
| Locked test | Inverse volatility | 7.50% | 3.65% | 0.88 | −2.92% | 0.37% |
| Locked test | BIL cash | 4.17% | 0.11% | — | 0.00% | 0.00% |

BIL's excess Sharpe is undefined because its excess return over itself is identically zero. On this locked period the strategy has higher gross return and Sharpe than both rebalanced benchmarks, but a larger drawdown than inverse volatility. One historical period does not establish persistent or net outperformance.

![Comparative cumulative performance, with validation and locked testing rebased separately](results/figures/performance.png)

The panels start at wealth one independently; they do not bridge the excluded boundary holding period. The strategy's measured drawdown and volatility also differ materially between evaluation periods.

![Executed strategy weights across validation and locked testing](results/figures/allocations.png)

The allocation chart shows executed targets held as steps for display, not intraperiod drifted weights. The dashed line marks the first locked decision. The strategy frequently holds substantial BIL exposure and changes its mix across the sample; statistical regimes do not establish economic causes.

### Acceptance verification

| Brief criterion | Verified evidence |
| --- | --- |
| Calendar and causal availability | All 624 research weeks, required session endpoints, vintage intervals, and mature training labels checked; 55 correctness tests pass. |
| Exhaustive selection and frozen choices | All 13,750 return-model and 25 allocation combinations recorded with scored coverage; the original frozen seal and selected settings are unchanged. |
| Feasible holdings and accounting | All 936 strategy/benchmark executions satisfy the bounds and investment constraint; pre-trade holdings, NAV, weekly returns, and half-L1 turnover reconcile. |
| Benchmarks and separate evaluation | Equal-weight and inverse-volatility benchmarks, cash reference, forecast MSE, and every required portfolio measure are present for both periods; boundary exclusions remain explicit. |
| Clean reproduction and deliverables | From recorded raw snapshots, an isolated replay rebuilt validation under frozen choices and ran `python run_pipeline.py` without a locked-test cache or receipt. All six scientific CSVs and both figures are byte-identical; see the [verification record](artifacts/reproduction_verification.json). |

## Setup and reproduction

Tested on Windows x86-64, CPython 3.12.14, with the complete pinned environment in [requirements.txt](requirements.txt). Core versions include NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, hmmlearn 0.3.3, CVXPY 1.9.3, CLARABEL 0.11.1, joblib 1.6.0, and Matplotlib 3.11.2. Reuse the existing `.venv`; for a fresh setup:

```powershell
py -3.12 -m venv .venv
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pip check
```

Supply the credential `FRED_API_KEY` through the process environment before data preparation. No credential value or `.env` file is distributed. The two thread-count variables above are reproducibility settings, not credentials.

Exact research reproduction requires the manifest-listed private raw snapshots and adjacent cache metadata restored under `data/raw/`. The data stage reconstructs the seven prepared tables and verifies cached responses. Without snapshots it attempts source retrieval, subject to provider access permissions; later provider revisions may change bytes, so this does not guarantee the frozen experiment. A public clone can inspect all published results and regenerate figures from the saved CSVs without the observation cache.

The research entry point is `run_pipeline.py`. These research and verification commands have been tested in the recorded environment:

```powershell
python run_pipeline.py --stage data
python run_pipeline.py --stage select
python run_pipeline.py --stage test
python run_pipeline.py
python -c "from regime_allocation.reporting import export_figures; export_figures()"
python -m unittest discover -s tests
python -c "from regime_allocation.reproduction import verify_reproduction; verify_reproduction()"
```

With the frozen artifact present, selection verifies and preserves its choices. The first test invocation performs 104 scheduled return refits using only mature labels, appends separately identified outputs, and writes a completion receipt. Later invocations, including the default full workflow, verify and reuse the completed assessment; they do not conduct another test or retune. Figures are a separate rendering of sealed tables and do not modify research code or choices. The retrieval manifest describes input preparation; the authoritative assessment state is `artifacts/locked_test_receipt.json` (`complete`).

The final command performs a clean computational replay rather than a cache check. It temporarily preserves the completed outputs, then creates an isolated workspace under ignored `data/`, copying raw snapshots, source code, frozen choices, and the original validation grid scores. No prepared tables, fitted-model caches, or test receipt are copied. It rebuilds the 208 selected validation fits and origin-time HMMs, verifies the frozen validation handoff, then executes `python run_pipeline.py` for the fresh 104-origin locked replay. It does not repeat grid search or use locked outcomes to revise choices. The original assessment and outputs remain untouched, and the temporary workspace is removed after verification.

The [clean-reproduction record](artifacts/reproduction_verification.json) includes input/code hashes, the pinned environment, fit counts, and per-file comparisons. Identifiers, timestamps, unavailable-value patterns, and table structure must agree exactly; floating-point differences must be at most $10^{-10}$ absolute. This run reproduced every scientific table and figure byte for byte, with zero observed numeric difference. Retrieval-manifest generation times and the isolated replay's completion timestamp are administrative and are not compared as scientific results.

CSV exports retain 17 significant digits. SHA-256 checks protect recorded input and result bytes; source-text hashes normalize line endings, and the frozen JSON uses a canonical seal. Numerical tolerances are $10^{-7}$ for weight bounds/sums, $10^{-10}$ absolute for self-financing NAV, and $10^{-12}$ for covariance symmetry/PSD. CLARABEL feasibility and gap tolerances are $10^{-10}$, with 500 iterations maximum. HMM convergence requires a final likelihood improvement below 0.001, allowing negative roundoff only down to $-10^{-7}$. Solver failures and unusable HMM fits stop reporting rather than substitute allocations or data.

## Repository contents

| Location | Contents |
| --- | --- |
| [README.md](README.md) | Methodology, results, attribution, and reproduction. |
| [.gitignore](.gitignore), [.gitattributes](.gitattributes) | Private-data exclusions and reproducible text line endings. |
| [requirements.txt](requirements.txt) | Tested pinned direct and transitive dependencies. |
| [run_pipeline.py](run_pipeline.py) | Single research execution entry point. |
| [Package initialization](regime_allocation/__init__.py), [configuration](regime_allocation/config.py) | Shared paths, calendar, settings, grids, and named seeds. |
| [Data retrieval](regime_allocation/data.py), [features](regime_allocation/features.py) | Snapshots, vintage availability, adjusted returns, predictors, and targets. |
| [Models](regime_allocation/models.py), [portfolio](regime_allocation/portfolio.py) | Origin-time HMM features, pooled forecasts, exhaustive selection, optimization, and holdings. |
| [Evaluation](regime_allocation/evaluation.py), [reporting](regime_allocation/reporting.py) | Separate period scoring, frozen/test integrity, and figures from sealed outputs. |
| [Reproduction verifier](regime_allocation/reproduction.py) | Isolated cold replay of the frozen model and allocator, with preserved-output comparisons. |
| [tests/test_project.py](tests/test_project.py) | Independent mathematical, causal, timing, accounting, seed, and rendering-data checks. |
| [Frozen configuration](artifacts/frozen_config.json) | Selected settings, grids, seed mappings, source/code provenance, and validation hashes. |
| [Locked-test receipt](artifacts/locked_test_receipt.json) | Completion state, 104 return fits, frozen seal, and final table hashes. |
| [Reproduction verification](artifacts/reproduction_verification.json) | Clean-run environment, input provenance, preserved assessment, and exact/numeric comparisons. |
| [Retrieval manifest](results/retrieval_manifest.json) | Input coverage, source terms/attribution, exact snapshot hashes, and reconstruction. |
| [Selection](results/selection.csv) | Scores, coverage, and choices for all 13,775 grid combinations. |
| [Weekly forecasts](results/weekly_forecasts.csv) | Forecasts, realized targets, training maturity, and decision/execution/realization times. |
| [Regime probabilities](results/regime_probabilities.csv) | Filtered probabilities, ordered means, fit windows, seeds, and convergence diagnostics. |
| [Weekly allocations](results/weekly_allocations.csv) | Strategy/benchmark targets, reference/pre-trade/executed weights, units, NAV, turnover, and solver diagnostics. |
| [Weekly returns](results/weekly_returns.csv), [performance](results/performance.csv) | Separate-period strategy, benchmark, and cash returns and comparative measures. |
| [Performance figure](results/figures/performance.png), [allocation figure](results/figures/allocations.png) | Cumulative performance and executed strategy weights across both evaluation periods. |

## Assumptions and limitations

- Four-week labels overlap. MSE is descriptive and does not treat those observations as independent evidence; portfolio performance uses weekly holdings returns.
- Hyperparameters were selected from a large validation grid. Locked results cover only one subsequent historical sample, with no further selection based on them.
- BIL is an ETF proxy, not a guaranteed risk-free instrument. Gross results omit commissions, spread, market impact, financing details, taxes, and execution uncertainty.
- Four-times-weekly covariance approximates horizon risk. Gaussian two-state models and 260-week estimation windows are modeling assumptions, not established economic mechanisms.
- Historical market-data revision vintages and exact macro intraday release times are unavailable. Exact snapshots and the conservative macro availability rule improve traceability but cannot remove those source limitations.
- Bounds apply at execution; holdings can drift outside them between trades. The performance comparison omits the boundary-crossing validation interval and rebases each evaluation period separately.
- Exact fresh reconstruction depends on access to the original private inputs and pinned numerical environment. Provider observations are not redistributed, and new downloads may not match recorded hashes.
