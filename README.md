# A Financial Digital Twin for Dynamic Corporate Risk Simulation and Strategic Decision Intelligence

**Author:** Vedant Patil · [github.com/VedantPatil99](https://github.com/VedantPatil99)

A working Financial Digital Twin of Apple Inc., built entirely on **primary-source
data** — SEC EDGAR XBRL filings and Federal Reserve macro series. No synthetic
or simulated data anywhere in the pipeline.

---

## 1. What a digital twin has to do

A forecaster predicts one number. A twin is *coupled to* the thing it mirrors:
it holds a structural model of the firm, propagates shocks through that
structure, and closes the loop by measuring itself against reality.

This system implements seven capabilities:

| Capability | What it means | Where |
|---|---|---|
| **Mirror** | Balanced three-statement state from filings | `src/twin/statements.py` |
| **Interrogate** | Counterfactual shock paths over time | `src/twin/dynamics.py` |
| **Anticipate** | Walk-forward validated forecasting | `src/forecast/` |
| **Optimise** | Goal-seek and multi-lever search | `src/twin/decisions.py` |
| **Explain** | Sensitivity ranking across shocks and levers | `src/twin/decisions.py` |
| **Warn** | Percentile risk scoring, cash-runway detection | `src/simulate/risk.py` |
| **Learn** | Drift tracking, elasticity evolution, break detection | `src/twin/learning.py` |

---

## 2. Requirements & Environment Setup Guide

### 2.1 Prerequisites
- **Python**: Version **3.10** or higher is required. Verify your version:
  ```bash
  python --version   # or python3 --version
  ```
- **OS**: Windows, macOS, or Linux supported.

---

### 2.2 Virtual Environment (`venv`) Setup

Creating an isolated virtual environment ensures clean dependency isolation and prevents package conflicts.

#### **Windows (Command Prompt)**
```cmd
# Navigate to project directory
cd /d "path\to\fdt"

# Create virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate.bat
```

#### **Windows (PowerShell)**
```powershell
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

#### **macOS / Linux**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

When activated, your shell prompt will show `(.venv)`.

---

### 2.3 Installing Dependencies (`requirements.txt`)

Once your virtual environment is active:

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### **Core Dependencies included in `requirements.txt`:**
- `pandas >= 2.0` — Financial panel data structures & time-series operations
- `numpy >= 1.24` — Numerical computation & Monte Carlo simulations
- `scikit-learn >= 1.3` — Gradient boosting & regression models
- `scipy >= 1.10` — Elasticity optimization & distribution fitting
- `streamlit >= 1.30` — Interactive web dashboard platform
- `plotly >= 5.18` — Dynamic interactive charting & scenario plots
- `requests >= 2.31` — Rate-limited SEC EDGAR XBRL & FRED API clients

> **Note on PyTorch (`torch`)**: PyTorch is optional (~2.5 GB download). The project runs completely without it; the forecasting module automatically detects whether `torch` is available and safely skips the LSTM model if absent. To install PyTorch optionally:
> ```bash
> pip install torch
> ```

---

### 2.4 One-Click Automated Setup (Windows)

On Windows, you can skip manual terminal setup by using the provided batch files:

1. **Setup**: Double-click `setup.bat`  
   *(Automatically checks Python 3.10+, creates `.venv`, installs dependencies, and runs full pipeline validation).*
2. **Launch Dashboard**: Double-click `run_dashboard.bat`  
   *(Launches the Streamlit app on `http://localhost:8501`).*

---

### 2.5 Running the Pipeline & Dashboard

With the virtual environment activated (`.venv`):

```bash
# Run the pipeline & validation suite (~60s rebuild from cached snapshots)
python run_pipeline.py

# Re-download fresh live data from SEC EDGAR + FRED
python run_pipeline.py --live

# Launch the Streamlit interactive platform
streamlit run dashboard/app.py
```

To generalise the digital twin for another company (e.g. Microsoft):
```bash
python src/ingest/sec_client.py --ticker MSFT --fy-end-month 6
```

---

### 2.6 Running the Tests

```bash
python -m pytest tests          # 45 checks across two suites
```

`run_pipeline.py` runs the same two suites plus 23 data-validation and
scenario sanity checks (68 in total).

---

### 2.7 Deactivating the Virtual Environment

To exit the virtual environment when finished:
```bash
deactivate
```

---

## 3. Headline results

### 3.1 Historical replay — the twin reproduces shocks it never saw

The strongest validation in the project. The twin is re-fitted on **pre-episode
data only** (no lookahead), fed the conditions that actually occurred, and its
projected statements are compared against what Apple actually filed.

| Episode | Gross margin MAPE | Operating income MAPE |
|---|---|---|
| COVID-19 (2020) | **1.23%** | **3.10%** |
| 2018-19 China / trade tension | **0.33%** | **2.37%** |
| 2022 inflation shock | 5.83% | 24.44% |

This tests the **causal claim**, not forecast accuracy: given what happened to
the top line, does the rest of the financial picture follow? It does.

Cash is reproduced far less well (32–42%), and that is itself the finding:
cash error is dominated by **discretionary capital allocation**, not by the
causal mechanism. The twin predicts operating outcomes well and financing
outcomes poorly — because financing is a *choice*, not a consequence. Which is
precisely why the platform exposes levers.

### 3.2 The internal structure is stable; the macro linkage is not

Re-estimating the causal graph on an expanding 20-year window:

| Edge | Range | Std dev | Stable |
|---|---|---|---|
| revenue → COGS | 0.908 – 0.928 | **0.005** | yes |
| revenue → receivables | 0.604 – 0.725 | 0.037 | yes |
| COGS → payables | 0.915 – 1.078 | 0.052 | yes |
| **inflation → revenue** | **−1.43 – +9.43** | **4.217** | no |

Apple's operating relationships have barely moved in two decades. Its macro
relationships are unstable to the point of meaninglessness. This independently
justifies treating demand as a **user-injected shock node** rather than a
macro-derived forecast.

### 3.3 Risk is external; cash is self-inflicted

Sensitivity ranking over 8 quarters, shocks and levers on one scale:

| Metric | Top driver | Share of swing |
|---|---|---|
| Net income | supply cost shock | 26% |
| **Final cash** | **buyback intensity** | **59%** |

A 10% demand shock moves final cash by $2.1B. Changing buyback intensity moves
it by $157B. Apple's cash position is overwhelmingly determined by its own
capital-allocation choices — a conclusion a revenue forecaster could never reach.

### 3.4 The LSTM loses, and that is the result

Walk-forward backtest, 32 out-of-sample quarters, refit at every step:

| Model | Revenue MAPE | Skill vs baseline |
|---|---|---|
| **SeasonalMomentum** | **7.62%** | **+10.3%** |
| ResidualGBM | 7.84% | +7.8% |
| SeasonalNaive (baseline) | 8.50% | 0.0% |
| GradientBoost (level) | 9.34% | −9.9% |
| **LSTM** | 10.57% | **−24.5%** |

With ~70 quarters the LSTM has roughly 60 training points; it overfits however
small the network. Reported rather than hidden — a validated negative result is
stronger than an unvalidated claim that "deep learning was applied."

**Formulation beat the learner.** Targeting the log *level*, gradient boosting
scored −9.9% skill; targeting the *residual from seasonality* — a stationary
target — the same estimator reached +7.8%.

---

## 4. Architecture

```
Ingestion ──► Panel ──► Features ──► Causal graph ──┐
(SEC, FRED)                                          │
                                                     ▼
                         Three-statement engine ◄── Shock paths + levers
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        Trajectories          Decision studio         Replay + drift
     (Monte Carlo, VaR)   (goal-seek, tornado)     (validation, learning)
                                    │
                                    ▼
                            Streamlit platform
```

### The three-statement engine

The core addition. Earlier versions shocked the P&L and stopped at operating
income, which cannot answer the question a CFO actually asks — not "how far does
profit fall?" but "do we run out of money, and when?"

The balance sheet balances **by construction**, not by a plug:

```
ΔCash   = NI + D&A − ΔAR − ΔInv + ΔAP − Capex + ΔDebt − Div − Buyback
ΔPP&E   = Capex − D&A
ΔEquity = NI − Div − Buyback

⇒ ΔAssets = NI + ΔAP + ΔDebt − Div − Buyback = ΔLiabilities + ΔEquity
```

The two sides are identically equal, so if the opening sheet ties, every
projected one ties. Verified to **$0.000000000M over 12 quarters** under
combined extreme shocks.

Three feedback loops close the model:

1. **Cash → interest → net income → cash.** Non-operating income is interest on
   the net cash position, so a scenario that drains cash raises net interest
   cost, which drains it further.
2. **Capex → PP&E → D&A → operating income.** Depreciation is driven by the
   asset base, not by revenue. Without this, "cut capex" would look free.
3. **OCF → buybacks → equity.** Shareholder returns scale with cash generated,
   which is why Apple's equity shrinks while the business grows.

### Temporal dynamics

Shocks are **paths**, not steps: magnitude, persistence, onset delay and ramp.
Each causal edge carries a lag estimated by cross-correlation — macro reaches
revenue after ~3 quarters, revenue reaches COGS immediately. The lags are the
risk: a firm that can absorb an instantaneous hit can still fail if the trough
is deep and late.

The baseline is not a flat line. It reuses **the model that won the backtest**
(seasonal-momentum), so projected fiscal Q1 lands at $157.6B against Q3 at
$120.0B — matching Apple's real holiday seasonality — and the cost of a shock
includes the growth it forgoes.

---

## 5. The dataset

**75 quarters, 2006-09-30 → 2026-06-27**, from SEC EDGAR XBRL (CIK 0000320193),
plus 5 FRED macro series.

| Block | Series |
|---|---|
| Income statement | revenue, COGS, gross profit, opex, R&D, operating income, non-operating, pretax, tax, net income |
| Balance sheet | cash, receivables, inventory, PP&E, total assets, payables, current liabilities, long-term debt, equity |
| Cash flow | operating cash flow, capex, D&A, buybacks, dividends, debt issued/repaid |
| Macro | FEDFUNDS, CPIAUCSL, UNRATE, DGS10, GDPC1 |

### Validation

| Check | Result |
|---|---|
| `Assets = Liabilities + Equity` | PASS — $0.000M, 70 quarters |
| `Revenue − COGS = Gross Profit` | PASS — $0.000M, 67 quarters |
| FY2024 OCF / capex / buybacks / dividends | PASS — all match the 10-K |
| Projected balance sheet ties | PASS — $0.000000000M over 12 quarters |
| **Total pipeline checks** | **68 passing** |

Spot-checked against published figures: FY2024 revenue $391,035M, net income
$93,736M, pretax $123,485M, tax $29,749M, assets $364,980M, equity $56,950M,
OCF $118,254M, capex $9,447M, buybacks $94,949M, PP&E $45,680M — all exact.

### Five data problems this pipeline solves

Each silently corrupts naive XBRL pipelines. Each is handled explicitly:

1. **Q4 does not exist.** Filers never report Q4 as its own duration fact — it
   lives only inside the 10-K annual figure. Derived as `FY − (Q1+Q2+Q3)`, and
   only when all three are present. *Verified: derived Q4 FY2025 revenue =
   $102,466M, matching Apple's reported figure.*

2. **Cash flow is filed year-to-date.** A fiscal-Q3 10-Q shows *nine months* of
   operating cash flow. Treating YTD facts as quarterly overstates Q2 by ~2× and
   Q3 by ~3×. Recovered by differencing within each fiscal year, keyed on the
   fact's start date.

3. **Tag migration across ASC 606.** Revenue moves from `SalesRevenueNet` to
   `RevenueFromContractWithCustomerExcludingAssessedTax` in 2018. Spliced only
   after verifying the tags agree on overlapping periods (they match exactly at
   three quarters), giving one continuous 2008→2026 series.

4. **Restatement boundaries.** FY2009 Q4 mixes *restated* annual COGS with
   *originally-filed* quarterly gross profit (ASU 2009-13/14), producing a
   $1,849M inconsistency. COGS is reconstructed from the identity and the row is
   flagged, not silently patched.

5. **Duration vs instant facts.** Flow concepts carry `start`+`end`; balance
   sheet concepts carry only `end`. Applying a duration filter to a
   balance-sheet tag returns nothing at all.

> **A note on method.** An early extraction that passed SEC JSON through a
> summarising step returned corrupted values — annual totals mislabelled as
> quarterly ($391B as one quarter), duplicated periods, gaps. It was caught by
> checking a figure that was known independently. The final pipeline extracts
> deterministically and asserts losslessness. **Plausible-looking output is not
> evidence of correctness.**

---

## 6. The causal graph

Two edge types, and the distinction is the core design decision.

**Structural edges are accounting identities.** Gross profit *is* revenue minus
COGS. Coefficient 1, never estimated — fitting a tautology fits noise.

**Behavioural edges are empirical**, estimated from Apple's own filings as
elasticities on year-over-year log-differences with fiscal-quarter fixed
effects.

| From | To | Elasticity | t | Lag (Q) | Reading |
|---|---|---|---|---|---|
| revenue | COGS | **0.924** | 31.9 | 0 | operating leverage |
| COGS | payables | **1.031** | 8.8 | 0 | payables stretch faster than volume |
| revenue | receivables | **0.687** | 5.6 | 0 | collections scale sub-proportionally |
| COGS | inventory | **0.738** | 3.4 | 0 | inventory tracks cost of sales |
| revenue | opex | 0.200 | 2.1 | 3 | opex is largely fixed short-run |
| GDP | revenue | 1.356 | 1.2 | 3 | **not significant** |
| inflation | revenue | −1.418 | −1.1 | 3 | **not significant** |

Together these explain Apple's persistently **negative cash conversion cycle**
(−53 to −72 days): payables grow faster than receivables and inventory, so
growth *releases* cash rather than consuming it.

### Why the revenue equation is weak — and why it is reported

R² = 0.031, with GDP (t = 1.19) and inflation (t = −1.10) both insignificant.
Apple's revenue is driven by **product cycles, not US macro**. Forcing
significance here would be fitting noise. It is also the architectural
justification for `demand_shock` being a user-injected node.

> **Horizon matters.** Estimating on quarter-over-quarter changes gave R² = 0.005
> and an implausible COGS→inventory elasticity of 0.225 — fiscal-Q1 seasonality
> (~50% above Q3) swamped the signal and mismatched the YoY macro regressors.
> Moving to YoY differencing raised COGS→inventory to 0.738 and payables to 1.031.

### Two bugs the tests caught

Worth documenting, because both were silent:

**Unit mismatch at the shock boundary.** A +10% demand shock produced +10.52%
revenue: the shock was specified as a fractional change but applied as a
log-change, and `exp(0.10) − 1 = 10.52%`. Fixed by converting at the boundary
with `ln(1 + shock)`. The error compounded along every downstream edge.

**Zero-lag parents read from an empty history.** In the replay validator, a
zero-lag parent's current-period value was looked up in a history list that had
not been appended to yet — the lookup missed, the parent contributed 0, and
shocks never reached their children. Scenarios still ran and still balanced;
they just quietly did nothing. Operating-income replay error fell from 28–77%
to 2.4–24% once fixed. Both call sites now share one propagation function so
they cannot drift apart again.

---

## 7. Scenarios and risk

### Calibrated from observed history

Shock magnitudes come from the empirical distribution of the macro series, not
round numbers: the inflation scenario uses the actual CPI maximum (8.58%, the
2022 peak), the demand scenario the actual GDP trough (−7.4%, COVID Q2 2020).

The twin distinguishes shock *types*: a supply-chain shock hits **margin**
(50.1% → 42.4%) while demand contraction hits **volume** at near-intact margin.
A revenue forecaster cannot tell these apart.

Each scenario reports a full distribution with **VaR** and **CVaR** — a single
deterministic what-if gives one number with false precision.

### Risk scoring

Scores are **percentiles within Apple's own filing history**, not textbook
thresholds. A current ratio of 0.9 is alarming for a utility and completely
normal for Apple, which runs lean working capital by design; fixed industry
cut-offs would flag Apple as permanently distressed. 0 = healthiest this firm
has been, 100 = most stressed. Risk is always shown as **icon + label + value**,
never colour alone.

---

## 8. Project structure

```
fdt/
├── run_pipeline.py                  # one-command rebuild + validation
├── setup.bat / run_dashboard.bat    # Windows launchers
├── data/
│   ├── raw/                         # SEC XBRL + FRED snapshots, as extracted
│   └── cache/                       # built panels, backtests, replay, drift
├── src/
│   ├── build_panel.py               # panel assembly + identity validation
│   ├── features.py                  # twin state variables
│   ├── ingest/
│   │   ├── sec_client.py            # live SEC EDGAR client (any CIK)
│   │   ├── cashflow.py              # YTD → quarterly recovery
│   │   └── macro.py                 # FRED alignment
│   ├── graph/causal_graph.py        # elasticity estimation + propagation
│   ├── twin/
│   │   ├── statements.py            # three-statement engine
│   │   ├── dynamics.py              # lags, shock paths, trajectories
│   │   ├── decisions.py             # levers, tornado, goal-seek, optimiser
│   │   ├── replay.py                # historical replay validation
│   │   └── learning.py              # drift, elasticity evolution, breaks
│   ├── forecast/                    # 8 models + walk-forward backtest
│   └── simulate/                    # Monte Carlo, VaR/CVaR, risk scoring
├── dashboard/app.py                 # 10-tab Streamlit platform
├── docs/factor-catalog.html         # 38 candidate factors, with sourcing
└── tests/                           # 45 checks across two suites
```

---

## 9. Mapping to the project objectives

| Objective (from proposal) | Where it is met |
|---|---|
| 1. Comprehensive FDT architecture | Three-statement engine over a causal graph of 20 series |
| 2. AI-based predictive modelling | 8 models, walk-forward validated; `src/forecast/` |
| 3. Real-time risk identification | Percentile risk scoring + cash-runway detection |
| 4. Simulation-driven decision support | Shock paths, levers, goal-seek, optimiser, tornado |
| 5. Secure, scalable, reliable | Rate-limited SEC client, cached snapshots, 68 validation checks, generalises across filers |

**Technology Readiness Level: TRL 4** — validated in a lab. The system runs
end-to-end on real primary-source data, reproduces historical shocks
out-of-sample with quantified error, but has not been deployed against a live
corporate finance function.

---

## 10. Honest limitations

These strengthen a viva rather than weaken it:

- **Sample size.** ~70 quarters is a lot of financial history and very little
  data. This is why the LSTM underperforms.
- **Single firm.** Elasticities are Apple-specific. `sec_client.py` generalises
  to any CIK, but coefficients must be re-estimated per firm.
- **Macro has weak explanatory power** for this firm's revenue (§6).
- **Capital allocation is discretionary.** Modelled as fixed ratios of cash flow;
  Apple's actual buyback timing is a judgement call. This is the dominant source
  of cash-replay error, and the reason those ratios are exposed as levers.
- **Linear propagation.** Elasticities are constant, so very large shocks (>50%)
  extrapolate beyond the estimation range.
- **Shock priors.** `demand_shock`, `fx_shock`, `supply_cost_shock` transmission
  coefficients are documented structural priors (Apple books ~60% of revenue
  abroad, hence the 0.6 FX elasticity), not fitted values.
- **No segment split.** Products vs Services mix is the single highest-value
  missing factor — services margin runs roughly double products, so a mix shift
  moves consolidated margin with no change in revenue. SEC segment dimensions
  are not exposed by the `companyconcept` API and require parsing filing
  instance documents. See `docs/factor-catalog.html`.

---

## 11. Data sources

- **SEC EDGAR XBRL company facts** — `https://data.sec.gov/api/xbrl/companyconcept/`
  (Apple Inc., CIK 0000320193). Public domain.
- **Federal Reserve Economic Data (FRED)**, St. Louis Fed — `FEDFUNDS`,
  `CPIAUCSL`, `UNRATE`, `DGS10`, `GDPC1`.
- Scenario design informed by the Federal Reserve's supervisory stress-test
  variable set and Gartner's Digital Twin of an Organization market definition
  (see `docs/factor-catalog.html` for full sourcing).
