# Additional analyses for the IEEE paper

These scripts add the checks a reviewer asked for. They import the project's
modules unchanged and do not modify anything in `src/` or `data/`.

```bash
pip install statsmodels torch      # statsmodels for HAC errors; torch for the LSTM rows
python run_pipeline.py             # rebuild caches (backtests include the LSTM only if torch is installed)
python analysis/reviewer_checks.py # writes analysis/out/*.csv and summary.json
python analysis/make_figures.py    # writes analysis/out/fig_dag.png, fig_replay.png, fig_mc.png
```

What each output holds:

| File | Content |
|---|---|
| `hac_elasticities.csv` | Every behavioral edge with OLS and Newey-West (4 lags) standard errors |
| `forecast_dm_mase.csv` | MAPE, MASE and Diebold-Mariano tests (HLN correction) vs. SeasonalNaive |
| `controlled_target_test.csv` | Level vs. residual target with identical gradient-boosting settings |
| `replay_benchmarks.csv` | Replay errors for the twin, a no-shock run, and a constant-margin / flat-cash rule |
| `plug_ablation.csv` | Percent-of-sales model with a cash plug vs. its own cash-flow statement; size of the twin's offsetting entry |
| `gross_profit_ablation.csv` | Gross profit estimated as its own edge vs. revenue minus COGS |
| `decision_tornado_cash_min_stagflation.csv` | Tornado on the lowest cash balance under stagflation |
| `summary.json` | Goal-seek result, estimated lags, opening residual line items, gross-profit edge fit |
| `run_log.txt` | Console output of the last run |

The cached backtests committed before these scripts were added were produced
without PyTorch and have no LSTM rows. The paper's LSTM figures come from a
rebuild with PyTorch installed (2.14.1, CPU), with all other cached results
matching the committed cache.
