# cheeks_hacksphere

# Lights On: Keep the Hostels Powered
**HackSphere · Aayam XI · Problem 2 · Hourly electricity forecasting for 34 hostel blocks**

## 1. Problem in one paragraph
Forecast hourly electricity use (`load_kwh`) for 34 hostel blocks from **5 Oct to 4 Dec 2025** (61 days). Weather and holidays are known for that period, the usage is not. Under-forecasting is penalised **3x** more than over-forecasting (metric: **BWE**, lower is better), because a too-low forecast means a power cut.

## 2. Our approach at a glance
| Decision | Why |
|---|---|
| **One XGBoost model** for all 34 blocks | Blocks share daily/weekly/weather patterns; the model learns each block's own shape through a building-ID feature. |
| **Quantile loss at 75%** (`reg:quantileerror`) | BWE is exactly a quantile loss: cost 3 for under, 1 for over. The best prediction is the 75th percentile, not the average. A normal (RMSE) model would under-forecast about half the time. |
| **Scale each block by its own mean** | BWE divides by each block's total usage, so small and large blocks count equally. We train on `load / block_mean` and multiply back at the end. |
| **Only a 364-day lag** | Our horizon is 61 days, so lags shorter than ~62 days are unknown at forecast time (data leakage). 364 days = 52 weeks (same weekday) is always known. 728+ days points before our data starts. |
| **Date-based time-series CV** | Always train on the past, validate on the next 61 days. Random splits would leak the future. |

## 3. Pipeline
```
Load + audit -> Clean bad meter rows -> Feature engineering -> Scale target per block
   -> 4-fold time-series CV -> Train XGBoost (quantile 0.75) -> Retrain on all data
   -> 3-seed average -> Rescale to kWh -> submission.csv + sanity checks
```

**Data audit.** The test set has 49,647 rows but a full grid would be 49,776, so some building-hours are missing. We therefore attach lags by **matching (building, timestamp)**, never by shifting rows.

**Cleaning.** We flag zero/negative readings, flat-lines of 24h or more, and spikes/dips versus a local 7-day median. Flagged rows are removed **from training only**. Normal holiday drops are real behaviour and are kept.

**Features (6 groups, about 50 columns)**
| Group | Examples | Idea |
|---|---|---|
| Calendar | hour, weekday, weekend, month, sin/cos of hour and day-of-year | Daily and weekly rhythm; 23:00 and 00:00 are close |
| Holiday | yesterday/tomorrow holiday, break length, position in break, days since/until holiday | Hostels empty out during breaks; the first and last days behave differently |
| Weather | raw weather, rolling temperature (3h to 7 days), humidity, heating/cooling degrees | Buildings react to weather slowly |
| Building | floor area, building age, building and campus ID | Each block has its own level and shape |
| Last year | load 364 days ago, plus 3h, daily and weekly smoothed versions | Captures seasonality and semester calendar |
| (no `year`) | removed | Trees cannot extrapolate a year number |

## 4. Model architecture: gradient-boosted trees (not a neural network)
We deliberately did **not** use a deep-learning network. This is tabular data (about 520k rows, about 50 columns) with a short history (639 days). On this kind of data, boosted trees are usually as accurate as or better than neural networks. They train fast, handle missing values natively (`cloud_cover_okta`, `year_built`), and are easy to explain.

**How XGBoost works here**
1. Tree 1 makes a rough guess of the (scaled) load.
2. Tree 2 is trained to fix the errors tree 1 made, tree 3 fixes what is still wrong, and so on, up to 3,000 trees.
3. Each tree is a set of yes/no questions ("holiday? after 8 PM? hotter than 30 C? this block?") ending in a number. The final prediction is the sum of all trees.
4. The "error" being fixed is the quantile loss, so the model learns to aim high on purpose.

**Key settings:** `max_depth=7`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.7`, early stopping after 100 trees without improvement (prevents memorising noise), native categorical handling for building/campus.

## 5. Validation
Four expanding-window folds, each forecasting **61 days** (the real horizon), cut by date. One fold (5 Oct to 4 Dec 2024) is the seasonal twin of the real test. It is expected to score worst because it has no previous year to look back on.

Diagnostics we use: simple baselines (profile mean/q75, "copy last year"), feature-group ablation, per-block and per-day error analysis, and a **calibration check** (actual should exceed prediction about 25% of the time).

## 6. Final prediction
Retrain on all clean data. Number of trees = median of CV best iterations x 1.15. Average 3 seeds, multiply by block scale, clip at 0, write `submission.csv` in `sample_submission` order, and assert row count, no NaNs and matching IDs. Visual checks: continuity with the last real days, and agreement with last year's seasonal shape.

## 7. Results
> Fill in after running on the real data. We did not have access to the full dataset while writing the code; the pipeline was tested end-to-end on synthetic data with the same schema.

| Item | Score |
|---|---|
| Baseline (profile q75) BWE | _TBD_ |
| XGBoost CV BWE (mean over 4 folds) | _TBD_ |
| Last-fold hold-out BWE | _TBD_ |

## 8. Limitations and next steps
- Cleaning thresholds and the 22 C / 15 C degree thresholds are educated guesses; tune on real data.
- If a block's consumption changed versus last year, the 364-day lag is biased. Next step: a year-over-year level correction, computed only from data before the forecast start.
- Possible gains: per-block calibration (only if folds agree), forward-looking weather windows, tuning alpha, ensembling LightGBM/CatBoost with the same quantile loss, Optuna tuning.
- Test weather is treated as a perfect forecast, as the problem allows; a real deployment would use forecasts.

## 9. How to run
```bash
pip install -U "xgboost>=2.0" pandas numpy matplotlib seaborn scikit-learn optuna openpyxl
# put train.csv, test.csv, sample_submission.csv next to the notebook (or set DATA_DIR)
jupyter notebook hostel_load_forecasting_xgboost.ipynb
```
Set `QUICK_RUN = True` for a fast smoke test, then `False` for the real run. Use `DEVICE = 'cuda'` on a GPU. Output: `submission.csv` and `model.json`.

**Repo contents:** `hostel_load_forecasting_xgboost.ipynb` (full code with an explanation for every block), `submission.csv`, `model.json`, `README.md`.

## 10. Credits
Dataset: adapted from the Building Data Genome Project 2 (Miller et al., Scientific Data 2020, CC BY-SA 4.0), provided by the DSAI Club, NIT Agartala. Notebook structure inspired by a public XGBoost time-series tutorial, adapted for multi-building data, an asymmetric metric and a 61-day horizon.
