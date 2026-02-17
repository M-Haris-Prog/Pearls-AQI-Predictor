# Plan: Pearls AQI Predictor — Lahore 3-Day Forecast

**TL;DR:** Build a serverless, end-to-end AQI prediction system for Lahore using OpenWeatherMap's Air Pollution + Weather APIs, Hopsworks as the Feature Store & Model Registry, multiple ML models (Ridge, Random Forest, XGBoost, LSTM), SHAP explainability, automated pipelines via GitHub Actions, and a Streamlit Cloud dashboard with Flask API. The workspace is empty — we build from scratch.

---

## Steps

### Phase 1 — Project Scaffolding & Environment

1. Create the following directory structure inside `c:\Users\haris\OneDrive\Desktop\AQI`:
   ```
   AQI/
   ├── .github/workflows/          # CI/CD
   ├── notebooks/                   # EDA & experiments
   ├── src/
   │   ├── feature_pipeline/       # Data ingestion + feature eng
   │   ├── training_pipeline/      # Model training + evaluation
   │   │   └── models/             # Individual model modules
   │   ├── inference/              # 3-day prediction
   │   └── explainability/         # SHAP / LIME
   ├── app/                        # Streamlit + Flask
   ├── tests/
   ├── data/raw/  data/processed/  # .gitignored local cache
   └── models/                     # .gitignored serialized models
   ```
2. Create `requirements.txt` with: `scikit-learn`, `tensorflow`, `xgboost`, `hopsworks`, `streamlit`, `flask`, `shap`, `lime`, `requests`, `pandas`, `numpy`, `plotly`, `python-dotenv`, `apscheduler`, `pytest`
3. Create `.env.example` with placeholders: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`
4. Create `.gitignore` — ignore `data/`, `models/`, `.env`, `__pycache__/`, `*.h5`
5. Initialize Git repo, create GitHub repository

---

### Phase 2 — Data Ingestion (Feature Pipeline)

6. Create `src/feature_pipeline/fetch_data.py` — OpenWeatherMap client:
   - **Air Pollution API** (`/air_pollution`): fetch PM2.5, PM10, NO₂, SO₂, CO, O₃ for Lahore (lat=31.5204, lon=74.3587)
   - **Weather API** (`/weather` or `/forecast`): fetch temperature, humidity, wind speed, pressure, visibility
   - Handle API rate limits with retry logic + exponential backoff
   - Return structured `pandas.DataFrame`

7. Create `src/feature_pipeline/feature_engineering.py`:
   - **Time features:** hour, day_of_week, month, is_weekend, season
   - **Derived features:** AQI change rate (delta from previous reading), rolling averages (6h, 12h, 24h), pollutant ratios (PM2.5/PM10)
   - **Lag features:** AQI at t-1, t-3, t-6, t-12, t-24 hours
   - **AQI calculation:** Convert raw pollutant concentrations to AQI using EPA breakpoint formula

8. Create `src/feature_pipeline/feature_store.py`:
   - Connect to Hopsworks via `hopsworks.login()`
   - Create/get Feature Group `lahore_aqi_features` (version 1)
   - `insert_features(df)` — upsert to Hopsworks
   - `get_features(start_date, end_date)` — read from Feature Store
   - Create Feature View for training data extraction

9. Create `src/feature_pipeline/run_pipeline.py` — orchestrate: fetch → engineer → store. This is the entry point for both manual runs and CI/CD.

---

### Phase 3 — Historical Backfill

10. Create `src/feature_pipeline/backfill.py`:
    - Use OpenWeatherMap **Historical Air Pollution API** (`/air_pollution/history`) — supports Unix timestamps for past data (free tier: last 1 year)
    - Loop through past dates in configurable chunks (e.g., 7-day windows)
    - Apply same feature engineering from step 7
    - Batch-insert into Hopsworks Feature Store
    - Add progress logging and resume-from-checkpoint capability

11. Create `notebooks/01_eda.ipynb`:
    - Load backfilled data from Hopsworks
    - Visualize AQI trends (daily, weekly, seasonal patterns in Lahore)
    - Correlation heatmaps between pollutants and weather features
    - Distribution analysis, outlier detection
    - Time-series decomposition (trend, seasonality, residual)

---

### Phase 4 — Training Pipeline

12. Create individual model modules in `src/training_pipeline/models/`:
    - `ridge_model.py` — `sklearn.linear_model.Ridge` with hyperparameter tuning via `GridSearchCV`
    - `random_forest.py` — `sklearn.ensemble.RandomForestRegressor`, tune `n_estimators`, `max_depth`
    - `xgboost_model.py` — `xgboost.XGBRegressor` with early stopping
    - `lstm_model.py` — TensorFlow/Keras `Sequential` with LSTM layers, input shape = (sequence_length, n_features), output = 3 values (day 1, 2, 3 AQI)
    - Each module exposes: `build_model()`, `train(X, y)`, `predict(X)` interface

13. Create `src/training_pipeline/evaluate.py`:
    - Compute **RMSE**, **MAE**, **R²** for each model
    - Generate comparison table and plots
    - Select best model based on RMSE

14. Create `src/training_pipeline/train.py` — orchestrate:
    - Fetch training features from Hopsworks Feature View
    - Train/test split (time-based, no random split for time-series)
    - Train all 4 models
    - Evaluate and log metrics
    - Register best model in Hopsworks Model Registry (`mr.python.create_model()`)
    - Save model artifacts (sklearn: joblib, TF: SavedModel/H5)

15. Create `notebooks/03_model_experiments.ipynb`:
    - Interactive model comparison
    - Hyperparameter tuning experiments
    - Residual analysis, prediction vs actual plots

---

### Phase 5 — Inference Pipeline

16. Create `src/inference/predict.py`:
    - Load best model from Hopsworks Model Registry
    - Fetch latest features from Feature Store
    - Generate predictions for **Day+1, Day+2, Day+3**
    - For LSTM: use autoregressive prediction (feed day+1 prediction as input for day+2)
    - For tree/linear models: create feature vectors for each future day using weather forecast data from OpenWeatherMap 5-day forecast API
    - Return structured prediction results with confidence intervals

---

### Phase 6 — Explainability

17. Create `src/explainability/shap_explain.py`:
    - `TreeExplainer` for Random Forest / XGBoost
    - `KernelExplainer` for Ridge / LSTM
    - Generate: summary plot, force plot for individual predictions, feature importance bar chart
    - Save SHAP plots as images for dashboard display

18. Create `src/explainability/lime_explain.py`:
    - `LimeTabularExplainer` for local interpretability
    - Generate per-prediction explanations

---

### Phase 7 — Web Dashboard

19. Create `app/flask_api.py`:
    - `GET /predict` — returns 3-day AQI forecast JSON
    - `GET /features` — returns latest feature values
    - `GET /explain` — returns SHAP feature importance
    - `GET /health` — health check endpoint

20. Create `app/alerts.py`:
    - Define AQI categories: Good (0-50), Moderate (51-100), Unhealthy for Sensitive (101-150), Unhealthy (151-200), Very Unhealthy (201-300), Hazardous (301-500)
    - Color coding and health recommendations per category
    - Alert trigger when predicted AQI > 150

21. Create `app/streamlit_app.py`:
    - **Header:** City name, current AQI with color-coded badge, last updated timestamp
    - **3-Day Forecast Card:** Day-by-day AQI with category labels, trend arrow, Plotly line chart
    - **Alert Banner:** Red warning if any day is predicted Unhealthy+
    - **Feature Importance Panel:** SHAP summary plot, top-5 features bar chart
    - **Historical Trends Tab:** Interactive time-series of past AQI (30-day window)
    - **Model Info Tab:** Current model type, RMSE/MAE/R², training date
    - **Sidebar:** City selection (extensible), model selector, refresh button

---

### Phase 8 — CI/CD Automation (GitHub Actions)

22. Create `.github/workflows/feature_pipeline.yml`:
    - **Schedule:** `cron: '0 * * * *'` (every hour)
    - Steps: checkout → setup Python → install deps → run `src/feature_pipeline/run_pipeline.py`
    - Secrets: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`

23. Create `.github/workflows/training_pipeline.yml`:
    - **Schedule:** `cron: '0 2 * * *'` (daily at 2 AM UTC)
    - Steps: checkout → setup Python → install deps → run `src/training_pipeline/train.py`
    - Artifact upload: model metrics log

24. Create `.github/workflows/ci.yml`:
    - **Trigger:** push/PR to main
    - Steps: lint (`flake8`), test (`pytest tests/`), type check

---

### Phase 9 — Testing & Documentation

25. Write tests in `tests/`:
    - `test_feature_pipeline.py` — mock API responses, verify feature engineering output shapes and values
    - `test_training.py` — verify model training on small synthetic dataset
    - `test_inference.py` — verify prediction output format

26. Create `README.md`:
    - Project overview, architecture diagram (Mermaid)
    - Setup instructions (API keys, Hopsworks account, venv)
    - How to run each pipeline manually
    - Dashboard screenshots
    - Model performance comparison table

---

## Verification

- **Feature Pipeline:** Run `python src/feature_pipeline/run_pipeline.py` — verify data appears in Hopsworks Feature Group
- **Backfill:** Run `python src/feature_pipeline/backfill.py --days 30` — verify ≥720 rows in Feature Store
- **Training:** Run `python src/training_pipeline/train.py` — verify RMSE < 30, model registered in Hopsworks
- **Inference:** Run `python src/inference/predict.py` — verify 3 AQI values returned
- **Dashboard:** Run `streamlit run app/streamlit_app.py` — verify all panels render, predictions display
- **CI/CD:** Push to GitHub — verify all 3 workflows trigger and pass
- **Tests:** Run `pytest tests/ -v` — all pass

---

## Decisions

- **API: OpenWeatherMap only** — single API for both weather + air pollution data, simpler key management
- **City: Lahore** (lat=31.5204, lon=74.3587) — hardcoded initially, extensible via sidebar later
- **Feature Store: Hopsworks** — provides both Feature Store and Model Registry in one platform
- **Deploy: Streamlit Community Cloud** — free, public URL, auto-deploys from GitHub `main` branch
- **Time-series split** over random split — prevents data leakage in temporal forecasting
- **4 model types** (Ridge → Random Forest → XGBoost → LSTM) — covers the statistical-to-deep-learning spectrum required by the project spec
- **SHAP primary, LIME secondary** — SHAP has native tree support for RF/XGB (faster), LIME as bonus
