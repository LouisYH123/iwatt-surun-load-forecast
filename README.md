# iWatt SuRun Load Forecast

Open-source research code and aggregate evaluation results for the SuRun
Jiangsu user-load forecasting study.

## Included

- Calendar, weather, market-signal, segmentation, fusion, PV-sensitive and
  weekend-aware forecasting code.
- Strict backtesting and operational-delay experiments.
- Aggregate backtest and experiment metrics under `results/`.
- Unit tests that do not require private production data.

## Not included

- Customer-level load curves, account names, phone numbers or identifiers.
- Raw API responses, database backups, credentials or internal host state.
- PowerScope or production deployment configuration.

The published CSV files are aggregate evaluation outputs. They are suitable
for model comparison, but are not a substitute for the original private
training dataset.

## Layout

```text
project/research/jiangsu/user_load_forecast/  research and backtest code
results/                                      aggregate CSV metrics
docs/                                         public data and release notes
```

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[ml]'
python -m unittest discover \
  project/research/jiangsu/user_load_forecast \
  -p 'test_*.py'
```

Some experiment modules use optional large dependencies such as PyTorch,
Chronos, LightGBM, XGBoost or CatBoost. Production API integration is not
bundled; provide your own adapters and endpoint configuration.

## License

Apache-2.0. See `LICENSE`.
