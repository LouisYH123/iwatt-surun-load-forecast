# SuRun load-forecast research module

This package contains the research and backtesting implementation used for
the Jiangsu aggregate load-forecast study. It includes calendar baselines,
weather features, segmented models, model fusion, PV-sensitive analysis,
weekend experts, and Chronos-family experiments.

The public repository deliberately excludes customer-level load records,
credentials, database dumps, internal API responses, and production runtime
state. Online forecast commands therefore require callers to provide their
own data and API adapters.

The original import namespace is retained:

```python
from project.research.jiangsu.user_load_forecast.calendar_features import day_type
```

Run the dependency-light tests with:

```bash
python -m unittest discover \
  project/research/jiangsu/user_load_forecast \
  -p 'test_*.py'
```
