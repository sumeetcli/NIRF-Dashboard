# NIRF dashboard analysis modules

Extracted from `nirf_dashboard_M3_M8_integrated.ipynb`. The member function implementations are grouped by module; shared data utilities are in `shared.py`.

## Files

- `shared.py`: schema, `load_data`, `apply_filters`, and `ranked_only`.
- `m3_mobility.py` through `m7_multivariate.py`: each member's preparation and rendering functions.
- `m8_language.py`: `parse_question` and `dispatch_question`, wired to the M3-M7 functions.
- `data/nirf_clean.csv`: the cleaned dataset used by the integrated notebook.

## Import example

Run from the directory containing the `nirf_dashboard` package's parent:

```python
from nirf_dashboard.shared import load_data
from nirf_dashboard.m3_mobility import prepare_mobility_analysis, render_mobility_analysis

df = load_data()
result = prepare_mobility_analysis(df, {"category": ["Overall"]})
fig = render_mobility_analysis(result, chart_type="trend")
fig.show()
```

For M8, import `parse_question` and `dispatch_question` from `nirf_dashboard.m8_language`; then pass the loaded `df` to both parsing and dispatch. Streamlit app assembly is a separate next step.

The default CSV path is this package's `data/nirf_clean.csv`. Override it with `load_data(path=...)` or set `NIRF_DATA_PATH`.


## Run the local dashboard

Open a terminal in this `nirf_dashboard` folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the bundled `data/nirf_clean.csv` by default. To use another cleaned CSV, set `NIRF_DATA_PATH` to its full path before launching. On Windows PowerShell:

```powershell
$env:NIRF_DATA_PATH = "C:\path\to\nirf_clean.csv"
streamlit run app.py
```

The sidebar filters are shared across M3-M8. The M8 question box routes to the extracted member functions and displays the returned Plotly figure in the app.

M4 uses a local state-boundary GeoJSON for choropleth maps: `data/india_states.geojson`. Boundary attribution and license details are in `data/BOUNDARY_ATTRIBUTION.txt`. The GeoJSON is bundled, so the app does not fetch map boundaries at runtime.
