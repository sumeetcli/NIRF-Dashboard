"""Shared data loading, schema, and filtering utilities for the NIRF dashboard."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

PARAMETER_COLUMNS = ["tlr", "rpc", "go", "oi", "pr"]
PARAMETER_LABELS = {
    "tlr": "Teaching, Learning & Resources (TLR)",
    "rpc": "Research & Professional Practice (RPC)",
    "go": "Graduation Outcomes (GO)",
    "oi": "Outreach & Inclusivity (OI)",
    "pr": "Perception (PR)",
    "score": "Overall score",
    "rank": "NIRF rank (lower is better)",
}
PLOTLY_COLORWAY = ["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#B0B0B0"]
REQUIRED_COLUMNS = [
    "year", "category", "institution", "state", "score", "rank",
    *PARAMETER_COLUMNS,
]
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("NIRF_DATA_PATH", PROJECT_DIR / "data" / "nirf_clean.csv"))


def institution_key_columns(data, include_category=True):
    """Return stable keys for institution records, preferring the dataset ID."""
    keys = (["category"] if include_category and "category" in data.columns else [])
    keys.append("institution_id" if "institution_id" in data.columns else "institution")
    return keys


def add_institution_display_label(data):
    """Add readable labels that disambiguate repeated institution names."""
    result = data.copy()
    if "institution" not in result:
        return result
    name = result["institution"].fillna("Unknown institution").astype(str)
    duplicate = result.assign(_name=name).groupby("_name", dropna=False)[
        "institution_id" if "institution_id" in result else "_name"
    ].transform("nunique").gt(1) if "institution_id" in result else name.duplicated(keep=False)
    label = name.copy()
    if "state" in result:
        label = label.where(~duplicate, name + " — " + result["state"].fillna("Unknown state").astype(str))
    if "institution_id" in result:
        still_duplicate = label.duplicated(keep=False)
        label = label.where(~still_duplicate, label + " [ID " + result["institution_id"].astype(str) + "]")
    result["institution_label"] = label
    return result


def apply_chart_theme(fig):
    """Apply shared dark dashboard styling while retaining chart-specific scales."""
    fig.update_layout(template="plotly_dark", colorway=PLOTLY_COLORWAY)
    if not fig.data:
        message = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "No data for the selected filters"
        fig.update_layout(
            title=None,
            showlegend=False,
            annotations=[dict(
                text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=18, color="#D9DEE7"),
                align="center",
            )],
        )
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    return fig


def normalize_years(data):
    """Keep only rows with a whole-number year and store years as integers."""
    result = data.copy()
    numeric_year = pd.to_numeric(result["year"], errors="coerce")
    valid = numeric_year.notna() & np.isclose(numeric_year, np.round(numeric_year))
    result = result.loc[valid].copy()
    result["year"] = np.round(numeric_year.loc[valid]).astype("int64")
    ranges = dict(result.attrs.get("category_year_ranges", {}))
    if "category" in result.columns:
        for category, rows in result.groupby("category", dropna=True):
            observed = pd.to_numeric(rows["year"], errors="coerce").dropna()
            if observed.empty:
                continue
            first = None
            if "category_intro_year" in rows.columns:
                intro = pd.to_numeric(rows["category_intro_year"], errors="coerce").dropna()
                intro = intro[np.isclose(intro, np.round(intro))]
                if not intro.empty:
                    first = int(intro.min())
            ranges.setdefault(str(category), (first or int(observed.min()), int(observed.max())))
    result.attrs["category_year_ranges"] = ranges
    return result


def set_integral_year_axis(fig, years, axis="x"):
    """Set a Plotly axis to show existing years as integral tick labels only."""
    numeric = pd.to_numeric(pd.Series(list(years)), errors="coerce").dropna()
    numeric = numeric[np.isclose(numeric, np.round(numeric))]
    ticks = sorted(set(np.round(numeric).astype(int).tolist()))
    if ticks:
        updater = fig.update_xaxes if axis == "x" else fig.update_yaxes
        updater(tickmode="array", tickvals=ticks, ticktext=[str(year) for year in ticks], title="Year")
    return fig


def format_category_label(category, data):
    """Format a category as ``Name (intro year-last dataset year)``."""
    label = str(category)
    if "category" not in data.columns or "year" not in data.columns:
        return label
    saved_range = data.attrs.get("category_year_ranges", {}).get(label)
    if saved_range:
        return f"{label} ({saved_range[0]}-{saved_range[1]})"
    rows = data[data["category"].astype(str).eq(label)]
    observed = pd.to_numeric(rows["year"], errors="coerce").dropna()
    if observed.empty:
        return label
    first = None
    if "category_intro_year" in rows.columns:
        intro = pd.to_numeric(rows["category_intro_year"], errors="coerce").dropna()
        intro = intro[np.isclose(intro, np.round(intro))]
        if not intro.empty:
            first = int(intro.min())
    if first is None:
        first = int(observed.min())
    last = int(observed.max())
    return f"{label} ({first}-{last})"


def load_data(path=None):
    """Load the cleaned CSV and check that dashboard-required columns exist.

    The default is ``data/nirf_clean.csv`` beside these modules. Set the
    ``NIRF_DATA_PATH`` environment variable or pass ``path`` to use another file.
    """
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"NIRF data file not found: {csv_path}. Put nirf_clean.csv in the data folder "
            "or set NIRF_DATA_PATH to its location."
        )
    data = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return normalize_years(data)


def apply_filters(df, filters=None):
    """Apply optional scalar or collection filters without changing ``df``."""
    result = normalize_years(df) if "year" in df.columns else df.copy()
    if not filters:
        return result
    for column, value in filters.items():
        if value is None or (isinstance(value, str) and value.lower() == "all"):
            continue
        if column not in result.columns:
            raise KeyError(f"Unknown filter column: {column}")
        values = value if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)) else [value]
        result = result[result[column].isin(values)]
    return result


def ranked_only(df):
    """Keep exact positive NIRF ranks; rank-band records remain in source data."""
    result = df.copy()
    numeric_rank = pd.to_numeric(result["rank"], errors="coerce")
    status = df["rank_status"] if "rank_status" in df.columns else pd.Series("Ranked", index=df.index)
    result["rank"] = numeric_rank
    return result[numeric_rank.ge(1) & status.ne("Rank withdrawn")].copy()
