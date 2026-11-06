"""M6: correlations and variable relationships."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import PARAMETER_COLUMNS, apply_filters, ranked_only, set_integral_year_axis, format_category_label

def prepare_relationship_analysis(df, filters=None, **params):
    """Prepare data for M6 correlation / relationship charts.

    Optional params
    ---------------
    columns : list of numeric columns to correlate (default: tlr, rpc, go, oi, pr, score)
    method  : 'pearson' (default) or 'spearman'

    Only exactly-ranked rows are used, and missing values are handled pairwise,
    so institutions with an unavailable parameter are never filled with zeros.
    """
    method = params.get("method", "pearson")
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    data = ranked_only(apply_filters(df, filters)).copy()

    requested = params.get("columns", PARAMETER_COLUMNS + ["score"])
    columns = [c for c in requested if c in data.columns]
    if not columns:
        raise KeyError(f"None of the requested columns exist: {requested}")

    for col in columns + ["year", "rank"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.empty:
        correlation = pd.DataFrame(np.nan, index=columns, columns=columns)
    else:
        correlation = data[columns].corr(method=method)

    return {
        "data": data,
        "columns": columns,
        "correlation": correlation,
        "method": method,
        "filters": filters or {},
    }

def _empty_figure(message):
    """Blank figure that explains why nothing is plotted."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _describe_scope(data):
    """Short 'category, year' text for chart titles."""
    parts = []
    if "category" in data.columns and data["category"].nunique() > 0:
        cats = sorted(data["category"].dropna().unique())
        parts.append(format_category_label(cats[0], data) if len(cats) == 1 else f"{len(cats)} categories")
    if "year" in data.columns and data["year"].notna().any():
        years = sorted(data["year"].dropna().astype(int).unique())
        parts.append(str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}")
    return ", ".join(parts)


def _check_columns(result, *names):
    for name in names:
        if name not in result["columns"]:
            raise KeyError(
                f"'{name}' is not an available numeric column. "
                f"Choose from {result['columns']}"
            )


def _fit_trendline(x, y, kind="ols", degree=2):
    """Least-squares trend line computed with numpy (no statsmodels needed).

    Returns (x_grid, y_grid, r_squared) or None when a fit is not possible.
    """
    if kind in (None, "none") or len(x) < 3 or np.ptp(x) == 0:
        return None
    deg = 1 if kind == "ols" else int(degree)
    if len(x) <= deg:
        return None
    coefficients = np.polyfit(x, y, deg)
    predicted = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    grid = np.linspace(x.min(), x.max(), 100)
    return grid, np.polyval(coefficients, grid), r_squared


def _yearly_correlation(data, x_var, y_var, method="pearson", min_n=5):
    """Correlation of two columns for each year (years with < min_n pairs skipped)."""
    rows = []
    for year, group in data.groupby("year"):
        pair = group[[x_var, y_var]].dropna()
        if len(pair) < min_n or pair[x_var].nunique() < 2 or pair[y_var].nunique() < 2:
            continue
        rows.append({
            "year": int(year),
            "correlation": pair[x_var].corr(pair[y_var], method=method),
            "institutions": len(pair),
        })
    return pd.DataFrame(rows, columns=["year", "correlation", "institutions"])


def render_relationship_analysis(
    result,
    x_var="rpc",
    y_var="score",
    chart_type="scatter",
    trendline="ols",
    degree=2,
    color_by=None,
    min_n=5,
):
    """Render an M6 relationship chart.

    chart_type
        'scatter'              two-variable scatter with optional trend line
        'matrix'               correlation heatmap of all parameters
        'scatter_matrix'       pairwise scatter grid of all parameters
        'trend'                yearly correlation between x_var and y_var
        'correlation_trend'    yearly correlation of every parameter with y_var
        'parameter_trend'      yearly mean of every parameter
    trendline
        'ols' (straight line), 'poly' (polynomial of `degree`), or 'none'.
    color_by
        Optional column used to colour scatter points (e.g. 'category', 'state').
    """
    data = result["data"]
    columns = result["columns"]
    method = result.get("method", "pearson")

    if data.empty:
        return _empty_figure("No data for the selected filters")

    scope = _describe_scope(data)

    if chart_type == "scatter":
        _check_columns(result, x_var, y_var)
        if color_by is not None and color_by not in data.columns:
            raise KeyError(f"Unknown color_by column: {color_by}")
        plot_data = data.dropna(subset=[x_var, y_var])
        if plot_data.empty:
            return _empty_figure(f"No complete {x_var}/{y_var} pairs for the selected filters")

        hover_cols = [c for c in ["year", "category", "rank", "score"] if c in plot_data.columns]
        fig = px.scatter(
            plot_data, x=x_var, y=y_var, color=color_by,
            hover_name="institution", hover_data=hover_cols,
        )
        r_text = ""
        if plot_data[x_var].nunique() > 1 and plot_data[y_var].nunique() > 1:
            r = plot_data[x_var].corr(plot_data[y_var], method=method)
            r_text = f" | {'r' if method == 'pearson' else 'rho'} = {r:.2f}"
        fit = _fit_trendline(
            plot_data[x_var].to_numpy(dtype=float),
            plot_data[y_var].to_numpy(dtype=float),
            trendline, degree,
        )
        if fit is not None:
            grid, fitted, r2 = fit
            label = "Linear trend" if trendline == "ols" else f"Polynomial trend (degree {int(degree)})"
            fig.add_trace(go.Scatter(
                x=grid, y=fitted, mode="lines", name=label,
                line=dict(color="black", dash="dash"),
            ))
            if not np.isnan(r2):
                r_text += f" | R² = {r2:.2f}"
        fig.update_layout(
            title=f"{x_var.upper()} vs {y_var.upper()} ({scope}, n = {len(plot_data)}){r_text}",
            xaxis_title=x_var.upper(), yaxis_title=y_var.upper(),
        )
        valid_ranks = pd.to_numeric(data["rank"], errors="coerce").dropna()
        if not valid_ranks.empty:
            rank_max = max(2, int(np.ceil(valid_ranks.max())))
            rank_step = max(1, int(np.ceil((rank_max - 1) / 8)))
            if x_var == "rank":
                fig.update_xaxes(range=[1, rank_max], autorange=False, dtick=rank_step, tick0=1)
            if y_var == "rank":
                fig.update_yaxes(range=[rank_max, 1], autorange=False, dtick=rank_step, tick0=1)
        return fig

    if chart_type == "matrix":
        corr = result["correlation"]
        labels = [c.upper() for c in corr.columns]
        fig = px.imshow(
            corr.to_numpy(), x=labels, y=labels, text_auto=".2f",
            zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
            title=f"{method.title()} Correlation Matrix ({scope})",
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        return fig

    if chart_type == "score_correlations":
        if "score" not in result["correlation"].columns:
            return _empty_figure("Overall score is not available for the selected filters")
        correlations = result["correlation"]["score"].drop(labels=["score"], errors="ignore").dropna()
        if correlations.empty:
            return _empty_figure("No parameter correlations with overall score are available")
        summary = correlations.rename_axis("parameter").reset_index(name="correlation")
        summary["direction"] = np.where(summary["correlation"].ge(0), "Positive", "Negative")
        summary["parameter"] = summary["parameter"].str.upper()
        summary = summary.reindex(summary["correlation"].abs().sort_values().index)
        fig = px.bar(
            summary, x="correlation", y="parameter", orientation="h", color="direction",
            color_discrete_map={"Positive": "#2589BD", "Negative": "#D1495B"},
            title=f"NIRF Parameter Correlations with Overall Score ({method.title()})",
            labels={"correlation": "Correlation coefficient", "parameter": "Parameter"},
        )
        fig.add_vline(x=0, line_color="#FFFFFF", line_width=1)
        return fig.update_xaxes(range=[-1, 1])

    if chart_type == "scatter_matrix":
        if color_by is not None and color_by not in data.columns:
            raise KeyError(f"Unknown color_by column: {color_by}")
        plot_data = data.dropna(subset=columns)
        if plot_data.empty:
            return _empty_figure("No institutions with all parameters available")
        fig = px.scatter_matrix(
            plot_data, dimensions=columns, color=color_by, hover_name="institution",
            title=f"Pairwise Relationships ({scope})",
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        return fig

    if chart_type in ("trend", "correlation_trend", "parameter_trend"):
        if data["year"].nunique() < 2:
            return _empty_figure("Trend charts need at least two years: remove the year filter")

        if chart_type == "trend":
            _check_columns(result, x_var, y_var)
            yearly = _yearly_correlation(data, x_var, y_var, method, min_n)
            if yearly.empty:
                return _empty_figure("Not enough data in any year to compute a correlation")
            fig = px.line(
                yearly, x="year", y="correlation", markers=True,
                hover_data=["institutions"],
                title=f"{x_var.upper()} vs {y_var.upper()}: Correlation by Year ({scope})",
            )
            fig.update_yaxes(range=[-1, 1], title=f"{method.title()} correlation")
            return set_integral_year_axis(fig, yearly["year"].unique())

        if chart_type == "correlation_trend":
            _check_columns(result, y_var)
            frames = []
            for column in [c for c in columns if c != y_var]:
                yearly = _yearly_correlation(data, column, y_var, method, min_n)
                yearly["parameter"] = column.upper()
                frames.append(yearly)
            long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if long.empty:
                return _empty_figure("Not enough data in any year to compute correlations")
            fig = px.line(
                long, x="year", y="correlation", color="parameter", markers=True,
                hover_data=["institutions"],
                title=f"Correlation of Each Parameter with {y_var.upper()} by Year ({scope})",
            )
            fig.update_yaxes(range=[-1, 1], title=f"{method.title()} correlation")
            return set_integral_year_axis(fig, long["year"].unique())

        # parameter_trend
        trend_columns = [column for column in columns if column != "rank"]
        if not trend_columns:
            return _empty_figure("No parameter score columns are available for this trend")
        means = data.groupby("year")[trend_columns].mean().reset_index()
        long = means.melt(id_vars="year", var_name="parameter", value_name="mean_value")
        long["parameter"] = long["parameter"].str.upper()
        fig = px.line(
            long, x="year", y="mean_value", color="parameter", markers=True,
            title=f"Average Parameter Scores by Year ({scope})",
        )
        fig.update_yaxes(title="Mean score")
        return set_integral_year_axis(fig, long["year"].unique())

    raise NotImplementedError(f"M6 chart type not implemented: {chart_type}")
