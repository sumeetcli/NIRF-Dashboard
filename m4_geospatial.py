"""M4: state, regional, and geospatial analyses."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import apply_filters, set_integral_year_axis

def prepare_geospatial_analysis(df, filters=None, **params):
    """Prepare state summaries, top-N records, and new-entry counts for M4 charts."""
    data = apply_filters(df, filters).copy()
    data["rank"] = pd.to_numeric(data["rank"], errors="coerce")
    data.loc[data["rank"].lt(1), "rank"] = np.nan
    top_n = params.get("top_n", 100)

    eligible = data[
        data["rank"].notna()
        & data["rank"].ge(1)
        & data.get("rank_status", pd.Series("Ranked", index=data.index)).ne("Rank withdrawn")
    ].copy()

    # Prefer the cleaned NIRF flags for standard cutoffs; use numeric rank otherwise.
    top_flag = {10: "top10", 50: "top50", 100: "top100"}.get(top_n)
    if top_n is None:
        ranked = eligible
    elif top_flag in eligible.columns:
        flag_values = eligible[top_flag].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        ranked = eligible[flag_values].copy()
    else:
        ranked = eligible[eligible["rank"].le(top_n)].copy()

    state_summary = ranked.groupby(["year", "state"], dropna=False).agg(
        institutions=("institution", "nunique"),
        average_rank=("rank", "mean"),
    ).reset_index()

    new_rows = data[
        data.get("is_new_entry", pd.Series(False, index=data.index))
        .astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    ]
    entry_summary = new_rows.groupby(["year", "state"], dropna=False).agg(
        new_entries=("institution", "nunique")
    ).reset_index()

    state_summary = state_summary.merge(entry_summary, on=["year", "state"], how="outer")
    for column in ["institutions", "new_entries"]:
        state_summary[column] = state_summary[column].fillna(0).astype(int)
    state_summary["average_rank"] = pd.to_numeric(state_summary["average_rank"], errors="coerce")
    # Numeric ranks from different NIRF categories are not directly comparable.
    if "category" in data.columns and data["category"].nunique() != 1:
        state_summary["average_rank"] = np.nan
    state_summary = state_summary.sort_values(["year", "state"]).reset_index(drop=True)

    return {
        "data": ranked,
        "all_data": data,
        "state_summary": state_summary,
        "entry_summary": entry_summary,
        "all_states": sorted(df["state"].dropna().astype(str).unique().tolist()),
        "top_n": top_n,
        "filters": filters or {},
    }


def render_geospatial_analysis(result, chart_type="map", top_states=8, states=None):
    """Render M4 state maps and comparisons from prepared geographic summaries."""
    summary = result["state_summary"].copy()
    if summary.empty:
        return go.Figure().update_layout(title="No state data for the selected filters")

    if chart_type == "map":
        geojson_path = Path(__file__).resolve().parent / "data" / "india_states.geojson"
        if not geojson_path.is_file():
            raise FileNotFoundError(
                f"State-boundary GeoJSON not found: {geojson_path}. "
                "Place the bundled india_states.geojson in the data folder."
            )
        with geojson_path.open(encoding="utf-8") as geojson_file:
            state_geojson = json.load(geojson_file)
        geo_names = {
            feature.get("properties", {}).get("name")
            for feature in state_geojson.get("features", [])
        }
        map_years = sorted(pd.to_numeric(result["all_data"]["year"], errors="coerce").dropna().astype(int).unique())
        if not map_years:
            return go.Figure().update_layout(title="No year data matches the selected filters")
        # Build a complete state-by-year grid so states without ranked
        # institutions remain present in the choropleth with a count of zero.
        map_data = pd.MultiIndex.from_product(
            [map_years, sorted(geo_names)], names=["year", "state"]
        ).to_frame(index=False).merge(summary, on=["year", "state"], how="left")
        map_data["institutions"] = map_data["institutions"].fillna(0).astype(int)
        map_data["new_entries"] = map_data["new_entries"].fillna(0).astype(int)
        max_count = max(1, int(map_data["institutions"].max()))
        zero_band = 0.5 / max_count
        choropleth_scale = [
            [0, "#808080"],
            [zero_band, "#808080"],
            [min(1, zero_band + 0.0001), "#deebf7"],
            [1, "#08519c"],
        ]
        fig = px.choropleth(
            map_data,
            geojson=state_geojson,
            locations="state",
            featureidkey="properties.name",
            color="institutions",
            hover_name="state",
            hover_data={
                "year": True,
                "institutions": True,
                "new_entries": True,
                "average_rank": ":.1f",
            },
            animation_frame="year" if map_data["year"].nunique() > 1 else None,
            color_continuous_scale=choropleth_scale,
            range_color=(0, max_count),
            title=f"Top {result['top_n']} Ranked Institutions by State",
            labels={"institutions": "Institutions", "year": "Year"},
        )
        fig.update_traces(marker_line_color="#D0D0D0", marker_line_width=0.8)
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="#000000")
        fig.update_layout(
            height=650,
            margin={"r": 10, "t": 55, "l": 10, "b": 10},
            coloraxis_colorbar_title="Institutions",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font_color="#F2F2F2",
        )
        return fig

    if chart_type == "state_bar":
        plot_data = summary.copy()
        plot_data["year"] = plot_data["year"].astype(int).astype(str)
        year_order = sorted(plot_data["year"].unique(), key=int)
        fig = px.bar(
            plot_data, x="state", y="institutions", color="year", barmode="group",
            category_orders={"year": year_order},
            color_discrete_sequence=px.colors.qualitative.Safe,
            hover_data={"year": True, "new_entries": True, "average_rank": ":.1f"},
            title=f"Top-{result['top_n']} Institutions by State",
            labels={"state": "State / UT", "institutions": "Institutions"},
        ).update_layout(
            height=480,
            margin={"r": 20, "t": 65, "l": 55, "b": 170},
            xaxis_tickangle=-50,
            xaxis_tickfont={"size": 11},
            xaxis_automargin=True,
        )
        return fig

    if chart_type == "state_trend":
        trend_data = summary[summary["institutions"].gt(0)].copy()
        coverage = trend_data.groupby("state")["year"].nunique()
        trend_states = coverage[coverage.ge(2)].index
        trend_data = trend_data[trend_data["state"].isin(trend_states)]
        if trend_data.empty:
            return go.Figure().update_layout(title="No states have ranked institutions in at least two selected years")
        totals = trend_data.groupby("state")["institutions"].sum().nlargest(top_states)
        plot_data = trend_data[trend_data["state"].isin(totals.index)]
        fig = px.line(
            plot_data, x="year", y="institutions", color="state", markers=True,
            title=f"Top {top_states} States: Top-{result['top_n']} Representation Over Time",
            labels={"year": "Year", "institutions": "Institutions"},
        )
        return set_integral_year_axis(fig, plot_data["year"].unique())

    if chart_type == "state_compare":
        category_values = result["all_data"].get("category", pd.Series(dtype=object)).dropna().unique()
        if len(category_values) != 1:
            raise ValueError("Select exactly one NIRF category for state rank comparisons.")
        chosen_states = states or ["Karnataka", "Tamil Nadu"]
        if not chosen_states:
            return go.Figure().update_layout(title="Select at least one state to compare")

        # Compare the best N institutions within each state, rather than only
        # institutions whose national rank is <= N. This ensures states with
        # ranked institutions still get a line even when none is in the national top N.
        ranked = result["all_data"].copy()
        ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
        status = ranked["rank_status"] if "rank_status" in ranked else pd.Series("Ranked", index=ranked.index)
        ranked = ranked[
            ranked["state"].isin(chosen_states)
            & ranked["rank"].notna()
            & ranked["rank"].ge(1)
            & status.ne("Rank withdrawn")
        ].copy()
        ranked = ranked.sort_values(["year", "state", "rank", "institution"])
        if result["top_n"] is not None:
            ranked = ranked.groupby(["year", "state"], group_keys=False).head(int(result["top_n"]))
        plot_data = ranked.groupby(["year", "state"], dropna=False).agg(
            average_rank=("rank", "mean"), institutions=("institution", "nunique")
        ).reset_index()
        plot_data = plot_data[plot_data["average_rank"].notna() & plot_data["average_rank"].ge(1)]
        state_year_counts = plot_data.groupby("state")["year"].nunique()
        states_with_trends = state_year_counts[state_year_counts.ge(2)].index
        plot_data = plot_data[plot_data["state"].isin(states_with_trends)]
        if plot_data.empty or plot_data["state"].nunique() < 2:
            return go.Figure().update_layout(
                title="At least two selected states need ranked records in two or more years"
            )
        highest_rank = max(1, int(np.ceil(plot_data["average_rank"].max())))
        # Keep roughly 8 readable tick labels instead of labeling every rank.
        rank_tick_step = max(1, int(np.ceil((highest_rank - 1) / 8)))
        n_label = "all" if result["top_n"] is None else str(result["top_n"])
        fig = px.line(
            plot_data, x="year", y="average_rank", color="state", markers=True,
            hover_data=["institutions"],
            title=f"Average Rank of Each State's Top {n_label} Institutions: " + " vs ".join(chosen_states),
            labels={"average_rank": "Average rank (lower is better)", "year": "Year"},
        )
        # NIRF rank 1 is the best possible rank. Avoid auto-padding to zero or below.
        fig.update_yaxes(range=[highest_rank, 1], autorange=False, dtick=rank_tick_step, tick0=1)
        return set_integral_year_axis(fig, plot_data["year"].unique())

    if chart_type == "new_entries":
        year_values = sorted(
            pd.to_numeric(result["all_data"]["year"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        )
        state_values = result.get("all_states") or sorted(
            result["all_data"]["state"].dropna().astype(str).unique().tolist()
        )
        if not year_values or not state_values:
            return go.Figure().update_layout(title="No new-entry records for the selected filters")
        entries = pd.MultiIndex.from_product(
            [year_values, state_values], names=["year", "state"]
        ).to_frame(index=False).merge(result["entry_summary"], on=["year", "state"], how="left")
        entries["new_entries"] = entries["new_entries"].fillna(0).astype(int)
        entries["year"] = entries["year"].astype(int).astype(str)
        year_order = sorted(entries["year"].unique(), key=int)
        fig = px.bar(
            entries, x="state", y="new_entries", color="year", barmode="group",
            category_orders={"year": year_order},
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Newly Appearing Institutions by State (All States / UTs)",
            labels={"state": "State / UT", "new_entries": "New institutions"},
        ).update_layout(
            width=max(1800, len(state_values) * 75 + 420),
            height=560,
            autosize=False,
            bargap=0.08,
            bargroupgap=0.02,
            margin={"r": 20, "t": 65, "l": 55, "b": 170},
            xaxis_tickangle=-50,
            xaxis_tickfont={"size": 11},
            xaxis_automargin=True,
        )
        return fig

    raise ValueError(f"Unknown M4 chart type: {chart_type}")
