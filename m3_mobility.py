"""M3: time series and rank mobility analyses."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import apply_filters, set_integral_year_axis

def prepare_mobility_analysis(df, filters=None, **params):
    """Prepare exact-rank trends and valid adjacent-year movement comparisons."""
    data = apply_filters(df, filters).copy()
    data["rank"] = pd.to_numeric(data["rank"], errors="coerce")
    data.loc[data["rank"].lt(1), "rank"] = np.nan
    if "rank_status" in data.columns:
        data.loc[data["rank_status"].eq("Rank withdrawn"), "rank"] = np.nan
    ranked = data[
        data["rank"].notna()
        & data["rank"].ge(1)
        & data.get("rank_status", pd.Series("Ranked", index=data.index)).ne("Rank withdrawn")
    ].copy()

    institution_filter = params.get("institution")
    if institution_filter:
        if isinstance(institution_filter, str):
            institution_filter = [institution_filter]
        data = data[data["institution"].isin(institution_filter)].copy()
        ranked = ranked[ranked["institution"].isin(institution_filter)].copy()

    # Ranks belong to a category. Never compare the same institution across categories.
    group_columns = (["category"] if "category" in ranked.columns else []) + ["institution"]
    ranked = ranked.sort_values(group_columns + ["year"]).copy()
    grouped = ranked.groupby(group_columns, dropna=False, sort=False)
    ranked["prev_rank"] = grouped["rank"].shift(1)
    ranked["prev_year"] = grouped["year"].shift(1)
    ranked["adjacent_year_comparison"] = ranked["year"].eq(ranked["prev_year"] + 1)
    ranked["rank_change"] = (ranked["prev_rank"] - ranked["rank"]).where(
        ranked["adjacent_year_comparison"]
    )
    ranked["movement"] = "No prior-year comparison"
    valid = ranked["adjacent_year_comparison"]
    ranked.loc[valid & ranked["rank_change"].gt(0), "movement"] = "up"
    ranked.loc[valid & ranked["rank_change"].lt(0), "movement"] = "down"
    ranked.loc[valid & ranked["rank_change"].eq(0), "movement"] = "same"
    for column in ["prev_rank", "prev_year", "adjacent_year_comparison", "rank_change", "movement"]:
        data[column] = ranked[column].reindex(data.index)
    data["movement"] = data["movement"].fillna("No exact ranked observation")
    return data


def render_mobility_analysis(
    result, chart_type="trend", start_year=2020, end_year=2025, top_n=10
):
    """Render a mobility view. Trend defaults to the best average-rank top N."""
    if result.empty:
        return go.Figure().update_layout(title="No data for the selected filters")

    if chart_type == "trend":
        plot_data = result[result["rank"].notna() & result["rank"].ge(1)].copy()
        if plot_data.empty:
            return go.Figure().update_layout(title="No valid ranked data for the selected filters")
        group_columns = (["category"] if "category" in plot_data.columns else []) + ["institution"]
        plot_data["year_count"] = plot_data.groupby(group_columns)["year"].transform("nunique")
        plot_data = plot_data[plot_data["year_count"].ge(2)].drop(columns="year_count")
        if plot_data.empty:
            return go.Figure().update_layout(title="No institutions have ranked records in at least two selected years")
        if top_n is not None:
            if "category" in plot_data.columns:
                keep = (
                    plot_data.groupby(["category", "institution"])["rank"].mean()
                    .groupby(level="category", group_keys=False)
                    .nsmallest(top_n)
                    .index
                )
                keep_pairs = set(keep.tolist())
                plot_data = plot_data[
                    plot_data.apply(lambda row: (row["category"], row["institution"]) in keep_pairs, axis=1)
                ]
            else:
                keep = plot_data.groupby("institution")["rank"].mean().nsmallest(top_n).index
                plot_data = plot_data[plot_data["institution"].isin(keep)]
        fig = px.line(
            plot_data, x="year", y="rank", color="institution", markers=True,
            facet_col="category" if "category" in plot_data and plot_data["category"].nunique() > 1 else None,
            facet_col_wrap=4,
            title=(
                f"Rank Trend: {plot_data['institution'].iloc[0]}"
                if plot_data["institution"].nunique() == 1
                else f"Institution Rank Trend (best average-rank {top_n} per category)"
            )
        )
        highest_rank = max(2, int(np.ceil(plot_data["rank"].max())))
        tick_step = max(1, int(np.ceil((highest_rank - 1) / 8)))
        fig.update_yaxes(range=[highest_rank, 1], autorange=False, dtick=tick_step, tick0=1, title="Rank")
        return set_integral_year_axis(fig, plot_data["year"].unique())

    if chart_type == "year_trend_bar":
        plot_data = result[result["rank"].notna() & result["rank"].ge(1)].copy()
        if plot_data.empty:
            return go.Figure().update_layout(title="No valid ranked data for the selected filters")
        group_columns = (["category"] if "category" in plot_data.columns else []) + ["institution"]
        coverage = plot_data.groupby(group_columns)["year"].nunique()
        eligible = coverage[coverage.ge(2)].index
        if "category" in plot_data.columns:
            eligible_pairs = set(eligible.tolist())
            plot_data = plot_data[plot_data.apply(
                lambda row: (row["category"], row["institution"]) in eligible_pairs, axis=1
            )]
            best = (plot_data.groupby(["category", "institution"])["rank"].mean()
                    .groupby(level="category", group_keys=False).nsmallest(top_n or 10).index)
            best_pairs = set(best.tolist())
            plot_data = plot_data[plot_data.apply(
                lambda row: (row["category"], row["institution"]) in best_pairs, axis=1
            )]
        else:
            plot_data = plot_data[plot_data["institution"].isin(eligible)]
            best = plot_data.groupby("institution")["rank"].mean().nsmallest(top_n or 10).index
            plot_data = plot_data[plot_data["institution"].isin(best)]
        yearly = plot_data.groupby("year", as_index=False).agg(
            average_rank=("rank", "mean"), institutions=("institution", "nunique")
        )
        if yearly.empty:
            return go.Figure().update_layout(title="No multi-year ranked records for this category")
        upper = max(2, int(np.ceil(yearly["average_rank"].max())))
        step = max(1, int(np.ceil((upper - 1) / 8)))
        fig = px.bar(yearly, x="year", y="average_rank", custom_data=["institutions"],
                     title=f"Average Rank by Year (best {top_n or 10} institutions)",
                     labels={"year": "Year", "average_rank": "Average rank (lower is better)"},
                     hover_data={"institutions": True})
        # Bars start at rank 1 so the axis never suggests a rank of zero.
        fig.update_traces(base=1, y=yearly["average_rank"] - 1)
        tick_ranks = list(range(1, upper + 1, step))
        fig.update_yaxes(range=[upper - 1, 0], autorange=False,
                         tickmode="array", tickvals=[rank - 1 for rank in tick_ranks],
                         ticktext=[str(rank) for rank in tick_ranks],
                         title="Average rank (lower is better)")
        return set_integral_year_axis(fig, yearly["year"].unique())

    if chart_type in {"transition", "transition_percentage"}:
        # Use the supplied band labels where available; numeric rank is a fallback.
        data = result.copy()
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")
        identity = (["category"] if "category" in data.columns else []) + ["institution"]

        def band_label(row):
            raw_band = row.get("rank_band")
            if pd.notna(raw_band):
                return {
                    "1-10": "Top 10", "11-50": "11-50", "51-100": "51-100",
                    "101-150": "101-150", "151-200": "151-200"
                }.get(str(raw_band), str(raw_band))
            if row.get("rank_status") == "Rank withdrawn":
                return "Rank withdrawn"
            rank = row["rank"]
            if pd.isna(rank) or rank < 1:
                return "Unranked / unavailable"
            if rank <= 10:
                return "Top 10"
            if rank <= 50:
                return "11-50"
            if rank <= 100:
                return "51-100"
            return "101+"

        start = data[data["year"].eq(start_year)].copy()
        end = data[data["year"].eq(end_year)].copy()
        start["start_band"] = start.apply(band_label, axis=1)
        end["end_band"] = end.apply(band_label, axis=1)
        start = start[identity + ["start_band"]]
        end = end[identity + ["end_band"]]
        merged = start.merge(end, on=identity, how="outer")
        merged["start_band"] = merged["start_band"].fillna("Not listed")
        merged["end_band"] = merged["end_band"].fillna("Not listed")
        matrix = pd.crosstab(merged["start_band"], merged["end_band"])
        transition_title = f"Rank Band Transitions ({start_year} to {end_year})"
        if chart_type == "transition_percentage":
            total = float(matrix.to_numpy().sum())
            if total:
                stayed = sum(
                    float(matrix.loc[band, band])
                    for band in matrix.index.intersection(matrix.columns)
                )
                transition_title = (
                    f"Rank Band Transition Shares ({start_year} to {end_year})"
                    f" - {stayed / total * 100:.1f}% stayed in the same band"
                )
                matrix = matrix.div(total).mul(100)
        return px.imshow(
            matrix, text_auto=(".1f" if chart_type == "transition_percentage" else True),
            color_continuous_scale="Blues",
            title=transition_title,
            labels={
                "x": f"Band in {end_year}", "y": f"Band in {start_year}",
                "color": "Percentage (%)" if chart_type == "transition_percentage" else "Institutions",
            }
        )

    if chart_type == "movement":
        valid = result[result["movement"].isin(["up", "down", "same"])]
        summary = valid.groupby(["year", "movement"]).size().reset_index(name="institutions")
        fig = px.bar(
            summary, x="year", y="institutions", color="movement", barmode="group",
            title="Rank Movement Between Adjacent Years"
        )
        return set_integral_year_axis(fig, summary["year"].unique())

    if chart_type in {"improvement", "decline"}:
        data = result[result["rank"].notna() & result["rank"].ge(1)].copy()
        if data.empty or data["year"].nunique() < 2:
            direction = "improvement" if chart_type == "improvement" else "decline"
            return go.Figure().update_layout(title=f"Rank {direction} needs ranked records in at least two years")
        identity = (["category"] if "category" in data.columns else []) + ["institution"]
        start_year, end_year = int(data["year"].min()), int(data["year"].max())
        start = data[data["year"].eq(start_year)][identity + ["rank"]].rename(columns={"rank": "start_rank"})
        end = data[data["year"].eq(end_year)][identity + ["rank"]].rename(columns={"rank": "end_rank"})
        comparison = start.merge(end, on=identity, how="inner")
        comparison["rank_positions_gained"] = comparison["start_rank"] - comparison["end_rank"]
        metric = "rank_positions_gained"
        if chart_type == "decline":
            comparison["rank_positions_lost"] = -comparison["rank_positions_gained"]
            metric = "rank_positions_lost"
        comparison = comparison[comparison[metric].gt(0)]
        if comparison.empty:
            direction = "improved" if chart_type == "improvement" else "declined"
            return go.Figure().update_layout(title=f"No institutions {direction} in rank from {start_year} to {end_year}")
        comparison = comparison.nlargest(top_n or 10, metric).sort_values(metric)
        direction = "Improvement" if chart_type == "improvement" else "Decline"
        fig = px.bar(
            comparison, x=metric, y="institution", orientation="h",
            title=f"Institutions with the Greatest Rank {direction} ({start_year}-{end_year})",
            labels={metric: f"Rank positions {direction.lower()}", "institution": "Institution"},
            hover_data={"start_rank": True, "end_rank": True},
        )
        return fig.update_layout(yaxis={"categoryorder": "total ascending"})

    if chart_type == "largest_movement":
        changes = result[result["rank_change"].notna()].copy()
        if changes.empty:
            return go.Figure().update_layout(title="No adjacent-year rank movements are available")
        n = top_n or 10
        upward = changes[changes["rank_change"].gt(0)].nlargest(n, "rank_change").copy()
        downward = changes[changes["rank_change"].lt(0)].nsmallest(n, "rank_change").copy()
        plot_data = pd.concat([upward, downward], ignore_index=True)
        if plot_data.empty:
            return go.Figure().update_layout(title="No rank movements are available")
        plot_data["movement_label"] = np.where(plot_data["rank_change"].gt(0), "Moved up", "Moved down")
        plot_data["institution_year"] = plot_data["institution"].astype(str) + " (" + plot_data["year"].astype(int).astype(str) + ")"
        plot_data = plot_data.sort_values("rank_change")
        return px.bar(
            plot_data, x="rank_change", y="institution_year", color="movement_label", orientation="h",
            hover_data={"institution": True, "year": True},
            title="Largest Upward and Downward Adjacent-Year Rank Movements",
            labels={"rank_change": "Rank positions gained (+) / lost (-)", "institution_year": "Institution (year)"},
            color_discrete_map={"Moved up": "#20A486", "Moved down": "#D1495B"},
        )

    if chart_type == "top10":
        data = result[result["rank"].notna() & result["rank"].ge(1)].copy()
        data["in_top10"] = data["rank"].le(10)
        group_columns = (["category"] if "category" in data.columns else []) + ["institution"]
        summary = data.groupby(group_columns, dropna=False).agg(
            years_observed=("year", "nunique"), years_in_top10=("in_top10", "sum")
        ).reset_index()
        summary = summary.sort_values("years_in_top10", ascending=False)
        return px.bar(
            summary.head(top_n or 20), x="institution", y="years_in_top10",
            color="category" if "category" in summary and summary["category"].nunique() > 1 else None,
            hover_data=["years_observed"], title="Top-10 Appearances by Institution"
        )

    raise NotImplementedError(f"M3 chart type not implemented: {chart_type}")
