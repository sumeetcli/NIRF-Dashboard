"""M7: multivariate patterns, gaps, outliers, and volatility."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import (PARAMETER_COLUMNS, PARAMETER_LABELS, add_institution_display_label, apply_filters,
                     institution_key_columns, ranked_only, set_integral_year_axis,
                     format_category_label)

def prepare_multivariate_analysis(df, filters=None, **params):
    parameter_columns = ['tlr', 'rpc', 'go', 'oi', 'pr']
    historical_filters = {k: v for k, v in (filters or {}).items() if k != 'year'}
    data = ranked_only(apply_filters(df, historical_filters or None)).copy()
    data = add_institution_display_label(data)
    for col in parameter_columns + ['score', 'rank', 'year']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    available = [c for c in parameter_columns if c in data.columns]
    parameter_data = data[available]

    # Compare parameters only when all five scores are present.
    data["available_parameter_count"] = parameter_data.notna().sum(axis=1)
    complete_rows = data["available_parameter_count"].eq(len(available))

    data["strongest_parameter"] = None
    data["weakest_parameter"] = None
    data.loc[complete_rows, "strongest_parameter"] = (
        parameter_data.loc[complete_rows].idxmax(axis=1)
    )
    data.loc[complete_rows, "weakest_parameter"] = (
        parameter_data.loc[complete_rows].idxmin(axis=1)
    )

    data["parameter_gap"] = (
        parameter_data.max(axis=1) - parameter_data.min(axis=1)
    ).where(complete_rows)
    data['research_teaching_gap'] = data['rpc'] - data['tlr']
    data['teaching_perception_gap'] = data['tlr'] - data['pr']
    scope_columns = [c for c in ['year', 'category'] if c in data]
    data['research_teaching_high'] = data.groupby(scope_columns)['research_teaching_gap'].transform(
        lambda s: s.ge(s[s.gt(0)].quantile(.75)) if s.gt(0).any() else False
    ).fillna(False)
    data['teaching_perception_high'] = data.groupby(scope_columns)['teaching_perception_gap'].transform(
        lambda s: s.ge(s[s.gt(0)].quantile(.75)) if s.gt(0).any() else False
    ).fillna(False)

    def detect_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    data['rpc_outlier'] = data.groupby(['year', 'category'])['rpc'].transform(detect_outliers).fillna(False)
    data['gap_outlier'] = data.groupby(['year', 'category'])['research_teaching_gap'].transform(detect_outliers).fillna(False)
    data['rpc_score_residual'] = np.nan
    for _, group in data.groupby(['year', 'category'], dropna=False):
        paired = group.dropna(subset=['score', 'rpc'])
        if len(paired) < 4 or paired['score'].nunique() < 2:
            continue
        slope, intercept = np.polyfit(paired['score'].to_numpy(), paired['rpc'].to_numpy(), 1)
        residual = paired['rpc'] - (slope * paired['score'] + intercept)
        data.loc[paired.index, 'rpc_score_residual'] = residual
    data['rpc_relative_outlier'] = data.groupby(['year', 'category'])['rpc_score_residual'].transform(detect_outliers).fillna(False)
    group_columns = institution_key_columns(data)
    data = data.sort_values(group_columns + ['year']).copy()
    grouped = data.groupby(group_columns)
    data['previous_score'] = grouped['score'].shift(1)
    data['previous_year'] = grouped['year'].shift(1)
    for column in parameter_columns:
        data[f'previous_{column}'] = grouped[column].shift(1)
    adjacent_year = data['year'] == data['previous_year'] + 1
    data['score_change'] = (data['score'] - data['previous_score']).where(adjacent_year)
    volatility = data.groupby(group_columns)['rank'].std().reset_index(name='rank_volatility')
    data = data.merge(volatility, on=group_columns, how='left')
    data['score_change_outlier'] = data.groupby(['year', 'category'])['score_change'].transform(detect_outliers).fillna(False)
    data = apply_filters(data, filters)
    return {'data': data, 'parameter_columns': available}


def render_multivariate_analysis(result, chart_type='radar'):
    data = result['data'].copy()
    parameters = result['parameter_columns']
    if data.empty:
        return go.Figure().update_layout(title="No data available for selected filters")

    if chart_type == "radar":
        if data["category"].nunique() != 1:
            raise ValueError("Select exactly one NIRF category for the radar chart.")
        if data["year"].nunique() != 1:
            raise ValueError("Select exactly one year for the radar chart.")

        plot_data = (
            data.dropna(subset=parameters + ["rank"])
            .sort_values("rank")
            .drop_duplicates(institution_key_columns(data, include_category=False))
            .head(5)
        )

        fig = go.Figure()
        for _, row in plot_data.iterrows():
            values = [row[p] for p in parameters]
            fig.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=[PARAMETER_LABELS.get(p, p.upper()) for p in parameters] + [PARAMETER_LABELS.get(parameters[0], parameters[0].upper())],
                fill="toself",
                name=row.get("institution_label", row["institution"])
            ))

        fig.update_layout(
            title=f"Top Five Institutions: {format_category_label(data['category'].iloc[0], data)}, {int(data['year'].iloc[0])}",
            polar=dict(radialaxis=dict(visible=True))
        )
        return fig
    elif chart_type == 'parallel':
        plot_data = data.dropna(subset=parameters + ['score'])
        return px.parallel_coordinates(plot_data, dimensions=parameters, color='score', color_continuous_scale='Blues', title='Comparison of Five NIRF Parameters', labels={**{p: PARAMETER_LABELS.get(p, p.upper()) for p in parameters}, 'score':'Overall score'})

    elif chart_type == 'gap':
        plot_data = data.dropna(subset=['parameter_gap']).nlargest(20, 'parameter_gap').copy()
        if plot_data.empty:
            return go.Figure().update_layout(title='No complete parameter scores are available')
        plot_data['institution_year'] = plot_data['institution_label'] + ' (' + plot_data['year'].astype(int).astype(str) + ')'
        fig = px.bar(plot_data.sort_values('parameter_gap'), x='parameter_gap', y='institution_year', orientation='h',
            color='strongest_parameter', hover_data=['state', 'strongest_parameter', 'weakest_parameter', 'score'],
            title='Institutions with the Largest Strongest-to-Weakest Parameter Score Gaps',
            labels={'parameter_gap':'Strongest minus weakest parameter score points','institution_year':'Institution (year)','strongest_parameter':'Strongest parameter'})
        return fig.update_layout(yaxis={'categoryorder':'total ascending'})

    elif chart_type == 'research_teaching':
        plot_data = data.dropna(subset=['tlr', 'rpc'])
        plot_data = plot_data.copy()
        plot_data['Research substantially higher'] = plot_data['research_teaching_high']
        fig = px.scatter(plot_data, x='tlr', y='rpc', color='Research substantially higher', hover_name='institution_label', hover_data=['state', 'year', 'score', 'rank'], title='Research Performance versus Teaching Performance', labels={'tlr':'Teaching, Learning & Resources (TLR) score','rpc':'Research & Professional Practice (RPC) score'})
        low = min(plot_data['tlr'].min(), plot_data['rpc'].min())
        high = max(plot_data['tlr'].max(), plot_data['rpc'].max())
        fig.add_shape(type='line', x0=low, y0=low, x1=high, y1=high, line={'color': '#FFFFFF', 'dash': 'dash'})
        return fig

    elif chart_type == 'teaching_perception':
        plot_data = data.copy()
        plot_data['Teaching substantially higher'] = plot_data['teaching_perception_high']
        return px.scatter(plot_data, x='tlr', y='pr', color='Teaching substantially higher', hover_name='institution_label', hover_data=['state', 'year', 'score', 'rank'], title='Teaching Performance versus Perception', labels={'tlr':'Teaching, Learning & Resources (TLR) score','pr':'Perception (PR) score'})

    elif chart_type == 'parameter_distribution':
        plot_data = data.copy()
        plot_data['category_label'] = plot_data['category'].map(lambda name: format_category_label(name, plot_data))
        return px.box(plot_data, x='category_label', y='parameter_gap', color='category_label', points='outliers', title='Distribution of Strongest-to-Weakest Parameter Score Gaps', labels={'category_label':'Category (available years)','parameter_gap':'Score point gap'})

    elif chart_type == 'outlier':
        return px.scatter(data, x='score', y='rpc', color='rpc_outlier', hover_name='institution_label', hover_data=['state', 'year', 'rank', 'tlr'], title='Research Score Outlier Detection', labels={'score':'Overall score','rpc':PARAMETER_LABELS['rpc'],'rpc_outlier':'Research outlier','tlr':PARAMETER_LABELS['tlr']})

    elif chart_type == 'research_score_relative_outlier':
        plot_data = data.dropna(subset=['score', 'rpc', 'rpc_score_residual'])
        if plot_data.empty:
            return go.Figure().update_layout(title='Not enough paired research and overall scores to identify relative outliers')
        fig = px.scatter(
            plot_data, x='score', y='rpc', color='rpc_relative_outlier', hover_name='institution_label',
            hover_data=['state', 'year', 'rank', 'rpc_score_residual'],
            title='Research Scores Higher than Expected for Overall Score',
            labels={'score': 'Overall score', 'rpc': PARAMETER_LABELS['rpc'], 'rpc_relative_outlier': 'Unusually high RPC relative to score'},
        )
        if plot_data['score'].nunique() > 1:
            slope, intercept = np.polyfit(plot_data['score'].to_numpy(), plot_data['rpc'].to_numpy(), 1)
            xline = np.linspace(plot_data['score'].min(), plot_data['score'].max(), 100)
            fig.add_trace(go.Scatter(x=xline, y=slope*xline+intercept, mode='lines', name='Expected RPC', line={'color':'#FFFFFF','dash':'dash'}))
        return fig

    elif chart_type == 'score_change':
        plot_data = data.dropna(subset=['score_change'])
        fig = px.scatter(plot_data, x='year', y='score_change', color='score_change_outlier', hover_name='institution_label', hover_data=['state', 'previous_score', 'score'], title='Year-to-Year Overall Score Changes', labels={'score_change':'Change in overall score','score_change_outlier':'Unusual change'})
        return set_integral_year_axis(fig, plot_data['year'].unique())

    elif chart_type == 'score_gain_parameter_decline':
        plot_data = data.dropna(subset=['score_change']).copy()
        declined_parameters = []
        for column in parameters:
            previous_column = f'previous_{column}'
            if previous_column in plot_data:
                declined_parameters.append(plot_data[column].lt(plot_data[previous_column]).rename(column))
        if not declined_parameters:
            return go.Figure().update_layout(title='No parameter history is available for comparison')
        decline_flags = pd.concat(declined_parameters, axis=1)
        plot_data['declining_parameters'] = decline_flags.apply(
            lambda row: ', '.join(row.index[row.fillna(False)].tolist()), axis=1
        )
        plot_data = plot_data[
            plot_data['score_change'].gt(0) & plot_data['declining_parameters'].ne('')
        ].nlargest(30, 'score_change')
        if plot_data.empty:
            return go.Figure().update_layout(title='No adjacent-year cases match: score increased while a parameter declined')
        plot_data['institution_year'] = plot_data['institution_label'].astype(str) + ' (' + plot_data['year'].astype(int).astype(str) + ')'
        return px.bar(
            plot_data, x='score_change', y='institution_year', color='declining_parameters', orientation='h',
            hover_data=['category', 'previous_score', 'score'],
            title='Institutions with Score Gains Despite Declining Parameters',
            labels={'score_change': 'Overall score change', 'institution_year': 'Institution (year)', 'declining_parameters': 'Declining parameters'},
        ).update_layout(yaxis={'categoryorder': 'total ascending'})

    elif chart_type == 'parameter_rank':
        plot_data = data.dropna(subset=parameters + ['rank']).copy()
        if plot_data.empty:
            return go.Figure().update_layout(title='No complete parameter scores and ranks are available')
        plot_data['strongest_parameter_score'] = plot_data[parameters].max(axis=1)
        plot_data['strongest_parameter'] = plot_data[parameters].idxmax(axis=1)
        rank_median = float(plot_data['rank'].median())
        parameter_threshold = float(plot_data['strongest_parameter_score'].quantile(0.75))
        plot_data['strong_parameter_poor_rank'] = plot_data['strongest_parameter_score'].ge(parameter_threshold) & plot_data['rank'].ge(rank_median)
        fig = px.scatter(
            plot_data, x='rank', y='strongest_parameter_score', color='strong_parameter_poor_rank',
            hover_name='institution_label', hover_data=['state', 'year', 'score', 'rank', 'strongest_parameter'],
            title='Strong Parameter Scores versus Overall Rank',
            labels={'rank': 'NIRF rank (higher number is worse)', 'strongest_parameter_score': 'Strongest parameter score', 'strongest_parameter': 'Strongest parameter'},
        )
        fig.add_vline(x=rank_median, line_dash='dash', line_color='#FFFFFF', annotation_text='Median rank')
        fig.add_hline(y=parameter_threshold, line_dash='dash', line_color='#FFFFFF', annotation_text='Top-quartile parameter score')
        return fig

    elif chart_type == "volatility":
        if data["category"].nunique() != 1:
            raise ValueError("Select exactly one NIRF category for the volatility chart.")

        plot_data = (
            data.drop_duplicates(institution_key_columns(data))
            .dropna(subset=["rank_volatility"])
            .nlargest(20, "rank_volatility")
        )

        fig = px.bar(
            plot_data,
            x="institution_label",
            y="rank_volatility",
            hover_name="institution_label",
            title=f"Top 20 Institutions by Rank Volatility: {format_category_label(data['category'].iloc[0], data)}"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Standard Deviation of Rank"
        )
        return fig

    elif chart_type == 'gap_outlier':
        return px.scatter(data, x='tlr', y='rpc', color='gap_outlier', hover_name='institution_label', hover_data=['state', 'year', 'research_teaching_gap', 'rank'], title='Unusual Research and Teaching Performance Gaps')

    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

