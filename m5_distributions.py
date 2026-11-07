"""M5: distributions and statistical comparisons."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import PARAMETER_LABELS, apply_filters, ranked_only, set_integral_year_axis, format_category_label

def prepare_distribution_analysis(df, filters=None, **params):
    data = ranked_only(apply_filters(df, filters))
    value_column = params.get('value_column', 'score')
    if value_column not in data.columns:
        raise KeyError(f'Unknown distribution column: {value_column}')
    group_keys = ['year', 'category']
    grouped = data.groupby(group_keys, dropna=False)[value_column]
    summary = grouped.agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
    quantiles = grouped.quantile([.25, .75]).unstack()
    quantiles.columns = ['q1', 'q3']
    summary = summary.merge(quantiles.reset_index(), on=group_keys, how='left')
    summary['iqr'] = summary['q3'] - summary['q1']
    return {'data': data, 'value_column': value_column, 'summary': summary}

def _render_distribution_analysis_raw(result, chart_type='histogram'):
    data, value = result['data'].copy(), result['value_column']
    value_label = PARAMETER_LABELS.get(value, value.replace('_', ' ').title())
    if 'category' in data:
        data['category_label'] = data['category'].map(lambda name: format_category_label(name, data))

    # box plot, spread of the value for each category
    if chart_type == 'box':
        return px.box(data, x='category_label', y=value, color='category_label', points='outliers',
                      title=f'Spread of {value_label} across categories',
                      labels={'category_label': 'Category (available years)', value: value_label})

    # violin plot, same idea as box but shows the shape too (box inside it)
    if chart_type == 'violin':
        return px.violin(data, x='category_label', y=value, color='category_label', box=True, points=False,
                         title=f'Shape of {value_label} across categories',
                         labels={'category_label': 'Category (available years)', value: value_label})

    # ecdf, shows what share of institutions sit at or below each value
    if chart_type == 'ecdf':
        return px.ecdf(data, x=value, color='category_label',
                       title=f'ECDF of {value_label}',
                       labels={value: value_label, 'category': 'Category'})

    # quartile chart, % of institutions in each score quartile
    if chart_type == 'quartile':
        counts = data.assign(category_label=data['category'].map(lambda name: format_category_label(name, data))).groupby(['category_label', 'score_quartile']).size().reset_index(name='institutions')
        totals = counts.groupby('category_label')['institutions'].transform('sum')
        counts['percent'] = (counts['institutions'] / totals * 100).round(1)
        return px.bar(counts, x='score_quartile', y='percent', color='category_label', barmode='group',
                      category_orders={'score_quartile': ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']},
                      title='Percentage of institutions in each score quartile',
                      labels={'score_quartile': 'Score quartile', 'percent': 'Percentage of institutions'})

    # table with count, mean, median, min, max, std (already made in prepare function)
    if chart_type == 'stats_table':
        summary = result['summary'].round(2)
        summary['category'] = summary['category'].map(lambda name: format_category_label(name, result['data']))
        summary = summary.rename(columns={
            value: PARAMETER_LABELS.get(value, value.title()),
            'q1': 'Q1', 'q3': 'Q3', 'iqr': 'IQR',
        })
        return go.Figure(go.Table(
            header=dict(values=list(summary.columns)),
            cells=dict(values=[summary[col] for col in summary.columns])
        )).update_layout(title=f'Summary statistics of {value_label}')

    # top 10 vs the rest, mark each row then compare the two groups
    if chart_type == 'top10_vs_rest':
        data['group'] = np.where(data['rank'] <= 10, 'Top 10', 'Rest')
        return px.box(data, x='group', y=value, color='group', points='all',
                      title=f'{value_label} of Top 10 vs the rest',
                      labels={'group': 'Group', value: value_label})

    # box for every year, to see which year is more spread out
    if chart_type == 'by_year':
        data['year'] = data['year'].astype(str)  # string so each year gets its own box
        return px.box(data, x='year', y=value, color='year', points='outliers',
                      title=f'Spread of {value_label} in each year',
                      labels={'year': 'Year', value: value_label})

    # median with Q1 and Q3 per year (filter to one category before using this)
    if chart_type == 'median_iqr':
        if data['year'].nunique() < 2:
            return go.Figure().update_layout(title='Median and IQR trends need data from at least two years')
        grouped = data.groupby('year')[value]
        yearly = pd.DataFrame({
            'Q1': grouped.quantile(0.25),
            'Median': grouped.median(),
            'Q3': grouped.quantile(0.75),
        }).reset_index()
        yearly['IQR'] = yearly['Q3'] - yearly['Q1']
        fig = px.line(yearly, x='year', y=['Q1', 'Median', 'Q3'], markers=True,
                       hover_data=['IQR'],
                       title=f'Median and IQR of {value_label} over the years',
                       labels={'year': 'Year', 'value': value_label, 'variable': 'Measure', 'IQR': 'IQR (Q3 - Q1)'})
        return set_integral_year_axis(fig, yearly['year'].unique())

    # box for each state, only states with 3 or more institutions so it's not cluttered
    if chart_type == 'by_state':
        counts = data['state'].value_counts()
        data = data[data['state'].isin(counts[counts >= 3].index)]
        order = data.groupby('state')[value].median().sort_values(ascending=False).index
        return px.box(data, x='state', y=value, color='state', points='outliers',
                      category_orders={'state': list(order)},
                      title=f'Spread of {value_label} across states',
                      labels={'state': 'State / UT', value: value_label})

    # which parameter varies the most (std dev of each one)
    if chart_type == 'parameter_variability':
        spread = data[['tlr', 'rpc', 'go', 'oi', 'pr']].std().sort_values(ascending=False).reset_index()
        spread.columns = ['parameter', 'std_dev']
        spread['parameter'] = spread['parameter'].map(lambda column: PARAMETER_LABELS.get(column, column.upper()))
        return px.bar(spread, x='parameter', y='std_dev', color='parameter',
                      title='Variability of each NIRF parameter (std dev)',
                      labels={'parameter': 'Parameter', 'std_dev': 'Standard deviation'})
    data, value = result['data'], result['value_column']
    if chart_type == 'histogram':
        if data.empty:
            return go.Figure().update_layout(title=f'No data for the selected {value} filters')
        category = data['category'].iloc[0] if data['category'].nunique() == 1 else None
        category = format_category_label(category, data) if category else None
        title = f'{category}: Distribution of {value}' if category else f'Distribution of {value}'
        return px.histogram(data, x=value, title=title.replace(value, value_label), labels={value: value_label, 'count': 'Institutions'})
    raise NotImplementedError(f'M5 chart type not implemented: {chart_type}')


def render_distribution_analysis(result, chart_type='histogram'):
    """Render M5 figures and keep rank axes within the valid NIRF rank domain."""
    fig = _render_distribution_analysis_raw(result, chart_type)
    if result.get('value_column') != 'rank' or chart_type in {'stats_table', 'quartile', 'parameter_variability'}:
        return fig

    ranks = pd.to_numeric(result['data']['rank'], errors='coerce').dropna()
    ranks = ranks[ranks >= 1]
    if ranks.empty:
        return fig
    upper = max(2, int(np.ceil(ranks.max())))
    tick_step = max(1, int(np.ceil((upper - 1) / 8)))
    if chart_type in {'histogram', 'ecdf'}:
        fig.update_xaxes(range=[1, upper], autorange=False, dtick=tick_step, tick0=1)
    elif chart_type in {'box', 'violin', 'top10_vs_rest', 'by_year', 'median_iqr', 'by_state'}:
        fig.update_yaxes(range=[upper, 1], autorange=False, dtick=tick_step, tick0=1)
    return fig
