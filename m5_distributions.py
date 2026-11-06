"""M5: distributions and statistical comparisons."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .shared import apply_filters, ranked_only, set_integral_year_axis, format_category_label

def prepare_distribution_analysis(df, filters=None, **params):
    data = ranked_only(apply_filters(df, filters))
    value_column = params.get('value_column', 'score')
    if value_column not in data.columns:
        raise KeyError(f'Unknown distribution column: {value_column}')
    summary = data.groupby(['year', 'category'], dropna=False)[value_column].agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
    return {'data': data, 'value_column': value_column, 'summary': summary}

def _render_distribution_analysis_raw(result, chart_type='histogram'):
    # TODO (M5): add box, violin, quartile, and comparison variants.
     # my M5 stuff, every chart_type below returns its own figure
    data, value = result['data'].copy(), result['value_column']

    # box plot, spread of the value for each category
    if chart_type == 'box':
        return px.box(data, x='category', y=value, color='category', points='outliers',
                      title=f'Spread of {value} across categories',
                      labels={'category': 'Category', value: value})

    # violin plot, same idea as box but shows the shape too (box inside it)
    if chart_type == 'violin':
        return px.violin(data, x='category', y=value, color='category', box=True, points=False,
                         title=f'Shape of {value} across categories',
                         labels={'category': 'Category', value: value})

    # ecdf, shows what share of institutions sit at or below each value
    if chart_type == 'ecdf':
        return px.ecdf(data, x=value, color='category',
                       title=f'ECDF of {value}',
                       labels={value: value, 'category': 'Category'})

    # quartile chart, % of institutions in each score quartile
    if chart_type == 'quartile':
        counts = data.groupby(['category', 'score_quartile']).size().reset_index(name='institutions')
        totals = counts.groupby('category')['institutions'].transform('sum')  # total per category
        counts['percent'] = (counts['institutions'] / totals * 100).round(1)
        return px.bar(counts, x='score_quartile', y='percent', color='category', barmode='group',
                      category_orders={'score_quartile': ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']},
                      title='Percentage of institutions in each score quartile',
                      labels={'score_quartile': 'Score quartile', 'percent': 'Percentage of institutions'})

    # table with count, mean, median, min, max, std (already made in prepare function)
    if chart_type == 'stats_table':
        summary = result['summary'].round(2)
        return go.Figure(go.Table(
            header=dict(values=list(summary.columns)),
            cells=dict(values=[summary[col] for col in summary.columns])
        )).update_layout(title=f'Summary statistics of {value}')

    # top 10 vs the rest, mark each row then compare the two groups
    if chart_type == 'top10_vs_rest':
        data['group'] = np.where(data['rank'] <= 10, 'Top 10', 'Rest')
        return px.box(data, x='group', y=value, color='group', points='all',
                      title=f'{value} of Top 10 vs the rest',
                      labels={'group': 'Group', value: value})

    # box for every year, to see which year is more spread out
    if chart_type == 'by_year':
        data['year'] = data['year'].astype(str)  # string so each year gets its own box
        return px.box(data, x='year', y=value, color='year', points='outliers',
                      title=f'Spread of {value} in each year',
                      labels={'year': 'Year', value: value})

    # median with Q1 and Q3 per year (filter to one category before using this)
    if chart_type == 'median_iqr':
        grouped = data.groupby('year')[value]
        yearly = pd.DataFrame({
            'Q1': grouped.quantile(0.25),
            'Median': grouped.median(),
            'Q3': grouped.quantile(0.75),
        }).reset_index()
        fig = px.line(yearly, x='year', y=['Q1', 'Median', 'Q3'], markers=True,
                       title=f'Median and IQR of {value} over the years',
                       labels={'year': 'Year', 'value': value, 'variable': 'Measure'})
        return set_integral_year_axis(fig, yearly['year'].unique())

    # box for each state, only states with 3 or more institutions so it's not cluttered
    if chart_type == 'by_state':
        counts = data['state'].value_counts()
        data = data[data['state'].isin(counts[counts >= 3].index)]
        order = data.groupby('state')[value].median().sort_values(ascending=False).index
        return px.box(data, x='state', y=value, color='state', points='outliers',
                      category_orders={'state': list(order)},
                      title=f'Spread of {value} across states',
                      labels={'state': 'State', value: value})

    # which parameter varies the most (std dev of each one)
    if chart_type == 'parameter_variability':
        spread = data[['tlr', 'rpc', 'go', 'oi', 'pr']].std().sort_values(ascending=False).reset_index()
        spread.columns = ['parameter', 'std_dev']
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
        return px.histogram(data, x=value, title=title, labels={value: value, 'count': 'Institutions'})
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
