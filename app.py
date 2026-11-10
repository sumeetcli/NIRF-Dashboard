"""Local Streamlit dashboard for NIRF analyses M3-M8."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
# -------------------- Render hosting compatibility --------------------
import types
nirf_dashboard = types.ModuleType("nirf_dashboard")
nirf_dashboard.__path__ = [str(ROOT)]
nirf_dashboard.__package__ = "nirf_dashboard"
sys.modules["nirf_dashboard"] = nirf_dashboard
# ------------------ End Render hosting compatibility ------------------
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nirf_dashboard.shared import load_data, PARAMETER_COLUMNS, format_category_label, apply_chart_theme
from nirf_dashboard.m3_mobility import prepare_mobility_analysis, render_mobility_analysis
from nirf_dashboard.m4_geospatial import prepare_geospatial_analysis, render_geospatial_analysis
from nirf_dashboard.m5_distributions import prepare_distribution_analysis, render_distribution_analysis
from nirf_dashboard.m6_relationships import prepare_relationship_analysis, render_relationship_analysis
from nirf_dashboard.m7_multivariate import prepare_multivariate_analysis, render_multivariate_analysis
from nirf_dashboard.m8_language import parse_question, dispatch_question

st.set_page_config(page_title="NIRF Insights Dashboard", page_icon="📊", layout="wide")
_CHART_RENDER_COUNT = 0
@st.cache_data
def get_data(): return load_data()
def fig_show(fig,key=None,horizontal_scroll=False):
    global _CHART_RENDER_COUNT
    chart_key=f"nirf_chart_{_CHART_RENDER_COUNT}_{key or 'auto'}"
    _CHART_RENDER_COUNT+=1
    apply_chart_theme(fig)
    if horizontal_scroll:
        chart_width=int(fig.layout.width or 1800)
        html=fig.to_html(
            full_html=False,include_plotlyjs=True,
            config={"responsive":False,"displaylogo":False}
        )
        scrollable_html=(
            '<div style="width:100%;overflow-x:auto;overflow-y:hidden;">'
            f'<div style="width:{chart_width}px;">{html}</div></div>'
        )
        components.html(
            scrollable_html,height=int(fig.layout.height or 480)+45,
            scrolling=False
        )
    else:
        st.plotly_chart(fig,width="stretch",key=chart_key)
try:
    df=get_data()
except Exception as exc:
    st.error(f"Could not load the NIRF dataset: {exc}"); st.stop()

st.title("NIRF Insights Dashboard")
st.caption("Explore NIRF institutions, rankings, performance parameters, and state representation.")
categories=sorted(df.category.dropna().unique().tolist())
years=sorted(pd.to_numeric(df.year,errors="coerce").dropna().astype(int).unique().tolist())
states=sorted(df.state.dropna().unique().tolist())
with st.sidebar:
    st.header("Filters")
    cat=st.selectbox("Category",["All categories",*categories],index=0)
    yr=st.selectbox("Year",["All years",*years],index=0,format_func=str)
    state=st.selectbox("State / UT",["All states / UTs",*states],index=0)
filters={}
if cat!="All categories":filters['category']=[cat]
if yr!="All years":filters['year']=[yr]
if state!="All states / UTs":filters['state']=[state]

T=st.tabs(["M3 Mobility","M4 Geospatial","M5 Distribution","M6 Relationships","M7 Multivariate","M8 Ask a question"])
with T[0]:
    st.subheader("Rank trends and mobility")
    choice=st.selectbox("Chart",["Rank trend","Year trend bar","Adjacent-year movement","Rank-band transition","Top-10 consistency"],key="m3")
    top=st.slider("Top institutions",5,50,10,key="m3top")
    hist_filters={k:v for k,v in filters.items() if k!='year'}
    result=prepare_mobility_analysis(df,hist_filters)
    chart={'Rank trend':'trend','Year trend bar':'year_trend_bar','Adjacent-year movement':'movement','Rank-band transition':'transition','Top-10 consistency':'top10'}[choice]
    if chart=='transition':
        a,b=st.columns(2); start=a.selectbox('Start year',years,index=max(0,len(years)-4)); end=b.selectbox('End year',years,index=len(years)-1)
        if start>=end:st.warning('End year must be later than start year.')
        else:fig_show(render_mobility_analysis(result,chart,start_year=start,end_year=end,top_n=top))
    elif chart in ('trend','year_trend_bar'):
        # This range overrides the single-year dashboard filter for M3 rank trends.
        trend_start,trend_end=st.slider(
            'Rank trend year range',min_value=years[0],max_value=years[-1],
            value=(years[0],years[-1]),step=1,key='m3_trend_year_range'
        )
        if trend_start>=trend_end:
            st.warning('Select a range containing at least two years to show a trend.')
        else:
            # Render one independent, vertically stacked chart per category.
            categories_to_plot=[cat] if cat!='All categories' else categories
            other_filters={k:v for k,v in filters.items() if k not in ('category','year')}
            for category_name in categories_to_plot:
                category_label=format_category_label(category_name,df)
                st.markdown(f"#### {category_label}")
                category_result=prepare_mobility_analysis(
                    df, {**other_filters, 'category':[category_name]}
                )
                category_result=category_result[
                    category_result.year.between(trend_start,trend_end)
                ]
                fig_show(
                    render_mobility_analysis(category_result,chart,top_n=top),
                    key=f"m3_{chart}_{category_name}"
                )
    else:
        if 'year' in filters:result=result[result.year.isin(filters['year'])]
        fig_show(render_mobility_analysis(result,chart,top_n=top))
with T[1]:
    st.subheader("State counts, trends, comparisons, and new entries")
    choice=st.selectbox("Chart",["Choropleth map","State counts","State trends","Compare state ranks","New entries"],key='m4')
    cutoff=st.selectbox('Rank cutoff',[10,50,100,'All ranked'],index=2,key='m4cutoff')
    cut=None if cutoff=='All ranked' else cutoff
    chart={'Choropleth map':'map','State counts':'state_bar','State trends':'state_trend','Compare state ranks':'state_compare','New entries':'new_entries'}[choice]
    if chart=='state_compare':
        start_year,end_year=st.slider(
            'Comparison year range', min_value=years[0], max_value=years[-1],
            value=(years[0],years[-1]), step=1, key='m4_compare_year_range'
        )
        if start_year>=end_year:
            st.warning('Select a range containing at least two years for state comparison.')
        else:
            st.caption('This comparison uses the year range below and the states selected here; the global Year and State filters do not restrict it.')
            state_options=st.multiselect(
                'States to compare', states,
                default=[s for s in ['Karnataka','Tamil Nadu'] if s in states],
                key='m4states'
            )
            compare_filters={k:v for k,v in filters.items() if k=='category'}
            compare_filters['year']=[yr for yr in years if start_year<=yr<=end_year]
            result=prepare_geospatial_analysis(df,compare_filters,top_n=cut)
            if result['all_data']['category'].nunique()!=1:
                st.warning('Select exactly one NIRF category in the dashboard filters to compare ranks.')
            elif len(state_options)<2:
                st.warning('Select at least two states for a state rank comparison.')
            else:
                fig_show(render_geospatial_analysis(result,chart,states=state_options))
    elif chart=='state_trend':
        trend_start,trend_end=st.slider(
            'State trend year range',min_value=years[0],max_value=years[-1],
            value=(years[0],years[-1]),step=1,key='m4_state_trend_year_range'
        )
        if trend_start>=trend_end:
            st.warning('Select at least two years to show state trends.')
        else:
            trend_filters={k:v for k,v in filters.items() if k!='year'}
            trend_filters['year']=[yr for yr in years if trend_start<=yr<=trend_end]
            result=prepare_geospatial_analysis(df,trend_filters,top_n=cut)
            fig_show(render_geospatial_analysis(result,chart))
    elif chart=='state_bar':
        bar_start,bar_end=st.slider(
            'State bar year range',min_value=years[0],max_value=years[-1],
            value=(years[0],years[-1]),step=1,key='m4_state_bar_year_range'
        )
        bar_filters={k:v for k,v in filters.items() if k!='year'}
        bar_filters['year']=[yr for yr in years if bar_start<=yr<=bar_end]
        result=prepare_geospatial_analysis(df,bar_filters,top_n=cut)
        fig=render_geospatial_analysis(result,chart)
        state_count=result['state_summary']['state'].nunique()
        chart_width=max(1800,state_count*75+420)
        fig.update_layout(width=chart_width,height=560,autosize=False,bargap=0.08,bargroupgap=0.02)
        fig_show(fig,key='m4_state_bar_scrollable',horizontal_scroll=True)
    elif chart=='new_entries':
        result=prepare_geospatial_analysis(df,filters,top_n=cut)
        fig=render_geospatial_analysis(result,chart)
        fig_show(fig,horizontal_scroll=True)
    else:
        result=prepare_geospatial_analysis(df,filters,top_n=cut)
        fig_show(render_geospatial_analysis(result,chart))
        if chart=='map':st.caption('State colors show the count of ranked institutions within the selected rank cutoff.')
with T[2]:
    st.subheader('Distributions and statistics')
    chart_names=['Histogram','Box plot','Violin plot','ECDF','Score quartiles','Summary statistics','Top 10 vs rest','By year','Median and IQR','By state','Parameter variability']
    choice=st.selectbox('Chart',chart_names,key='m5')
    measure=st.selectbox('Measure',['score','rank',*PARAMETER_COLUMNS],key='m5value')
    chart=dict(zip(chart_names,['histogram','box','violin','ecdf','quartile','stats_table','top10_vs_rest','by_year','median_iqr','by_state','parameter_variability']))[choice]
    if chart=='quartile' and 'score_quartile' not in df.columns:st.warning('This CSV has no score_quartile column.')
    elif chart=='histogram':
        # Keep category histograms separate and vertically stacked so the page
        # can be scrolled without overplotting/stacking every category together.
        categories_to_plot=[cat] if cat!='All categories' else categories
        histogram_filters={k:v for k,v in filters.items() if k!='category'}
        for category_name in categories_to_plot:
            category_filters={**histogram_filters,'category':[category_name]}
            result=prepare_distribution_analysis(df,category_filters,value_column=measure)
            category_label=format_category_label(category_name,df)
            st.markdown(f'#### {category_label}')
            fig=render_distribution_analysis(result,chart)
            fig.update_layout(height=340,title=f'{category_label}: Distribution of {measure}')
            fig_show(fig,key=f'm5_histogram_{category_name}')
    else:
        distribution_filters=filters
        if chart in ['by_year','median_iqr']:
            range_start,range_end=st.slider(
                'Distribution trend year range',min_value=years[0],max_value=years[-1],
                value=(years[0],years[-1]),step=1,key='m5_trend_year_range'
            )
            if range_start>=range_end:
                st.warning('Select at least two years to show this trend.')
                distribution_filters=None
            else:
                distribution_filters={k:v for k,v in filters.items() if k!='year'}
                distribution_filters['year']=[yr for yr in years if range_start<=yr<=range_end]
        if distribution_filters is not None:
            result=prepare_distribution_analysis(df,distribution_filters,value_column=measure)
            fig_show(render_distribution_analysis(result,chart))
with T[3]:
    st.subheader('Correlations and variable relationships')
    choices=['Scatter plot','Correlation matrix','Scatter matrix','Correlation by year','Parameter correlations by year','Parameter averages by year']
    choice=st.selectbox('Chart',choices,key='m6')
    numeric=[c for c in [*PARAMETER_COLUMNS,'score','rank'] if c in df.columns]
    x=st.selectbox('X variable',numeric,index=numeric.index('rpc') if 'rpc' in numeric else 0)
    y=st.selectbox('Y variable',numeric,index=numeric.index('score') if 'score' in numeric else min(1,len(numeric)-1))
    method=st.selectbox('Correlation method',['pearson','spearman'])
    chart=dict(zip(choices,['scatter','matrix','scatter_matrix','trend','correlation_trend','parameter_trend']))[choice]
    relationship_filters=filters
    if chart in ['trend','correlation_trend','parameter_trend']:
        range_start,range_end=st.slider(
            'Relationship trend year range',min_value=years[0],max_value=years[-1],
            value=(years[0],years[-1]),step=1,key='m6_trend_year_range'
        )
        relationship_filters={k:v for k,v in filters.items() if k!='year'}
        relationship_filters['year']=[yr for yr in years if range_start<=yr<=range_end]
        if range_start==range_end:st.warning('Select at least two years to show a trend.')
    base_columns=[*PARAMETER_COLUMNS,'score']
    if chart in ['scatter','trend','correlation_trend']:
        base_columns.extend([x,y])
    cols=list(dict.fromkeys(base_columns))
    result=prepare_relationship_analysis(df,relationship_filters,columns=cols,method=method)
    if chart in ['trend','correlation_trend','parameter_trend'] and len(relationship_filters.get('year',years))<2:
        pass
    elif chart in ['trend','correlation_trend','parameter_trend'] and result['data'].year.nunique()<2:
        st.warning('Not enough data across the selected years to show a trend.')
    else:fig_show(render_relationship_analysis(result,x_var=x,y_var=y,chart_type=chart))
with T[4]:
    st.subheader('Multivariate patterns, gaps, outliers, and volatility')
    labels=['Radar: top five','Parallel coordinates','Parameter gap','Research vs teaching','Teaching vs perception','Parameter-gap distribution','Outliers','Score changes','Rank volatility','Research-teaching gap outliers']
    values=['radar','parallel','gap','research_teaching','teaching_perception','parameter_distribution','outlier','score_change','volatility','gap_outlier']
    choice=st.selectbox('Chart',labels,key='m7'); chart=values[labels.index(choice)]
    if chart in ['radar','volatility'] and len(filters.get('category',[]))!=1:st.warning('Select exactly one category in the sidebar.')
    elif chart=='radar' and len(filters.get('year',[]))!=1:st.warning('Select exactly one year in the sidebar.')
    else:fig_show(render_multivariate_analysis(prepare_multivariate_analysis(df,filters),chart))
with T[5]:
    st.subheader('Ask a question about the data')
    st.caption('Examples: “hist RPC for Overall in 2025”, “state map top 100”, “RPC vs score scatter in Overall”.')
    question=st.text_input('Question',key='m8question')
    if st.button('Generate chart'):
        if not question.strip():st.warning('Enter a question first.')
        else:
            parsed=parse_question(question,df)
            st.session_state['m8parsed']=parsed
            answer=dispatch_question(parsed,df,filters=filters)
            st.session_state['m8answer']=answer
    parsed_view=st.session_state.get('m8parsed')
    if parsed_view:
        with st.expander('Parsed query (JSON)',expanded=True):
            st.json(parsed_view)
    answer=st.session_state.get('m8answer')
    if answer:
        if answer.get('status')=='ready':
            st.success(answer['message'])
            if answer.get('direct_answer'):
                st.markdown(f"**Answer:** {answer['direct_answer']}")
                answer_table=answer.get('answer_table')
                if answer_table is not None and not answer_table.empty:
                    with st.expander('Chart data for this answer'):
                        st.dataframe(answer_table, use_container_width=True, hide_index=True)
            fig_show(answer['figure'],horizontal_scroll=answer.get('chart_type')=='new_entries')
        else:
            st.info(answer.get('message','Unable to answer this question.'))
            for item in answer.get('alternatives',[]):st.write('• '+item)
