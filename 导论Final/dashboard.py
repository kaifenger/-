import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import numpy as np
from storage import IoTDBStorage
from main import process_github_repo
from predictor import HealthMetricsPredictor
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Dash应用
app = dash.Dash(__name__)

# 定义应用布局
app.layout = html.Div([
    # 标题
    html.H1("开源项目健康度监控与预测仪表盘", style={'textAlign': 'center'}),
    
    # 输入区域
    html.Div([
        html.Label("输入GitHub仓库名 (格式: owner/repo):"),
        dcc.Input(
            id='repo-input',
            value='X-lab2017/open-digger',
            type='text',
            style={'width': '50%', 'margin': '10px'}
        ),
        html.Button('分析', id='submit-button', n_clicks=0)
    ], style={'textAlign': 'center', 'margin': '20px'}),
    # 健康度总览区域
    html.Div([
        # 左侧健康度趋势图
        html.Div([
            dcc.Graph(id='total-health')
        ], style={'width': '70%', 'display': 'inline-block'}),
        
        # 右侧健康度数值展示
        html.Div([
            html.Div([
                html.H4(id='current-time', style={'textAlign': 'center', 'color': '#666'}),
                html.H4("本月健康度", style={'textAlign': 'center'}),
                html.H2(id='current-score', style={'textAlign': 'center'}),
                html.H3(id='current-grade', style={'textAlign': 'center'}),
                html.Hr(),
                html.H4(id='next-month-title', style={'textAlign': 'center'}),
                html.H2(id='predicted-score', style={'textAlign': 'center'}),
                html.H3(id='predicted-grade', style={'textAlign': 'center'})
            ], style={'margin': '20px'})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'margin': '20px'}),   
    # 加载提示
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            # 图表区域
            html.Div([
                # 活跃度指标图表
                dcc.Graph(id='activity-metrics'),
                # 社区指标图表
                dcc.Graph(id='community-metrics'),
                # 影响力指标图表
                dcc.Graph(id='impact-metrics'),
                # 仓库完整性指标图表
                dcc.Graph(id='code-quality-metrics'),
                # 各维度分数图表
                dcc.Graph(id='dimension-scores')
            ])
        ]
    )
])

def load_data(repo_name):
    """从数据库加载仓库数据"""
    try:
        storage = IoTDBStorage()
        storage.connect()
        
        try:
            # 设置时间范围
            now = datetime.now(timezone.utc)
            end_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - relativedelta(months=12)
            next_month = end_time + relativedelta(months=1)
            
            # 查询历史数据
            metrics = storage.query_metrics(repo_name, start_time, end_time)
            scores = storage.query_metrics(f"{repo_name}_scores", start_time, end_time)
            
            # 获取预测月份并查询预测数据
            prediction_month = next_month.strftime('%Y_%m')
            prediction = storage.query_metrics(
                f"{repo_name}_{prediction_month}_prediction",
                end_time,
                next_month
            )
            
            # 转换为DataFrame
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index') if metrics else pd.DataFrame()
            scores_df = pd.DataFrame.from_dict(scores, orient='index') if scores else pd.DataFrame()
            prediction_df = pd.DataFrame.from_dict(prediction, orient='index') if prediction else pd.DataFrame()
            
            # 统一处理数据格式
            for df in [metrics_df, scores_df, prediction_df]:
                if not df.empty:
                    df.index = pd.to_datetime(df.index, unit='ms')
                    df.columns = df.columns.str.split('.').str[-1]
                    df = df.sort_index()
            
            logger.info(f"加载数据完成: metrics={len(metrics_df)}行, scores={len(scores_df)}行, prediction={len(prediction_df)}行")
            return metrics_df, scores_df, prediction_df
            
        finally:
            storage.close()
            
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
@app.callback(
    [Output('total-health', 'figure'),
     Output('current-time', 'children'),
     Output('current-score', 'children'),
     Output('current-grade', 'children'),
     Output('next-month-title', 'children'),
     Output('predicted-score', 'children'),
     Output('predicted-grade', 'children'),
     Output('activity-metrics', 'figure'),
     Output('community-metrics', 'figure'),
     Output('impact-metrics', 'figure'),
     Output('code-quality-metrics', 'figure'),
     Output('dimension-scores', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('repo-input', 'value')]
)


def update_graphs(n_clicks, repo_name):
    """更新所有图表"""
    try:
        if not repo_name:
            logger.warning("未输入仓库名")
            return [go.Figure()] * 12
            
        # 采集数据
        logger.info(f"开始处理仓库: {repo_name}")
        asyncio.run(process_github_repo(repo_name, months=12))
        
        # 步骤2: 获取当前时间范围
        now = datetime.now(timezone.utc)
        current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = current_month + relativedelta(months=1)
        prediction_month = next_month.strftime('%Y_%m')
        
        # 步骤3: 生成并存储预测数据
        predictor = HealthMetricsPredictor()
        predicted_scores = predictor.predict_next_month(repo_name)
        
        if predicted_scores:
            storage = IoTDBStorage()
            storage.connect()
            storage.store_metrics(
            repo=f"{repo_name}_{prediction_month}_prediction",
            metrics=predicted_scores,
            timestamp=next_month
            )
            logger.info("已生成并存储预测数据")
        
        # 加载数据
        metrics_df, scores_df, prediction_df = load_data(repo_name)
        
        if metrics_df.empty or scores_df.empty:
            logger.error(f"未找到仓库 {repo_name} 的数据")
            return [go.Figure()] * 12
            
        # 初始化主题
        light_theme = {
            'plot_bgcolor': '#f5f5f5',
            'paper_bgcolor': '#f5f5f5',
            'font': {'color': '#333333', 'size': 12},
            'title': {'font': {'color': '#333333', 'size': 14}},
            'xaxis': {
                'gridcolor': '#cccccc',  # 更深的网格线颜色
                'color': '#000000',      # 更深的坐标轴颜色
                'zerolinecolor': '#666666',
                'linewidth': 2,          # 加粗坐标轴
                'linecolor': '#666666',  # 坐标轴颜色
                'tickfont': {'size': 12} # 刻度标签字号
            },
            'yaxis': {
                'gridcolor': '#cccccc',
                'color': '#000000',
                'zerolinecolor': '#666666',
                'linewidth': 2,
                'linecolor': '#666666',
                'tickfont': {'size': 12}
            }
        }
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 格式化时间轴
        x_axis = metrics_df.index.strftime('%Y-%m')
        x_axis_scores = scores_df.index.strftime('%Y-%m')
        
        # 获取当前月份数据
        current_score = scores_df['total_score'].iloc[-1]
        current_grade = scores_df['health_grade'].iloc[-1]
        next_month = (scores_df.index[-1] + relativedelta(months=1)).strftime('%Y-%m')
        
        # 处理预测数据
        if not prediction_df.empty:
            prediction_df.columns = prediction_df.columns.str.split('.').str[-1]
            predicted_score = prediction_df['total_score'].iloc[0]
            predicted_grade = prediction_df['health_grade'].iloc[0]
            logger.info(f"预测数据: score={predicted_score}, grade={predicted_grade}")
        else:
            logger.warning("未找到预测数据")
            predicted_score = current_score
            predicted_grade = current_grade
            
        # 总体健康度图表
        total_health_fig = go.Figure()
        total_health_fig.add_trace(go.Scatter(
            x=x_axis_scores,
            y=scores_df['total_score'],
            name='历史健康度',
            mode='lines+markers',
            line={'color': '#4CAF50'}
        ))
        
        if not prediction_df.empty:
            total_health_fig.add_trace(go.Scatter(
                x=[next_month],
                y=[predicted_score],
                name='预测健康度',
                mode='markers',
                marker=dict(color='#FF4444', size=12, symbol='star')
            ))
            
        total_health_fig.update_layout(
            **light_theme,
            height=350,
            title_text="总体健康度趋势",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # 各维度分数图表
        dimension_fig = make_subplots(rows=2, cols=2, subplot_titles=(
            '活跃度得分（占比30%）', '社区得分（占比30%）', '影响力得分（占比30%）', '仓库完整性得分（占比10%）'
        ))
        
        metrics_pairs = [
            ('activity_score', '活跃度得分', 1, 1),
            ('community_score', '社区得分', 1, 2),
            ('impact_score', '影响力得分', 2, 1),
            ('code_quality_score', '仓库完整性得分', 2, 2)
        ]
        
        for metric, title, row, col in metrics_pairs:
            dimension_fig.add_trace(go.Scatter(
                x=x_axis_scores, y=scores_df[metric],
                name=f'{title}(线)', mode='lines+markers',
                line={'color': '#4DD0E1'}), row=row, col=col)
                
        dimension_fig.update_layout(**light_theme, height=400, showlegend=True, title_text="各维度分数变化趋势")
        
        for i in range(1, 3):
            for j in range(1, 3):
                dimension_fig.update_xaxes(gridcolor=light_theme['xaxis']['gridcolor'], 
                             linecolor=light_theme['xaxis']['linecolor'],
                             linewidth=light_theme['xaxis']['linewidth'], row=i, col=j)
                dimension_fig.update_yaxes(gridcolor=light_theme['yaxis']['gridcolor'],
                             linecolor=light_theme['yaxis']['linecolor'], 
                             linewidth=light_theme['yaxis']['linewidth'], row=i, col=j)
        # 活跃度指标图表
        activity_fig = make_subplots(rows=2, cols=2, subplot_titles=(
            '提交频率', '发布频率', 'Issue解决时间', 'PR解决时间'
        ))
        
        metrics_pairs = [
            ('commit_frequency', '提交频率', 1, 1),
            ('release_frequency', '发布频率', 1, 2),
            ('issue_resolution_time', 'Issue解决时间', 2, 1),
            ('pr_resolution_time', 'PR解决时间', 2, 2)
        ]
        
        for metric, title, row, col in metrics_pairs:
            activity_fig.add_trace(go.Bar(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(柱)', opacity=0.5,
                marker_color='#4CAF50'), row=row, col=col)
            activity_fig.add_trace(go.Scatter(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(线)', mode='lines+markers',
                line={'color': '#81C784'}), row=row, col=col)
                
        activity_fig.update_layout(**light_theme, height=400, showlegend=True, title_text="活跃度指标")
        for i in range(1, 3):
            for j in range(1, 3):
                activity_fig.update_xaxes(gridcolor=light_theme['xaxis']['gridcolor'],
                        linecolor=light_theme['xaxis']['linecolor'],
                        linewidth=light_theme['xaxis']['linewidth'], row=i, col=j)
                activity_fig.update_yaxes(gridcolor=light_theme['yaxis']['gridcolor'],
                        linecolor=light_theme['yaxis']['linecolor'],
                        linewidth=light_theme['yaxis']['linewidth'], row=i, col=j)
        
        # 社区指标图表
        community_fig = make_subplots(rows=2, cols=3, subplot_titles=(
            '贡献者总数', '贡献者增长率', '维护者保持率',
            '语言多样性', '响应时间'
        ))
        
        metrics_pairs = [
            ('total_contributors', '贡献者总数', 1, 1),
            ('contributor_growth', '贡献者增长率', 1, 2),
            ('maintainer_retention', '维护者保持率', 1, 3),
            ('language_diversity', '语言多样性', 2, 1),
            ('response_time', '响应时间', 2, 2)
        ]
        
        for metric, title, row, col in metrics_pairs:
            community_fig.add_trace(go.Bar(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(柱)', opacity=0.5,
                marker_color='#2196F3'), row=row, col=col)
            community_fig.add_trace(go.Scatter(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(线)', mode='lines+markers',
                line={'color': '#64B5F6'}), row=row, col=col)
                
        community_fig.update_layout(**light_theme, height=400, showlegend=True, title_text="社区指标")
        for i in range(1, 3):
            for j in range(1, 4):
                community_fig.update_xaxes(gridcolor=light_theme['xaxis']['gridcolor'],
                             linecolor=light_theme['xaxis']['linecolor'],
                             linewidth=light_theme['xaxis']['linewidth'], row=i, col=j)
                community_fig.update_yaxes(gridcolor=light_theme['yaxis']['gridcolor'],
                             linecolor=light_theme['yaxis']['linecolor'],
                             linewidth=light_theme['yaxis']['linewidth'], row=i, col=j)

        # 影响力指标图表
        impact_fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Stars总数', 'Stars增长率', 'Forks总数', 'Forks增长率'
        ))
        
        metrics_pairs = [
            ('stars_count', 'Stars总数', 1, 1),
            ('stars_growth', 'Stars增长率', 1, 2),
            ('forks_count', 'Forks总数', 2, 1),
            ('forks_growth', 'Forks增长率', 2, 2)
        ]
        
        for metric, title, row, col in metrics_pairs:
            impact_fig.add_trace(go.Bar(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(柱)', opacity=0.5,
                marker_color='#FFC107'), row=row, col=col)
            impact_fig.add_trace(go.Scatter(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(线)', mode='lines+markers',
                line={'color': '#FFD54F'}), row=row, col=col)
                
        impact_fig.update_layout(**light_theme, height=400, showlegend=True, title_text="影响力指标")
        for i in range(1, 3):
            for j in range(1, 3):
                impact_fig.update_xaxes(gridcolor=light_theme['xaxis']['gridcolor'],
                            linecolor=light_theme['xaxis']['linecolor'],
                            linewidth=light_theme['xaxis']['linewidth'], row=i, col=j)
                impact_fig.update_yaxes(gridcolor=light_theme['yaxis']['gridcolor'],
                            linecolor=light_theme['yaxis']['linecolor'],
                            linewidth=light_theme['yaxis']['linewidth'], row=i, col=j)
        
        # 仓库完整性指标图表
        quality_fig = make_subplots(rows=1, cols=3, subplot_titles=(
            '测试覆盖率', '文档完整度', 'CI/CD得分'
        ))
        
        metrics_pairs = [
            ('test_coverage', '测试覆盖率', 1, 1),
            ('documentation_score', '文档完整度', 1, 2),
            ('cicd_score', 'CI/CD得分', 1, 3)
        ]
        
        for metric, title, row, col in metrics_pairs:
            quality_fig.add_trace(go.Bar(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(柱)', opacity=0.5,
                marker_color='#9C27B0'), row=row, col=col)
            quality_fig.add_trace(go.Scatter(
                x=x_axis, y=metrics_df[metric],
                name=f'{title}(线)', mode='lines+markers',
                line={'color': '#BA68C8'}), row=row, col=col)
                
        quality_fig.update_layout(**light_theme, height=300, showlegend=True, title_text="仓库完整性指标")
        for j in range(1, 4):
            quality_fig.update_xaxes(gridcolor=light_theme['xaxis']['gridcolor'],
                       linecolor=light_theme['xaxis']['linecolor'],
                       linewidth=light_theme['xaxis']['linewidth'], row=1, col=j)
            quality_fig.update_yaxes(gridcolor=light_theme['yaxis']['gridcolor'],
                       linecolor=light_theme['yaxis']['linecolor'],
                       linewidth=light_theme['yaxis']['linewidth'], row=1, col=j)

        return (
            total_health_fig,
            f"更新时间: {current_time}",
            f"{current_score:.2f}",
            f"健康度等级：{current_grade}",
            f"预测 {next_month} 健康度",
            f"{predicted_score:.2f}",
            f"预测等级：{predicted_grade}",
            dimension_fig,
            activity_fig,
            community_fig,
            impact_fig,
            quality_fig       
        )
        
    except Exception as e:
        logger.error(f"更新图表失败: {str(e)}", exc_info=True)
        return [go.Figure()] * 12
if __name__ == '__main__':
    app.run_server(debug=True)