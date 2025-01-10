import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from typing import Dict, Any
import logging
from storage import IoTDBStorage
from calculator import HealthScoreCalculator
from models import ProjectHealth, ActivityMetrics, CommunityMetrics, ImpactMetrics, CodeQualityMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthMetricsPredictor:
    def __init__(self):
        self.storage = IoTDBStorage()
        self.calculator = HealthScoreCalculator()
        self.predicted_metrics = None
        
    def _predict_trend(self, data: pd.Series, metric_name: str) -> float:
        """根据指标类型使用不同的预测策略"""
        try:
            if len(data) < 2:
                return float(data.iloc[-1]) if len(data) > 0 else 0.0
                
            # 数据类型转换和清理
            data = pd.to_numeric(data, errors='coerce')
            data = data.dropna()
            
            if len(data) < 2:
                return float(data.iloc[-1]) if len(data) > 0 else 0.0
                
            # 累积型指标列表
            cumulative_metrics = ['total_contributors', 'stars_count', 'forks_count']
            
            if metric_name in cumulative_metrics:
                # 对于累积型指标，确保预测值不小于最后一个实际值
                last_value = float(data.iloc[-1])
                # 使用最近3个月的平均增长量
                recent_growth = data.diff().tail(3).mean()
                return max(last_value + recent_growth, last_value)
            else:
                # 其他指标使用线性回归
                X = np.arange(len(data), dtype=np.float64).reshape(-1, 1)
                y = np.array(data.values, dtype=np.float64)
                
                try:
                    A = np.vstack([X.ravel(), np.ones(len(X))]).T
                    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                    next_value = slope * len(data) + intercept
                    return max(0.0, float(next_value))
                    
                except np.linalg.LinAlgError:
                    return float(data.iloc[-1])
                    
        except Exception as e:
            logger.error(f"趋势预测失败: {str(e)}")
            return float(data.iloc[-1]) if len(data) > 0 else 0.0
    
    def predict_next_month(self, repo: str) -> Dict[str, Any]:
        """预测下一个月的健康度指标"""
        try:
            self.storage.connect()
            
            # 获取当前时间和时间范围
            now = datetime.now(timezone.utc)
            end_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - relativedelta(months=12)
            
            # 查询历史数据
            metrics = self.storage.query_metrics(repo, start_time, end_time)
            
            if not metrics:
                raise Exception("没有足够的历史数据用于预测")
                
            # 转换为DataFrame并清理列名
            df = pd.DataFrame.from_dict(metrics, orient='index')
            df.index = pd.to_datetime(df.index, unit='ms')
            df.columns = df.columns.str.split('.').str[-1]
            
            # 预测下一个月的指标
            next_month = end_time + relativedelta(months=1)
            predicted_metrics = {}
            
            # 整数类型的指标
            integer_metrics = ['total_contributors', 'stars_count', 'forks_count']
            
            for column in df.columns:
                if column != 'repo_name':
                    predicted_value = self._predict_trend(df[column],column)
                    # 对整数类型指标取整
                    if column in integer_metrics:
                        predicted_value = int(round(predicted_value))
                    predicted_metrics[column] = predicted_value
            
            # 分类并规范化指标
            activity_metrics = {
                'commit_frequency': float(predicted_metrics.get('commit_frequency', 0)),
                'release_frequency': float(predicted_metrics.get('release_frequency', 0)),
                'issue_resolution_time': float(predicted_metrics.get('issue_resolution_time', 0)),
                'pr_resolution_time': float(predicted_metrics.get('pr_resolution_time', 0))
            }
            
            community_metrics = {
                'total_contributors': int(predicted_metrics.get('total_contributors', 0)),
                'contributor_growth': float(predicted_metrics.get('contributor_growth', 0)),
                'maintainer_retention': float(predicted_metrics.get('maintainer_retention', 0)),
                'language_diversity': float(predicted_metrics.get('language_diversity', 0)),
                'response_time': float(predicted_metrics.get('response_time', 0))
            }
            
            impact_metrics = {
                'stars_count': int(predicted_metrics.get('stars_count', 0)),
                'stars_growth': float(predicted_metrics.get('stars_growth', 0)),
                'forks_count': int(predicted_metrics.get('forks_count', 0)),
                'forks_growth': float(predicted_metrics.get('forks_growth', 0))
            }
            
            code_quality_metrics = {
                'test_coverage': float(predicted_metrics.get('test_coverage', 0)),
                'documentation_score': float(predicted_metrics.get('documentation_score', 0)),
                'cicd_score': float(predicted_metrics.get('cicd_score', 0))
            }
            
            # 创建ProjectHealth对象
            health_metrics = ProjectHealth(
                repo_name=repo,
                timestamp=next_month,
                activity=ActivityMetrics(**activity_metrics),
                community=CommunityMetrics(**community_metrics),
                impact=ImpactMetrics(**impact_metrics),
                code_quality=CodeQualityMetrics(**code_quality_metrics)
            )
            
            # 计算健康度分数
            scores = self.calculator.calculate_total_score(health_metrics)
            
            # 合并预测指标和分数
            complete_metrics = {
                **activity_metrics,
                **community_metrics,
                **impact_metrics,
                **code_quality_metrics,
                'repo_name': repo
            }
            
            # 存储预测指标
            self.storage.store_metrics(repo=repo, metrics=complete_metrics, timestamp=next_month)
            
            # 存储预测分数
            score_data = {
                'total_score': scores['total_score'],
                'health_grade': scores['health_grade'],
                **{f"{dim}_score": details['score'] for dim, details in scores['dimensions'].items()},
                **{f"{dim}_grade": details['grade'] for dim, details in scores['dimensions'].items()}
            }
            
            self.storage.store_metrics(
                repo=f"{repo}_scores",
                metrics=score_data,
                timestamp=next_month
            )
            
            # 保存完整预测结果
            self.predicted_metrics = {
                'timestamp': next_month,
                'repo_name': repo,
                **complete_metrics,
                **score_data
            }
            
            return score_data
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
        finally:
            self.storage.close()
    def get_predicted_metrics(self) -> Dict[str, Any]:
        """获取最近一次预测的指标数据"""
        return self.predicted_metrics if self.predicted_metrics else {}