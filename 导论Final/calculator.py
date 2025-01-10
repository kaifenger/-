from typing import Dict, Any
import logging
from datetime import datetime, timezone
from models import ProjectHealth
import math
logger = logging.getLogger(__name__)

class HealthScoreCalculator:
    WEIGHTS = {
        'community': 0.30,     # 社区基础
        'activity': 0.30,      # 活跃程度
        'impact': 0.30,        # 技术影响力
        'code_quality': 0.10,  # 仓库完整性得分
    }

    @staticmethod
    def normalize_score(score: float) -> float:
        """将分数归一化到0-100范围"""
        return max(0, min(100, score))

    @staticmethod
    def calculate_activity_score(metrics: Dict[str, Any]) -> float:
        """计算活跃度得分"""
        weights = {
            'commit_frequency': 0.25,
            'release_frequency': 0.25,
            'issue_resolution_time': 0.25,
            'pr_resolution_time': 0.25,
        }
        
        # 合理的得分计算
        scores = {
            'commit_frequency':  min(metrics['commit_frequency'] * (100/15), 100),  # 15次/月满
            'release_frequency': min(metrics['release_frequency'] * (100/20), 100), # 20次/月满
            'issue_resolution_time': 100 * (1 / (1 + metrics['issue_resolution_time']/7)),  # 7天内处理最佳
            'pr_resolution_time': 100 * (1 / (1 + metrics['pr_resolution_time']/7)),  # 7天内处理最佳
        }
        
        # 归一化总分
        total_score = sum(scores[key] * weights[key] for key in weights)
        return HealthScoreCalculator.normalize_score(total_score)

    @staticmethod
    def calculate_community_score(metrics: Dict[str, Any]) -> float:
        weights = {
            'maintainer_retention': 0.35,  
            'contributor_growth': 0.15,        
            'total_contributors': 0.35,         
            'response_time': 0.10,              
            'language_diversity': 0.05          
        }
        
        scores = {
            'maintainer_retention': metrics['maintainer_retention'] * 100,
            'contributor_growth': min(metrics['contributor_growth'] * 1000, 100),  # 10%增长率满分
            'total_contributors': min(math.log(max(metrics['total_contributors'], 1), 2) * 20, 100),  # 64人达到满分，32人80分，16人60分。。。
            'response_time': 100 * (1 / (1 + metrics['response_time']/48)),  # 48小时内响应最佳
            'language_diversity': metrics['language_diversity'] * 100
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return HealthScoreCalculator.normalize_score(total_score)

    @staticmethod
    def calculate_impact_score(metrics: Dict[str, Any]) -> float:
        weights = {
            'stars_growth': 0.20,
            'forks_growth': 0.20,
            'stars_count': 0.30,
            'forks_count': 0.30
        }
        
        scores = {
        # 增长率评分
        'stars_growth': min(metrics['stars_growth'] * 1000, 100),  # 10%增长率满分
        'forks_growth': min(metrics['forks_growth'] * 1000, 100),  # 10%增长率满分
        # 数量评分基准
        'stars_count': min(math.log(max(metrics['stars_count'], 1), 10) * 25, 100),#得分 = min(log10(指标数量) * 25, 100)
        'forks_count': min(math.log(max(metrics['forks_count'], 1), 10) * 25, 100)
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return HealthScoreCalculator.normalize_score(total_score)

    @staticmethod
    def calculate_code_quality_score(metrics: Dict[str, Any]) -> float:
        weights = {
            'documentation_score': 0.35,
            'test_coverage': 0.35,
            'cicd_score': 0.30
        }
        
        scores = {
            'documentation_score': metrics['documentation_score'] * 100,
            'test_coverage': metrics['test_coverage'] * 100,
            'cicd_score': metrics['cicd_score'] * 100
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return HealthScoreCalculator.normalize_score(total_score)
    
    @staticmethod
    def get_health_grade(score: float) -> str:
        """根据得分返回健康度等级"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 40:
            return 'C'
        else:
            return 'D'

    def calculate_total_score(self, health_metrics: ProjectHealth) -> Dict[str, Any]:
        """计算总健康度分数和等级"""
        if health_metrics.timestamp and health_metrics.timestamp.tzinfo is None:
            health_metrics.timestamp = health_metrics.timestamp.replace(tzinfo=timezone.utc)
        # 计算各维度得分
        dimension_scores = {
            'activity': self.calculate_activity_score(health_metrics.activity.model_dump()),
            'community': self.calculate_community_score(health_metrics.community.model_dump()),
            'impact': self.calculate_impact_score(health_metrics.impact.model_dump()),
            'code_quality': self.calculate_code_quality_score(health_metrics.code_quality.model_dump()),
        }
        
        # 计算每个维度的等级
        dimension_grades = {
            name: self.get_health_grade(score) 
            for name, score in dimension_scores.items()
        }
        
        # 计算加权总分
        total_score = sum(
            dimension_scores[key] * self.WEIGHTS[key] 
            for key in self.WEIGHTS
        )
        
        # 获取整体健康度等级
        health_grade = self.get_health_grade(total_score)
        
        # 返回详细的评分结果
        return {
            'total_score': round(total_score, 2),
            'health_grade': health_grade,
            'dimensions': {
                name: {
                    'score': round(dimension_scores[name], 2),
                    'grade': dimension_grades[name],
                    'weight': self.WEIGHTS[name]
                }
                for name in dimension_scores
            }
        }
    