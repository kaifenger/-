# models.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ActivityMetrics(BaseModel):
    commit_frequency: float
    release_frequency: float
    issue_resolution_time: float
    pr_resolution_time: float

class CommunityMetrics(BaseModel):
    total_contributors: int
    contributor_growth: float
    maintainer_retention: float

    response_time: float
    language_diversity: float

class ImpactMetrics(BaseModel):
    stars_count: int
    stars_growth: float
    forks_count: int
    forks_growth: float


class CodeQualityMetrics(BaseModel):

    test_coverage: float
    documentation_score: float
    cicd_score: float



class ProjectHealth(BaseModel):
    repo_name: str
    timestamp: datetime
    activity: ActivityMetrics
    community: CommunityMetrics
    impact: ImpactMetrics
    code_quality: CodeQualityMetrics
