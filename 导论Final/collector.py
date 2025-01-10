import aiohttp
import asyncio
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Optional, AsyncGenerator
import json
import math
from models import (ActivityMetrics, CommunityMetrics, ImpactMetrics,
                   CodeQualityMetrics,  ProjectHealth)
import re
import time
import base64
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil.relativedelta import relativedelta
import pandas as pd

class HealthMetricsCollector:
    def __init__(self, token: str = None):
        self.headers = {
            'Authorization': f'token {token}' if token else '',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.batch_size = 4  # 增加批次大小
        self.cache = {}  # 添加缓存
        self.max_concurrent_requests = 10  # 最大并发请求数
        self.semaphore = None  # 限制并发
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        try:
            _headers = {**self.headers}
            if headers:
                _headers.update(headers)
                
            async with self.semaphore:
                async with self.session.get(url, params=params, headers=_headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 403:
                        # Handle rate limiting
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        if reset_time:
                            wait_time = max(reset_time - time.time(), 0)
                            logger.info(f"Rate limited, waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)
                        raise Exception("Rate limit exceeded")
                    elif response.status == 404:
                        return {}
                    else:
                        raise Exception(f"API request failed: {response.status}")
        except Exception as e:
            logger.error(f"请求失败 {url}: {str(e)}")
            return {}

    async def _cached_request(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        try:
            cache_key = f"{url}_{str(params)}_{str(headers)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            data = await self._make_request(url, params, headers)
            if data:
                self.cache[cache_key] = data
            return data
        except Exception as e:
            if "404" in str(e):
                # Return empty dict for not found instead of None
                return {}
            raise
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.session.close()
    

            

#收集活跃度指标数据
    async def _calculate_commit_pattern(self, commit_dates: List[datetime], target_date: datetime) -> float:
        """计算提交模式规律性得分"""
        try:
            if not commit_dates:
                return 0.0
                
            # 过滤target_date之前的提交
            valid_dates = [d for d in commit_dates if d <= target_date]
            if not valid_dates:
                return 0.0
            
            # 1. 计算工作日/周末比例
            weekday_commits = sum(1 for d in valid_dates if d.weekday() < 5)
            total_commits = len(valid_dates)
            weekday_ratio = weekday_commits / total_commits if total_commits > 0 else 0
            
            # 2. 计算提交间隔
            sorted_dates = sorted(valid_dates)
            intervals = [(sorted_dates[i+1] - sorted_dates[i]).days 
                        for i in range(len(sorted_dates)-1)]
            
            if not intervals:
                return 0.4 * (1 - abs(weekday_ratio - 5/7))
            
            # 3. 计算间隔标准差
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            std_dev = math.sqrt(variance)
            
            # 4. 生成最终得分
            regularity_score = 0.6 * math.exp(-std_dev/30)
            balance_score = 0.4 * (1 - abs(weekday_ratio - 5/7))
            
            return regularity_score + balance_score
            
        except Exception as e:
            logger.error(f"计算提交模式得分失败: {str(e)}")
            return 0.0



    async def _get_commit_frequency(self, repo: str, target_date: datetime = None) -> float:
        """计算commit频率 (每月)"""
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            commits_url = f"https://api.github.com/repos/{repo}/commits"
            params = {'per_page': 100}
            
            commit_dates = []
            page = 1
            
            while True:
                current_params = {**params, 'page': page}
                async with self.session.get(commits_url, params=current_params) as response:
                    if response.status == 200:
                        commits = await response.json()
                        if not commits:
                            break
                            
                        for commit in commits:
                            if commit_date := commit.get('commit', {}).get('committer', {}).get('date'):
                                date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                                if date <= target_date:
                                    commit_dates.append(date)
                                    
                        if len(commits) < 100:
                            break
                        page += 1
                    else:
                        break
                        
            await asyncio.sleep(1)
        
            if len(commit_dates) < 2:
                return 0.0
                
            # 计算月频率
            time_span = (max(commit_dates) - min(commit_dates)).days
            if time_span <= 0:
                return 0.0
                
            monthly_frequency = (len(commit_dates) / time_span) * 30
            return round(monthly_frequency, 2)  # 保留两位小数
            
        except Exception as e:
            logger.error(f"计算commit频率失败: {str(e)}")
            return 0.0
    async def _get_release_frequency(self, repo: str, target_date: datetime = None) -> float:
        """计算发布频率 (每月)"""
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            releases_url = f"https://api.github.com/repos/{repo}/releases"
            params = {'per_page': 100}
            
            release_dates = []
            page = 1
            
            while True:
                current_params = {**params, 'page': page}
                async with self.session.get(releases_url, params=current_params) as response:
                    if response.status == 200:
                        releases = await response.json()
                        if not releases:
                            break
                            
                        for release in releases:
                            if published_at := release.get('published_at'):
                                date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                if date <= target_date:
                                    release_dates.append(date)
                                    
                        if len(releases) < 100:
                            break
                        page += 1
                    else:
                        break
                        
                await asyncio.sleep(1)
            
            if len(release_dates) < 2:
                return 0.0
                
            # 计算月频率
            time_span = (max(release_dates) - min(release_dates)).days
            if time_span <= 0:
                return 0.0
                
            monthly_frequency = (len(release_dates) / time_span) * 30
            return min(monthly_frequency, 10.0)  # 限制最大值
            
        except Exception as e:
            logger.error(f"计算发布频率失败: {str(e)}")
            return 0.0

    async def _get_issue_resolution_time(self, repo: str, target_date: datetime = None) -> float:
        """计算issue平均解决时间 (天)"""
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            issues_url = f"https://api.github.com/repos/{repo}/issues"
            params = {
                'state': 'closed',
                'per_page': 100
            }
            
            resolution_times = []
            page = 1
            
            while True:
                current_params = {**params, 'page': page}
                async with self.session.get(issues_url, params=current_params) as response:
                    if response.status == 200:
                        issues = await response.json()
                        if not issues:
                            break
                            
                        for issue in issues:
                            if not issue.get('pull_request'):  # 排除PR
                                created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                                closed = datetime.fromisoformat(issue['closed_at'].replace('Z', '+00:00'))
                                if closed <= target_date:
                                    days = (closed - created).days
                                    if 0 <= days <= 365:  # 排除异常值
                                        resolution_times.append(days)
                                        
                        if len(issues) < 100:
                            break
                        page += 1
                    else:
                        break
                        
                await asyncio.sleep(1)
            
            return sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
            
        except Exception as e:
            logger.error(f"计算issue解决时间失败: {str(e)}")
            return 0.0

    async def _get_pr_resolution_time(self, repo: str, target_date: datetime) -> float:
        """计算PR平均处理时间 (天)"""
        try:
            pulls_url = f"https://api.github.com/repos/{repo}/pulls"
            params = {
                'state': 'closed',
                'per_page': 100
            }
            
            resolution_times = []
            page = 1
            
            while True:
                current_params = {**params, 'page': page}
                async with self.session.get(pulls_url, params=current_params) as response:
                    if response.status == 200:
                        prs = await response.json()
                        if not prs:
                            break
                            
                        for pr in prs:
                            created = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                            closed = datetime.fromisoformat(pr['closed_at'].replace('Z', '+00:00'))
                            if closed <= target_date:
                                days = (closed - created).days
                                if 0 <= days <= 365:  # 排除异常值
                                    resolution_times.append(days)
                                    
                        if len(prs) < 100:
                            break
                        page += 1
                    else:
                        break
                        
                await asyncio.sleep(1)
            
            return sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
            
        except Exception as e:
            logger.error(f"计算PR处理时间失败: {str(e)}")
            return 0.0

    async def collect_activity_metrics(self, repo: str, target_date: datetime = None) -> dict:
        """收集指定时间点的活跃度指标数据
        
        Args:
            repo: 仓库名称 (owner/repo)
            target_date: 目标日期
            
        Returns:
            包含活跃度指标的字典
        """
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            # 并发调用所有活跃度相关函数
            release_freq, issue_time, pr_time, commit_freq = await asyncio.gather(

                self._get_release_frequency(repo, target_date),
                self._get_issue_resolution_time(repo, target_date),
                self._get_pr_resolution_time(repo, target_date),
                self._get_commit_frequency(repo, target_date)
            )
            
            # 返回活跃度指标数据
            return {

                'release_frequency': release_freq,
                'issue_resolution_time': issue_time,
                'pr_resolution_time': pr_time,
                'commit_frequency': commit_freq
            }
            
        except Exception as e:
            logger.error(f"收集活跃度指标失败: {str(e)}")
            return {}
#收集社区指标数据
    async def _get_total_contributors(self, repo: str, until_date: datetime) -> int:
        """获取指定日期前的贡献者数量"""
        try:
            url = f"https://api.github.com/repos/{repo}/commits"
            params = {
                'until': until_date.isoformat(),
                'per_page': 100
            }
            
            # 获取所有提交者
            contributors = set()
            page = 1
            
            while True:
                current_params = {**params, 'page': page}
                async with self.session.get(url, params=current_params) as response:
                    if response.status == 200:
                        commits = await response.json()
                        if not commits:
                            break
                            
                        # 收集提交者
                        for commit in commits:
                            if commit.get('author') and commit['author'].get('login'):
                                contributors.add(commit['author']['login'])
                                
                        if len(commits) < 100:
                            break
                        page += 1
                    else:
                        logger.error(f"Commits API请求失败: {response.status}")
                        return 0
                        
                await asyncio.sleep(1)
                
            return len(contributors)
                    
        except Exception as e:
            logger.error(f"获取贡献者数量失败: {str(e)}")
            return 0
        
    async def _calculate_contributor_growth(self, repo: str, target_date: datetime) -> float:
        """计算贡献者增长率"""
        try:
            current_count = await self._get_total_contributors(repo, target_date)
            past_date = target_date - relativedelta(months=1)
            past_count = await self._get_total_contributors(repo, past_date)
            
            if past_count == 0:
                return 0.0
            return (current_count - past_count) / past_count
        except Exception as e:
            logger.error(f"计算贡献者增长率失败: {str(e)}")
            return 0.0

    async def _get_maintainer_retention(self, repo: str, target_date: datetime) -> float:
        """计算维护者留存率"""
        try:
            # 获取6个月前的维护者列表
            past_date = target_date - relativedelta(months=6)
            url = f"https://api.github.com/repos/{repo}/contributors"
            params = {'per_page': 100}
            
            past_maintainers = set()
            current_maintainers = set()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    contributors = await response.json()
                    # 取前20%作为核心维护者
                    core_count = max(1, len(contributors) // 5)
                    for i in range(core_count):
                        if i < len(contributors):
                            past_maintainers.add(contributors[i]['login'])
            
            # 获取当前维护者
            params['until'] = target_date.isoformat()
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    contributors = await response.json()
                    core_count = max(1, len(contributors) // 5)
                    for i in range(core_count):
                        if i < len(contributors):
                            current_maintainers.add(contributors[i]['login'])
            
            # 计算留存率
            if not past_maintainers:
                return 0.0
            retained = len(past_maintainers.intersection(current_maintainers))
            return retained / len(past_maintainers)
            
        except Exception as e:
            logger.error(f"计算维护者留存率失败: {str(e)}")
            return 0.0

    async def _get_language_diversity(self, repo: str, target_date: datetime) -> float:
        """计算语言多样性"""
        try:
            url = f"https://api.github.com/repos/{repo}/languages"
            data = await self._cached_request(url)
            
            if not data:
                return 0.0
                
            # 计算语言比例的熵
            total = sum(data.values())
            if total == 0:
                return 0.0
                
            proportions = [count/total for count in data.values()]
            entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
            
            # 归一化到0-1
            max_entropy = math.log2(len(data))
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算语言多样性失败: {str(e)}")
            return 0.0

    async def _get_average_response_time(self, repo: str, target_date: datetime) -> float:
        """计算平均响应时间(小时)"""
        try:
            issues_url = f"https://api.github.com/repos/{repo}/issues"
            params = {
                'state': 'all',
                'per_page': 100,
                'since': (target_date - relativedelta(months=1)).isoformat()
            }
            
            response_times = []
            
            async with self.session.get(issues_url, params=params) as response:
                if response.status == 200:
                    issues = await response.json()
                    for issue in issues:
                        created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                        if issue.get('comments', 0) > 0:
                            comments_url = issue['comments_url']
                            async with self.session.get(comments_url) as comment_response:
                                if comment_response.status == 200:
                                    comments = await comment_response.json()
                                    if comments:
                                        first_response = datetime.fromisoformat(
                                            comments[0]['created_at'].replace('Z', '+00:00')
                                        )
                                        hours = (first_response - created_at).total_seconds() / 3600
                                        if 0 <= hours <= 720:  # 最多30天
                                            response_times.append(hours)
            
            return sum(response_times) / len(response_times) if response_times else 0.0
            
        except Exception as e:
            logger.error(f"计算平均响应时间失败: {str(e)}")
            return 0.0

    async def _collect_community_metrics(self, repo: str, target_date: datetime = None) -> dict:
        """收集指定时间点的社区指标数据
        
        Args:
            repo: 仓库名称 (owner/repo)
            target_date: 目标日期
        
        Returns:
            包含社区指标的字典
        """
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            # 并发调用所有社区指标相关函数
            total_contribs, growth_rate, retention, lang_div, resp_time = await asyncio.gather(
                self._get_total_contributors(repo, target_date),
                self._calculate_contributor_growth(repo, target_date),
                self._get_maintainer_retention(repo, target_date),
                self._get_language_diversity(repo, target_date),
                self._get_average_response_time(repo, target_date)
            )
            
            # 返回社区指标数据
            return {
                'total_contributors': total_contribs,
                'contributor_growth': growth_rate,
                'maintainer_retention': retention,
                'language_diversity': lang_div,
                'response_time': resp_time
            }
            
        except Exception as e:
            logger.error(f"收集社区指标失败: {str(e)}")
            return {
                'total_contributors': 0,
                'contributor_growth': 0.0,
                'maintainer_retention': 0.0,
                'language_diversity': 0.0,
                'response_time': 0.0
            }
#收集影响力指标数据
    async def get_stars_count(self, repo: str, until_date: datetime) -> int:
        """获取指定日期前的stars数量"""
        try:
            url = f"https://api.github.com/repos/{repo}/stargazers"
            params = {
                'per_page': 100,
                'page': 1
            }
            headers = {
                **self.headers,
                'Accept': 'application/vnd.github.star+json'
            }
            
            total_stars = 0
            
            while True:
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        stars = await response.json()
                        # 只计算until_date之前的stars
                        valid_stars = [
                            s for s in stars 
                            if datetime.fromisoformat(s['starred_at'].replace('Z', '+00:00')) <= until_date
                        ]
                        total_stars += len(valid_stars)
                        
                        if len(stars) < 100 or len(valid_stars) < len(stars):
                            break
                        params['page'] += 1
                    else:
                        logger.error(f"Stars API请求失败: {response.status}")
                        return 0
                        
                await asyncio.sleep(1)
                
            return total_stars
            
        except Exception as e:
            logger.error(f"获取stars数量失败: {str(e)}")
            return 0

    async def _calculate_stars_growth(self, repo: str, target_date: datetime) -> float:
        """计算stars增长率"""
        try:
            current_stars = await self.get_stars_count(repo, target_date)
            past_date = target_date - relativedelta(months=1)
            past_stars = await self.get_stars_count(repo, past_date)
            
            if past_stars == 0:
                return 0.0
            return (current_stars - past_stars) / past_stars
        except Exception as e:
            logger.error(f"计算stars增长率失败: {str(e)}")
            return 0.0

    async def _get_forks_count(self, repo: str, until_date: datetime) -> int:
        """获取指定日期前的forks数量"""
        try:
            url = f"https://api.github.com/repos/{repo}/forks"
            params = {
                'per_page': 100,
                'page': 1
            }
            
            total_forks = 0
            while True:
                data = await self._cached_request(url, params)
                if not data:
                    break
                    
                # 筛选日期范围内的forks
                valid_forks = [
                    f for f in data 
                    if datetime.fromisoformat(f['created_at'].replace('Z', '+00:00')) <= until_date
                ]
                total_forks += len(valid_forks)
                
                if len(data) < 100 or len(valid_forks) < len(data):
                    break
                params['page'] += 1
                
            return total_forks
            
        except Exception as e:
            logger.error(f"获取forks数量失败: {str(e)}")
            return 0

    async def _calculate_forks_growth(self, repo: str, target_date: datetime) -> float:
        """计算forks增长率"""
        try:
            current_forks = await self._get_forks_count(repo, target_date)
            past_date = target_date - relativedelta(months=1)
            past_forks = await self._get_forks_count(repo, past_date)
            
            if past_forks == 0:
                return 0.0
            return (current_forks - past_forks) / past_forks
        except Exception as e:
            logger.error(f"计算forks增长率失败: {str(e)}")
            return 0.0

    async def _collect_impact_metrics(self, repo: str, target_date: datetime = None) -> dict:
        """收集指定时间点的影响力指标数据
        
        Args:
            repo: 仓库名称 (owner/repo)
            target_date: 目标日期
        
        Returns:
            包含影响力指标的字典
        """
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                
            # 并发调用所有影响力指标相关函数
            stars_count, stars_growth, forks_count, forks_growth = await asyncio.gather(
                self.get_stars_count(repo, target_date),
                self._calculate_stars_growth(repo, target_date),
                self._get_forks_count(repo, target_date),
                self._calculate_forks_growth(repo, target_date)
            )
            
            # 返回影响力指标数据
            return {
                'stars_count': stars_count,
                'stars_growth': stars_growth,
                'forks_count': forks_count,
                'forks_growth': forks_growth
            }
            
        except Exception as e:
            logger.error(f"收集影响力指标失败: {str(e)}")
            return {
                'stars_count': 0,
                'stars_growth': 0.0,
                'forks_count': 0,
                'forks_growth': 0.0
            }                       
#代码质量指标数据

    async def _get_test_coverage(self, repo: str, target_date: datetime) -> float:
        """评估测试覆盖率"""
        try:
            contents_url = f"https://api.github.com/repos/{repo}/contents"
            data = await self._cached_request(contents_url)
            
            if not data:
                return 0.0
            
            # 检查常见的测试目录
            test_dirs = ['tests', 'test', '__tests__', 'spec', 'specs']
            has_tests = any(item['name'].lower() in test_dirs for item in data)
            
            # 如果有测试目录,给一个基础分
            base_score = 0.5 if has_tests else 0.0
            
            # 检查CI配置中的测试相关配置
            ci_files = ['.travis.yml', '.github/workflows/test.yml', 'circle.yml']
            ci_score = 0.0
            
            for ci_file in ci_files:
                ci_data = await self._cached_request(f"{contents_url}/{ci_file}")
                if ci_data:
                    ci_score += 0.1667  # 每个CI文件最高0.5分
                    
            return min(base_score + ci_score, 1.0)
            
        except Exception as e:
            logger.error(f"评估测试覆盖率失败: {str(e)}")
            return 0.0

    async def _calculate_documentation_score(self, repo: str, target_date: datetime) -> float:
        """评估文档完整性"""
        try:
            docs = ['README.md', 'CONTRIBUTING.md', 'docs/', 'wiki']
            score = 0.0
            
            for doc in docs:
                if doc == 'wiki':
                    url = f"https://api.github.com/repos/{repo}/wiki"
                else:
                    url = f"https://api.github.com/repos/{repo}/contents/{doc}"
                    
                data = await self._cached_request(url)
                if data:
                    score += 0.25
                    
            return score
            
        except Exception as e:
            logger.error(f"评估文档完整性失败: {str(e)}")
            return 0.0

    async def _evaluate_cicd(self, repo: str, target_date: datetime) -> float:
        """评估CI/CD完整性"""
        try:
            workflows_url = f"https://api.github.com/repos/{repo}/actions/workflows"
            data = await self._cached_request(workflows_url)
            
            if not data:
                return 0.0
                
            expected = {'build', 'test', 'deploy', 'release'}
            found = set()
            
            for workflow in data.get('workflows', []):
                name = workflow['name'].lower()
                for key in expected:
                    if key in name:
                        found.add(key)
                        
            return len(found) / len(expected)
            
        except Exception as e:
            logger.error(f"评估CI/CD完整性失败: {str(e)}")
            return 0.0

    async def _collect_code_quality_metrics(self, repo: str, target_date: datetime = None) -> dict:
        """收集代码质量指标"""
        try:
            if target_date is None:
                target_date = datetime.now(timezone.utc)
                 
            test_coverage, doc_score, cicd_score = await asyncio.gather(
                self._get_test_coverage(repo, target_date), 
                self._calculate_documentation_score(repo, target_date),
                self._evaluate_cicd(repo, target_date)
            )
            
            return {

                'test_coverage': test_coverage,
                'documentation_score': doc_score,
                'cicd_score': cicd_score
            }
            
        except Exception as e:
            logger.error(f"收集代码质量指标失败: {str(e)}")
            return {

                'test_coverage': 0.0,
                'documentation_score': 0.0,
                'cicd_score': 0.0
            }


    async def collect_monthly_data(self, repo: str) -> pd.DataFrame:
        """收集12个月的数据"""
        try:
            data = []
            now = datetime.now(timezone.utc)
            
            # 预获取仓库信息
            repo_info = await self._cached_request(f"https://api.github.com/repos/{repo}")
            if not repo_info:
                raise Exception("Failed to get repo info")

            dates = [
                (now.replace(day=1) - relativedelta(months=i))
                .replace(hour=0, minute=0, second=0)
                for i in range(12)
            ]
            
            # 并发收集所有指标数据
            tasks = []
            for date in dates:
                tasks.extend([
                    self.collect_activity_metrics(repo, date),
                    self._collect_community_metrics(repo, date),
                    self._collect_impact_metrics(repo, date),
                    self._collect_code_quality_metrics(repo, date),

                ])
            all_results = await asyncio.gather(*tasks)
            
            # 处理结果 - 每四个结果为一组
            for i in range(0, len(all_results), 4):
                activity_data = all_results[i]
                community_data = all_results[i+1]
                impact_data = all_results[i+2]
                quality_data = all_results[i+3]

                month_index = i // 4
                
                data.append({
                    'repo': repo,
                    'date': dates[month_index].strftime('%Y-%m-%d'),
                    **activity_data,          # 展开活跃度数据
                    **community_data,         # 展开社区指标数据
                    **impact_data,            # 展开影响力指标数据
                    **quality_data,           # 展开仓库完整性数据

                })
                
                logger.info(f"收集完成: {repo} at {dates[month_index].strftime('%Y-%m')}")
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"收集月度数据失败: {str(e)}")
            return pd.DataFrame()
        
    # async def export_to_csv(self, repo: str, output_file: str = None):
    #     """导出数据到CSV文件"""
    #     try:
    #         df = await self.collect_monthly_data(repo)
            
    #         if df.empty:
    #             logger.error("没有收集到数据")
    #             return False
                
    #         if output_file is None:
    #             output_file = f"测试啊{repo.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
                
    #         df.to_csv(output_file, index=False, encoding='utf-8')
    #         logger.info(f"数据已导出到: {output_file}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"导出CSV失败: {str(e)}")
    #         return False
        
# async def main():
#     """主函数"""
#     # GitHub personal access token (可选)
#     token = "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"
#     repo = "X-lab2017/open-digger"
    
#     async with HealthMetricsCollector(token) as collector:
#         await collector.export_to_csv(repo)

# if __name__ == "__main__":
#     asyncio.run(main())