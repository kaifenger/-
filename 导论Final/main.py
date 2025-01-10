import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from collector import HealthMetricsCollector
from calculator import HealthScoreCalculator
from storage import IoTDBStorage
from models import ProjectHealth, ActivityMetrics, CommunityMetrics, ImpactMetrics, CodeQualityMetrics
from predictor import HealthMetricsPredictor
import logging
import threading
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_github_repo(repo: str, months: int = 12):
    """处理GitHub仓库数据"""
    try:
        # 从环境变量获取token
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("未找到GitHub Token，请检查环境变量配置")
      
        collector = HealthMetricsCollector(token)
        calculator = HealthScoreCalculator()
        storage = IoTDBStorage()
        
        async with collector:
            storage.connect()
            try:
                # 修改为生成月初时间列表
                now = datetime.now(timezone.utc)
                dates = []
                for i in range(months):
                    # 计算月初时间
                    month_start = (
                        (now.replace(day=1) - relativedelta(months=i))
                        .replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                    )
                    dates.append(month_start)
                dates.reverse()  # 按时间顺序排序
                for date in dates:
                    # 检查是否已存在数据
                    existing_data = storage.query_metrics(
                        repo,
                        date,  # 月初
                        (date + relativedelta(months=1))  # 下月初
                    )
                    
                    if existing_data:
                        logger.info(f"跳过已存在数据: {repo} at {date.strftime('%Y-%m')}")
                        continue
                        
                    try:
                        # 收集数据
                        activity = await collector.collect_activity_metrics(repo, date)
                        community = await collector._collect_community_metrics(repo, date)
                        impact = await collector._collect_impact_metrics(repo, date)
                        code_quality = await collector._collect_code_quality_metrics(repo, date)

                        health_metrics = ProjectHealth(
                            repo_name=repo,
                            timestamp=date,
                            activity=ActivityMetrics(**activity),
                            community=CommunityMetrics(**community),
                            impact=ImpactMetrics(**impact),
                            code_quality=CodeQualityMetrics(**code_quality)
                        )

                        scores = calculator.calculate_total_score(health_metrics)

                        # 存储metrics数据
                        metrics_data = {
                            'repo_name': repo, 
                            **activity,
                            **community,
                            **impact,
                            **code_quality
                        }
                        storage.store_metrics(repo=repo, metrics=metrics_data, timestamp=date)

                        # 存储scores数据
                        score_data = {
                            'total_score': scores['total_score'],
                            'health_grade': scores['health_grade'],
                            **{f"{dim}_score": details['score'] 
                               for dim, details in scores['dimensions'].items()},
                            **{f"{dim}_grade": details['grade']
                               for dim, details in scores['dimensions'].items()}
                        }
                        storage.store_metrics(
                            repo=f"{repo}_scores",
                            metrics=score_data,
                            timestamp=date
                        )

                        logger.info(f"完成 {repo} 在 {date.strftime('%Y-%m')} 的数据处理")

                    except Exception as e:
                        logger.error(f"处理 {repo} 在 {date.strftime('%Y-%m')} 时出错: {str(e)}")
                        continue

            finally:
                storage.close()

    except Exception as e:
        logger.error(f"处理仓库 {repo} 失败: {str(e)}")
        raise

async def ensure_utc_timestamp(timestamp):
    """确保时间戳使用UTC时区"""
    ts = pd.to_datetime(timestamp)
    return ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')

async def main():
    """主函数：启动仪表盘应用"""
    try:
        #清空数据库
        # logger.info("正在清空数据库...")
        # storage = IoTDBStorage()
        # storage.clear_all_data()
        # storage.close()
        # logger.info("数据库清空完成")
        logger.info("启动数据可视化仪表盘...")
        from dashboard import app
        
        def run_dash():
            app.run_server(debug=False, port=8050)
        
        # 在新线程中启动Dash应用
        dash_thread = threading.Thread(target=run_dash, daemon=True)
        dash_thread.start()
        
        logger.info("仪表盘已启动，访问 http://localhost:8050 查看")
        logger.info("按Ctrl+C退出程序...")
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("程序正在退出...")
    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}", exc_info=True)
if __name__ == "__main__":
    asyncio.run(main())