import logging
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from typing import Dict, Any
from collector import HealthMetricsCollector
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
class IoTDBStorage:
    def __init__(self, host='127.0.0.1', port=6667, username='root', password='root'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.session = None
        self.storage_group_created = False
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            self.session.close()
            self.session = None
            self.storage_group_created = False

    def connect(self):
        try:
            self.session = Session(self.host, self.port, self.username, self.password, fetch_size=1024)
            self.session.open(False)
            # 创建根存储组
            if not self.storage_group_created:
                try:
                    self.session.set_storage_group("root.github")
                    self.storage_group_created = True
                except Exception:
                    pass
            logger.info("成功连接到IoTDB")
        except Exception as e:
            logger.error(f"连接IoTDB失败: {str(e)}")
            raise

    def query_metrics(self, repo: str, start_time: datetime, end_time: datetime) -> Dict:
        """查询指标数据"""
        try:
            if not self.session:
                self.connect()
                
            repo_path = repo.replace('/', '.').replace('-', '_')
            device_path = f"root.github.{repo_path}"
            
            # 转换时间戳并确保使用月初
            start_ts = int(start_time.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ).timestamp() * 1000)
            end_ts = int(end_time.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ).timestamp() * 1000)
            
            # 使用基本的查询语法
            sql = f"""
                SELECT *
                FROM {device_path}
                WHERE time >= {start_ts} AND time <= {end_ts}
            """
            
            with self.session.execute_query_statement(sql) as session_data_set:
                df = session_data_set.todf()
                
            if df.empty:
                return {}
                
            # 在 Python 中进行数据处理，确保使用月初时间
            df['Time'] = pd.to_datetime(df['Time'], unit='ms').dt.to_period('M').dt.to_timestamp()
            df = df.drop_duplicates(subset=['Time'], keep='first')
            
            # 转换为字典格式
            result = {}
            for _, row in df.iterrows():
                timestamp = int(row['Time'].timestamp() * 1000)
                values = row.drop(['Time']).to_dict()
                result[timestamp] = values
                
            return result
                
        except Exception as e:
            logger.error(f"查询指标数据失败: {str(e)}")
            return {}

    def view_all_data(self, repo: str, start_time: datetime, end_time: datetime):
        """查看指定时间范围内的所有数据"""
        try:
            if not self.session:
                self.connect()
                
            repo_path = repo.replace('/', '.').replace('-', '_')
            device_paths = [
                f"root.github.{repo_path}",
                f"root.github.{repo_path}_scores"
            ]
            
            for device_path in device_paths:
                print(f"\n查看 {device_path} 的数据:")
                
                start_ts = int(start_time.timestamp() * 1000)
                end_ts = int(end_time.timestamp() * 1000)
                
                sql = f"""
                    SELECT *
                    FROM {device_path}
                    WHERE time >= {start_ts} AND time <= {end_ts}
                """
                
                with self.session.execute_query_statement(sql) as session_data_set:
                    df = session_data_set.todf()
                    if not df.empty:
                        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
                        print(df)
                    else:
                        print("无数据")
                        
        except Exception as e:
            logger.error(f"查看数据失败: {str(e)}")
        
    def store_metrics(self, repo: str, metrics: Dict[str, Any], timestamp: datetime):
        try:
            # 1. 使用每月1号 00:00:00
            normalized_timestamp = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0,tzinfo=timezone.utc)
            
            repo_path = repo.replace('/', '.').replace('-', '_')
            device_path = f"root.github.{repo_path}"
            
            # 2. 检查当月是否已存在数据 - 使用简单的 SQL 语法
            month_start = normalized_timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_end = month_start + relativedelta(months=1)
            
            sql = f"""
                SELECT *
                FROM {device_path}
                WHERE time >= {int(month_start.timestamp() * 1000)}
                AND time < {int(month_end.timestamp() * 1000)}
                LIMIT 1
            """
            
            with self.session.execute_query_statement(sql) as session_data_set:
                df = session_data_set.todf()
                if not df.empty:
                    logger.info(f"跳过已存在的月度数据: {repo} at {normalized_timestamp.strftime('%Y-%m')}")
                    return

            # 3. 准备数据
            measurements = []
            data_types = []
            values = []

            for key, value in metrics.items():
                measurements.append(key)
                if isinstance(value, (int, float)):
                    data_types.append(TSDataType.DOUBLE)
                    values.append(float(value))
                else:
                    data_types.append(TSDataType.TEXT)
                    values.append(str(value))

            # 4. 确保时间序列存在
            for m, t in zip(measurements, data_types):
                try:
                    self.session.create_time_series(
                        f"{device_path}.{m}",
                        t,
                        TSEncoding.PLAIN,
                        Compressor.SNAPPY,
                    )
                except Exception:
                    pass

            # 5. 存储数据
            timestamp_ms = int(normalized_timestamp.timestamp() * 1000)
            self.session.insert_records(
                [device_path],
                [timestamp_ms],
                [measurements],
                [data_types],
                [values]
            )
            logger.info(f"成功存储月度指标数据: {repo} at {normalized_timestamp.strftime('%Y-%m')}")

        except Exception as e:
            logger.error(f"存储指标数据失败: {str(e)}")
            raise

    def clear_all_data(self):
        """清空数据库中的所有数据"""
        try:
            if not self.session:
                self.connect()
                
            # 删除存储组
            self.session.delete_storage_group("root.github")
            self.storage_group_created = False
            
            # 重新创建存储组
            self.session.set_storage_group("root.github")
            self.storage_group_created = True
            
            logger.info("已清空数据库中的所有数据")
            
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
        
    def close(self):
        """关闭数据库连接"""
        if self.session:
            self.session.close()
            self.session = None
            self.storage_group_created = False