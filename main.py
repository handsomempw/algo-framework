import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 配置日志
def setup_logger():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,  # 或 DEBUG 以查看更多细节
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 第三方库导入
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 内部模块导入
from algo_framework.core.task_manager import TaskManager

class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    def __init__(self, task_manager):
        self.task_manager = task_manager
        
    def on_modified(self, event):
        if event.src_path.endswith('tasks.json'):
            logging.info("检测到任务配置文件变更，重新加载...")
            self.task_manager.reload_tasks()

class AlgoSystem:
    """算法系统主类"""
    def __init__(self):
        # 设置日志
        setup_logger()
        # 初始化任务管理器
        self.task_manager = TaskManager()
        self.logger = logging.getLogger(__name__)
        
    def start(self, run_duration: int = 20):
        """
        启动系统并运行指定时长
        Args:
            run_duration: 运行时长(秒)
        """
        try:
            self.logger.info("正在启动算法系统...")
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=run_duration)
            
            # 加载并启动测试任务
            self.task_manager.load_tasks()
            self.task_manager.start_all_tasks()
            
            # 记录初始状态
            self.logger.info("\n初始任务状态:")
            self._print_tasks_status()
            
            # 主循环,每5秒打印一次状态
            while datetime.now() < end_time:
                time.sleep(5)
                self.logger.info("\n当前任务状态:")
                self._print_tasks_status()
                
            # 打印最终性能统计
            self.logger.info("\n=== 最终性能统计 ===")
            self._print_performance_metrics()
            
        except KeyboardInterrupt:
            self.logger.info("正在关闭系统...")
        finally:
            self.task_manager.stop_all_tasks()
            
    def _print_tasks_status(self):
        """打印所有任务状态"""
        for task_id, task in self.task_manager.tasks.items():
            status = self.task_manager.get_task_status(task_id)
            if status:
                self.logger.info(f"任务 {task.name}:")
                self.logger.info(f"- 状态: {task.status}")
                if 'instance' in status:
                    inst = status['instance']
                    self.logger.info(f"- 处理帧数: {inst.get('status', {}).get('frame_count', 0)}")
                    self.logger.info(f"- 最后处理时间: {inst.get('last_active')}")
                self.logger.info("-" * 30)
                
    def _print_performance_metrics(self):
        """打印性能指标"""
        for task_id, task in self.task_manager.tasks.items():
            status = self.task_manager.get_task_status(task_id)
            if status and 'instance' in status:
                inst = status['instance']
                metrics = inst.get('status', {})
                
                self.logger.info(f"\n任务 {task.name} 性能指标:")
                self.logger.info(f"总处理帧数: {metrics.get('frame_count', 0)}")
                if 'process_times' in metrics:
                    times = metrics['process_times']
                    if times:
                        avg_time = sum(times) / len(times)
                        self.logger.info(f"平均处理时间: {avg_time:.3f}秒")
                        self.logger.info(f"最大处理时间: {max(times):.3f}秒")
                        self.logger.info(f"最小处理时间: {min(times):.3f}秒")
                
                if 'match_counts' in metrics:
                    matches = metrics['match_counts']
                    if matches:
                        avg_matches = sum(matches) / len(matches)
                        self.logger.info(f"平均匹配数: {avg_matches:.2f}")
                
                self.logger.info("-" * 50)

def main():
    """主程序入口"""
    try:
        system = AlgoSystem()
        # 运行20秒后自动停止
        system.start(run_duration=20)
    except Exception as e:
        logging.error(f"主程序异常: {str(e)}")
        raise

if __name__ == "__main__":
    main()