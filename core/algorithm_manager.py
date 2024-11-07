# 标准库导入
from typing import Dict, Optional, List, Type
import threading
import logging
import importlib
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# 内部模块导入
from ..algorithms.base_algorithm import BaseAlgorithm, AlgorithmStatus

@dataclass
class AlgorithmInfo:
    """算法实例信息"""
    algo_id: str
    instance_id: str
    config: dict
    create_time: datetime = datetime.now()
    last_active: datetime = datetime.now()

class AlgorithmManager:
    """算法管理器"""
    def __init__(self):
        self.algo_classes: Dict[str, Type[BaseAlgorithm]] = {}  # 算法类字典
        self.instances: Dict[str, BaseAlgorithm] = {}  # 算法实例字典
        self.instance_info: Dict[str, AlgorithmInfo] = {}  # 实例信息字典
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._load_algorithms()
        
    def _load_algorithms(self):
        """从算法目录加载所有算法类"""
        try:
            # 获取算法目录的绝对路径
            algo_dir = Path(__file__).parent.parent / 'algorithms'
            self.logger.info(f"正在从目录加载算法: {algo_dir}")
            
            # 添加项目根目录到Python路径
            project_root = algo_dir.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            for item in os.listdir(algo_dir):
                if os.path.isdir(algo_dir / item) and not item.startswith('__'):
                    try:
                        self.logger.info(f"尝试加载算法: {item}")
                        # 使用绝对导入路径
                        module_name = f"{item}_algorithm"
                        module_file = algo_dir / item / f"{module_name}.py"
                        
                        self.logger.info(f"模块名称: {module_name}")
                        self.logger.info(f"模块文件路径: {module_file}")
                        self.logger.info(f"Python路径: {sys.path}")
                        
                        spec = importlib.util.spec_from_file_location(
                            module_name, 
                            module_file
                        )
                        
                        if spec is None:
                            self.logger.error(f"无法找到模块规范: {module_name}")
                            raise ImportError(f"模块 {module_name} 不存在")
                            
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module  # 将模块添加到sys.modules
                        
                        self.logger.info(f"正在执行模块: {module_name}")
                        spec.loader.exec_module(module)
                        
                        algo_class = getattr(module, f'{item.capitalize()}Algorithm')
                        self.algo_classes[item] = algo_class
                        self.logger.info(f"成功加载算法类: {item}")
                        
                    except Exception as e:
                        self.logger.error(f"加载算法 {item} 失败: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"加载算法目录失败: {str(e)}")
                
    def create_instance(self, algo_id: str, instance_id: str, config: dict = None) -> bool:
        """创建算法实例"""
        with self._lock:
            if algo_id not in self.algo_classes:
                self.logger.error(f"算法 {algo_id} 不存在")
                return False
                
            if instance_id in self.instances:
                self.logger.error(f"实例 {instance_id} 已存在")
                return False
                
            try:
                algo_class = self.algo_classes[algo_id]
                instance = algo_class(config or {})
                self.instances[instance_id] = instance
                self.instance_info[instance_id] = AlgorithmInfo(
                    algo_id=algo_id,
                    instance_id=instance_id,
                    config=config or {}
                )
                self.logger.info(f"成功创建算法实例: {instance_id}")
                return True
            except Exception as e:
                self.logger.error(f"创建算法实例失败: {str(e)}")
                return False
                
    def start_instance(self, instance_id: str) -> bool:
        """启动算法实例"""
        with self._lock:
            if instance_id not in self.instances:
                self.logger.error(f"实例 {instance_id} 不存在")
                return False
                
            instance = self.instances[instance_id]
            if instance.status == AlgorithmStatus.RUNNING:
                self.logger.warning(f"实例 {instance_id} 已在运行")
                return True
                
            try:
                if instance.start():
                    self.instance_info[instance_id].last_active = datetime.now()
                    self.logger.info(f"成功启动实例: {instance_id}")
                    return True
                self.logger.error(f"启动实例 {instance_id} 失败")
                return False
            except Exception as e:
                self.logger.error(f"启动实例 {instance_id} 异常: {str(e)}")
                return False
                
    def stop_instance(self, instance_id: str) -> bool:
        """停止算法实例"""
        with self._lock:
            if instance_id not in self.instances:
                self.logger.error(f"实例 {instance_id} 不存在")
                return False
                
            try:
                instance = self.instances[instance_id]
                instance.stop()
                self.logger.info(f"成功停止实例: {instance_id}")
                return True
            except Exception as e:
                self.logger.error(f"停止实例 {instance_id} 异常: {str(e)}")
                return False
                
    def remove_instance(self, instance_id: str) -> bool:
        """移除算法实例"""
        with self._lock:
            if instance_id not in self.instances:
                return False
                
            try:
                self.stop_instance(instance_id)
                del self.instances[instance_id]
                del self.instance_info[instance_id]
                return True
            except Exception as e:
                self.logger.error(f"移除实例 {instance_id} 异常: {str(e)}")
                return False
                
    def get_instance_status(self, instance_id: str) -> Optional[dict]:
        """获取实例状态"""
        with self._lock:
            if instance_id not in self.instances:
                return None
                
            instance = self.instances[instance_id]
            info = self.instance_info[instance_id]
            
            return {
                'instance_id': instance_id,
                'algo_id': info.algo_id,
                'status': instance.get_status(),
                'create_time': info.create_time.isoformat(),
                'last_active': info.last_active.isoformat()
            }
            
    def get_all_instances(self) -> List[dict]:
        """获取所有实例状态"""
        with self._lock:
            return [self.get_instance_status(i_id) for i_id in self.instances.keys()]

    def get_instance(self, instance_id: str) -> Optional[BaseAlgorithm]:
        """获取算法实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            BaseAlgorithm: 算法实例，不存在则返回None
        """
        with self._lock:
            return self.instances.get(instance_id)