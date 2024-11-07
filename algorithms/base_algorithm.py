from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from uuid import uuid4
from enum import Enum

class AlgorithmStatus(Enum):
    """算法状态枚举"""
    INITIALIZED = "initialized"  # 初始化完成
    RUNNING = "running"         # 正在运行
    STOPPED = "stopped"        # 已停止
    ERROR = "error"           # 发生错误

class BaseAlgorithm(ABC):
    """算法基类"""
    
    def __init__(self, algo_config: Dict[str, Any]):
        self.algo_config = algo_config
        self.status = AlgorithmStatus.INITIALIZED
        self.error_msg = None
        self.is_running = False
        # 算法类型名称
        self.type_name = self.__class__.__name__
        # 算法实例唯一ID
        self.instance_id = f"{self.type_name}_{uuid4().hex[:8]}"
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化算法"""
        pass
        
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单帧"""
        pass
        
    @abstractmethod
    def release(self):
        """释放资源"""
        pass
    
    def start(self):
        """启动算法"""
        if self.initialize():
            self.status = AlgorithmStatus.RUNNING
            self.is_running = True
            return True
        return False
        
    def stop(self):
        """停止算法"""
        self.status = AlgorithmStatus.STOPPED
        self.is_running = False
        self.release()
        
    def get_status(self) -> Dict[str, Any]:
        """获取算法状态"""
        return {
            'status': self.status.value,
            'error_msg': self.error_msg,
            'is_running': self.is_running
        }