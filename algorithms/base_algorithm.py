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
    """算法基类
    
    所有具体算法实现都应该继承这个基类。该类定义了算法的基本接口和通用功能。
    
    主要功能:
    1. 提供算法生命周期管理（初始化、启动、停止）
    2. 定义帧处理接口
    3. 提供状态管理和错误处理
    4. 实现资源管理
    
    Attributes:
        algo_config (Dict[str, Any]): 算法配置参数字典
        status (AlgorithmStatus): 当前算法状态
        error_msg (Optional[str]): 错误信息，如果有的话
        is_running (bool): 算法是否正在运行
        type_name (str): 算法类型名称
        instance_id (str): 算法实例唯一标识符
    """
    
    def __init__(self, algo_config: Dict[str, Any]):
        """初始化算法实例
        
        Args:
            algo_config: 算法配置参数字典，包含算法所需的所有配置项
                例如：{
                    'threshold': 0.5,
                    'min_size': 100,
                    'max_results': 5
                }
        """
        self.algo_config = algo_config
        self.status = AlgorithmStatus.INITIALIZED
        self.error_msg = None
        self.is_running = False
        # 算法类型名称，使用类名
        self.type_name = self.__class__.__name__
        # 算法实例唯一ID，使用类名+随机字符串
        self.instance_id = f"{self.type_name}_{uuid4().hex[:8]}"
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化算法
        
        执行算法所需的初始化操作，例如：
        1. 加载模型
        2. 初始化外部API连接
        3. 分配必要的资源
        
        Returns:
            bool: 初始化是否成功
        """
        pass
        
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单帧图像
        
        对输入的图像帧进行算法处理
        
        Args:
            frame: 输入图像帧，numpy数组格式(H,W,C)
            
        Returns:
            Dict[str, Any]: 处理结果字典，应包含：
                - status: 处理状态
                - results: 算法输出结果
                - performance: 性能指标
                例如：{
                    'status': 'success',
                    'results': [...],
                    'process_time': 0.05
                }
        """
        pass
        
    @abstractmethod
    def release(self):
        """释放算法资源
        
        清理算法使用的资源，例如：
        1. 释放GPU内存
        2. 关闭外部连接
        3. 清理临时文件
        """
        pass
    
    def start(self) -> bool:
        """启动算法
        
        执行算法启动流程：
        1. 调用initialize()进行初始化
        2. 设置运行状态
        3. 启动必要的后台线程
        
        Returns:
            bool: 启动是否成功
        """
        if self.initialize():
            self.status = AlgorithmStatus.RUNNING
            self.is_running = True
            return True
        return False
        
    def stop(self):
        """停止算法
        
        执行算法停止流程：
        1. 设置停止状态
        2. 停止所有后台线程
        3. 调用release()释放资源
        """
        self.status = AlgorithmStatus.STOPPED
        self.is_running = False
        self.release()
        
    def get_status(self) -> Dict[str, Any]:
        """获取算法状态信息
        
        Returns:
            Dict[str, Any]: 状态信息字典，包含：
                - status: 当前状态(AlgorithmStatus)
                - error_msg: 错误信息(如果有)
                - is_running: 是否正在运行
        """
        return {
            'status': self.status.value,
            'error_msg': self.error_msg,
            'is_running': self.is_running
        }
        
    def _set_error(self, error_msg: str):
        """设置错误状态和信息
        
        Args:
            error_msg: 错误信息描述
        """
        self.status = AlgorithmStatus.ERROR
        self.error_msg = error_msg
        self.is_running = False
        
    def _reset_status(self):
        """重置算法状态
        
        清除错误信息并将状态设置为初始化
        """
        self.status = AlgorithmStatus.INITIALIZED
        self.error_msg = None
        self.is_running = False