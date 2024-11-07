"""算法框架包初始化文件
提供框架版本信息和核心组件导出
"""

__version__ = '0.1.0'

from .core.algorithm_manager import AlgorithmManager
from .core.task_manager import TaskManager
from .core.resource_manager import ResourceManager

__all__ = [
    'AlgorithmManager',
    'TaskManager', 
    'ResourceManager'
]