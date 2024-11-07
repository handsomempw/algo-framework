# 标准库导入
from typing import Dict, Optional, List, Tuple, Union
import threading
import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# 第三方库导入
import requests

# 内部模块导入
from .algorithm_manager import AlgorithmManager
from .resource_manager import ResourceManager, VideoStream
from ..algorithms.base_algorithm import BaseAlgorithm

@dataclass
class TaskConfig:
    """任务配置类"""
    task_id: str
    name: str
    camera_name: str
    camera_id: str
    camera_gb28181_id: str
    algo_type: str
    algo_type_name: str
    algo_config: dict
    status: str = '禁用'
    instance_id: Optional[str] = None
    create_time: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    process_thread: Optional[threading.Thread] = None
    error_msg: str = ''
    _stream_url: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not hasattr(self, '_stream_url'):
            self._stream_url = None

    @property
    def stream_url(self) -> str:
        """获取视频流URL"""
        if not self._stream_url:
            # 使用固定的测试视频流地址
            self._stream_url = "rtsp://192.168.96.136:30028/play/7199616326262984705"
        return self._stream_url

    @stream_url.setter
    def stream_url(self, value: str):
        self._stream_url = value

    @property
    def enabled(self) -> bool:
        """任务是否启用"""
        return self.status == "启用"

    @enabled.setter
    def enabled(self, value: bool):
        self.status = "启用" if value else "禁用"

    @classmethod
    def from_api_response(cls, data: dict) -> List['TaskConfig']:
        """从API响应数据创建测试任务配置"""
        tasks = []
        
        # 基础任务信息
        base_info = {
            'camera_name': '测试相机',
            'camera_id': '7199616326262984705',
            'camera_gb28181_id': '34020000001310000001'
        }
        
        # 创建两个测试任务
        for i in range(2):
            task = cls(
                task_id=f"{data.get('id', 'test')}_stream_{i+1}",
                name=f"视频流测试_{i+1}",
                **base_info,
                algo_type="blacklist",
                algo_type_name=f"黑名单检测_{i+1}",
                algo_config={
                    'match_threshold': 0.7 + i*0.1,
                    'min_face_size': 100 + i*20,
                    'blacklist_db': 'aa4'
                },
                status='启用'
            )
            tasks.append(task)
        
        return tasks

class TaskManager:
    """任务管理器
    
    负责从API获取任务配置并管理任务的生命周期
    """
    # 默认API地址
    DEFAULT_API_URL = "http://192.168.100.137:30337/api/aisp-video-compute-manager/v1"
    
    def __init__(self, api_base_url=None):
        self._running = True
        self._lock = threading.Lock()
        self.tasks = {}
        self.algo_manager = AlgorithmManager()
        self.resource_manager = ResourceManager()
        self.logger = logging.getLogger(self.__class__.__module__)
        # 使用传入的API地址或默认地址
        self.api_base_url = api_base_url or self.DEFAULT_API_URL
        self.logger.info(f"初始化任务管理器, API地址: {self.api_base_url}")
        
        self._monitor_thread = None
        
    def _fetch_tasks(self):
        """从API获取任务列表
        
        Returns:
            list: 任务列表
        """
        try:
            url = f"{self.api_base_url}/tasks"
            response = requests.get(url)
            response.raise_for_status()
            
            # API直接返回任务列表，不需要获取data字段
            tasks = response.json()
            if not isinstance(tasks, list):
                raise ValueError(f"API返回数据格式错误: {tasks}")
                
            self.logger.info(f"成功获取任务列表: {len(tasks)}个")
            return tasks
            
        except Exception as e:
            self.logger.error(f"获取任务列表失败: {str(e)}")
            return []
            
    def _convert_task(self, task_data: dict) -> TaskConfig:
        """转换API任务数据为TaskConfig对象"""
        try:
            return TaskConfig.from_api_response(task_data)
        except Exception as e:
            self.logger.error(f"转换任务配置失败: {str(e)}, 任务数据: {task_data}")
            return None
        
    def _add_task(self, task_data: dict):
        """添加任务"""
        try:
            tasks = TaskConfig.from_api_response(task_data)
            for task in tasks:
                self.tasks[task.task_id] = task
                self.logger.info(f"\n测试任务详情:")
                self.logger.info(f"任务ID: {task.task_id}")
                self.logger.info(f"名称: {task.name}")
                self.logger.info(f"算法类型: {task.algo_type_name}")
                self.logger.info(f"相机ID: {task.camera_id}")
                self.logger.info(f"配置: {task.algo_config}")
                self.logger.info("-" * 50)
        except Exception as e:
            self.logger.error(f"添加任务失败: {str(e)}")
            raise

    def load_tasks(self):
        """加载任务配置"""
        self.logger.info("开始加载测试任务...")
        
        try:
            # 从API获取任务列表用生成测试用例
            tasks_data = self._fetch_tasks()
            if tasks_data:
                # 使用TaskConfig类方法创建测试任务
                test_tasks = TaskConfig.from_api_response(tasks_data[0])
                
                # 添加所有测试任务
                for task in test_tasks:
                    self.tasks[task.task_id] = task
                    self.logger.info(f"\n测试任务详情:")
                    self.logger.info(f"任务ID: {task.task_id}")
                    self.logger.info(f"名称: {task.name}")
                    self.logger.info(f"算法类型: {task.algo_type_name}")
                    self.logger.info(f"相机ID: {task.camera_id}")
                    self.logger.info(f"配置: {task.algo_config}")
                    self.logger.info("-" * 50)
                
                self.logger.info(f"成功加载 {len(self.tasks)} 个测试任务")
                
        except Exception as e:
            self.logger.error(f"加载任务失败: {str(e)}")
            
    def _setup_task_resources(self, task: TaskConfig) -> Tuple[VideoStream, BaseAlgorithm]:
        """配置任务资源"""
        with self._lock:
            try:
                # 获取视频流URL
                stream_url = task.stream_url
                self.logger.info(f"获取视频流URL: {stream_url}")
                
                # 检查是否可以复用视频流
                stream = self.resource_manager.get_video_stream(stream_url)
                if not stream:
                    # 创建新的视频流
                    stream = VideoStream(stream_url)
                    self.resource_manager.add_video_stream(stream_url, stream)
                    self.logger.info(f"创建新视频流: {stream_url}")
                else:
                    self.logger.info(f"复用已有视频流: {stream_url}")
                    
                # 创建并启动算法实例
                instance_id = f"{task.algo_type}_{task.task_id}"
                if not task.instance_id:
                    if not self.algo_manager.create_instance(task.algo_type, instance_id, task.algo_config):
                        raise RuntimeError(f"创建算法实例失败: {instance_id}")
                    task.instance_id = instance_id
                    
                    # 启动算法实例
                    if not self.algo_manager.start_instance(instance_id):
                        raise RuntimeError(f"启动算法实例失败: {instance_id}")
                    
                instance = self.algo_manager.get_instance(instance_id)
                return stream, instance
                
            except Exception as e:
                self.logger.error(f"配置任务资源失败: {str(e)}")
                raise
            
    def start_task(self, task_id: str) -> bool:
        """启动任务"""
        try:
            task = self.tasks[task_id]
            stream, instance = self._setup_task_resources(task)
            
            # 创建任务处理线程
            task_thread = threading.Thread(
                target=self._task_process_loop,
                args=(task, stream, instance)
            )
            task_thread.daemon = True
            task_thread.start()
            
            task.process_thread = task_thread
            return True
            
        except Exception as e:
            self.logger.error(f"启动任务失败: {str(e)}")
            return False
            
    def _task_process_loop(self, task: TaskConfig, stream: VideoStream, instance: BaseAlgorithm):
        """任务处理循环"""
        try:
            while task.enabled and instance.is_running and self._running:
                try:
                    frame = stream.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                        
                    result = instance.process_frame(frame)
                    self._handle_result(task, result)
                    task.last_active = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"处理帧异常: {str(e)}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"任务处理异常: {str(e)}")
            task.status = '异常'
        finally:
            # 确保资源被正确释放
            if instance:
                instance.release()
                self.logger.info(f"任务 {task.task_id} 处理循环已停止")
            
    def stop_task(self, task_id: str) -> bool:
        """停止任务"""
        with self._lock:
            if task_id not in self.tasks:
                self.logger.error(f"任务 {task_id} 不存在")
                return False
                
            task = self.tasks[task_id]
            try:
                # 停止并移除算法实例
                if task.instance_id:
                    self.algo_manager.stop_instance(task.instance_id)
                    self.algo_manager.remove_instance(task.instance_id)
                    task.instance_id = None
                    
                # 释放视频流
                self.resource_manager.release_stream(task.stream_url)
                self.logger.info(f"成功停止任务: {task_id}")
                return True
            except Exception as e:
                self.logger.error(f"停止任务 {task_id} 异常: {str(e)}")
                return False
                
    def get_task_status(self, task_id: str) -> Optional[dict]:
        """取任务状态"""
        with self._lock:
            if task_id not in self.tasks:
                return None
                
            task = self.tasks[task_id]
            status = {
                'task_id': task_id,
                'algo_type': task.algo_type,
                'stream_url': task.stream_url,
                'enabled': task.enabled,
                'create_time': task.create_time.isoformat(),
                'last_active': task.last_active.isoformat()
            }
            
            # 获取算法实例状态
            if task.instance_id:
                status['instance'] = self.algo_manager.get_instance_status(task.instance_id)
                
            # 获取视频流状态
            stream_status = self.resource_manager.get_stream_status(task.stream_url)
            if stream_status:
                status['stream'] = stream_status
                
            return status
            
    def get_all_tasks(self) -> List[dict]:
        """获取所有任务状态"""
        with self._lock:
            return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
            
    def reload_tasks(self):
        """重新加载任务配置"""
        with self._lock:
            # 停止所有任务
            self.stop_all_tasks()
            # 清空任务列表
            self.tasks.clear()
            # 重新加载
            self.load_tasks()
            # 启动所有任务
            self.start_all_tasks()
            
    def start_all_tasks(self):
        """启动所有已启用的任务"""
        self._running = True
        for task_id, task in self.tasks.items():
            if task.enabled:
                self.start_task(task_id)
                
    def stop_all_tasks(self):
        """停止所有任务"""
        with self._lock:
            # 先设置停止标志
            self._running = False
            time.sleep(0.1)  # 给处理循环一点时间响应停止信号
            
            # 停止所有任务
            for task_id in list(self.tasks.keys()):
                try:
                    # 先禁用任务
                    task = self.tasks[task_id]
                    task.enabled = False
                    
                    # 等待处理线程结束
                    if task.process_thread and task.process_thread.is_alive():
                        task.process_thread.join(timeout=1.0)
                        
                    # 停止任务实例
                    self.stop_task(task_id)
                except Exception as e:
                    self.logger.error(f"停止任务 {task_id} 失败: {str(e)}")

    def start_monitor(self):
        """启动任务监控线程"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_tasks)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def _monitor_tasks(self):
        """实时监控任务状态"""
        while self._running:
            try:
                tasks_data = self._fetch_tasks()
                
                # 检查任务状态变化
                for task_data in tasks_data:
                    task_id = str(task_data['id'])
                    status = task_data.get('status', '禁用')
                    
                    if task_id in self.tasks:
                        current_task = self.tasks[task_id]
                        if status != current_task.status:
                            self.logger.info(f"任务状态变更 - {task_id}: {current_task.status} -> {status}")
                            current_task.status = status
                            
                            if status == '启用':
                                if not current_task.is_running:
                                    self.start_task(task_id)
                            elif status == '禁用':
                                if current_task.is_running:
                                    self.stop_task(task_id)
                                    
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                self.logger.error(f"控任务异常: {str(e)}")
                time.sleep(1)

    def _handle_result(self, task: TaskConfig, result: dict):
        """处理算法结果"""
        try:
            matches = result.get('matches', [])
            frame_info = result.get('frame_info', {})
            
            if matches:
                self.logger.info(f"任务 {task.task_id} 检测到匹配:")
                for match in matches:
                    self.logger.info(f"- 相似度: {match.get('score', 0):.2f}")
                    self.logger.info(f"- 位置: {match.get('face_location', [])}")
                    if 'person_info' in match:
                        self.logger.info(f"- 人员信息: {match['person_info']}")
                        
            # 更新任务状态
            task.last_active = datetime.now()
            
        except Exception as e:
            self.logger.error(f"处理结果异常: {str(e)}")

    def stop_monitor(self):
        """停止任务监控线程"""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)  # 等待监控线程结束