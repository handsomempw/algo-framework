"""视频资源管理模块

该模块提供视频流资源的管理功能，包括:
1. 视频流的创建、复用和释放
2. 帧读取和缓存机制
3. 多任务并发访问控制
4. 资源状态监控

主要类:
- StreamInfo: 视频流基础信息数据类
- VideoStream: 视频流处理类，实现帧读取和缓存
- ResourceManager: 资源管理器，负责所有视频流的生命周期管理
"""

from typing import Dict, Optional, List
import cv2
import threading
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import time

@dataclass
class StreamInfo:
    """视频流信息数据类
    
    Attributes:
        url (str): 视频流URL地址
        ref_count (int): 引用计数，记录使用该流的任务数
        last_active (datetime): 最后活跃时间，用于清理过期资源
    """
    url: str = "http://192.168.100.137:30337/api/aisp-video-compute-manager/v1/cameras/7199616326262984705/_play"
    ref_count: int = 0
    last_active: datetime = datetime.now()
    
class VideoStream:
    """视频流处理类
    
    实现视频帧的读取、缓存和分发功能。采用生产者-消费者模式，
    通过线程安全的方式支持多任务并发访问。
    
    Attributes:
        url (str): 视频流地址
        cap (cv2.VideoCapture): OpenCV视频捕获对象
        frame_count (int): 已处理帧计数
        ref_count (int): 引用计数
        _current_frame (np.ndarray): 当前帧缓存
        callbacks (list): 回调函数列表，用于帧处理通知
    """
    
    def __init__(self, url: str):
        """初始化视频流对象
        
        Args:
            url (str): 视频流地址
        """
        self.url = url
        self.cap = None
        self.frame_count = 0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()  # 保护帧数据的互斥锁
        self.ref_count = 1
        self.logger = logging.getLogger(__name__)
        # 帧缓存机制
        self._current_frame = None  # 当前帧缓存
        self._frame_ready = threading.Event()  # 帧就绪事件
        self.callbacks = []  # 帧处理回调函数列表
        
    def initialize(self) -> bool:
        """初始化视频流
        
        创建视频捕获对象并启动帧读取线程。
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.url)
            # 设置最小缓冲区，减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                self.logger.error(f"无法打开视频流: {self.url}")
                return False
                
            self._running = True
            self._thread = threading.Thread(target=self._read_frames)
            self._thread.daemon = True
            self._thread.start()
            return True
            
        except Exception as e:
            self.logger.error(f"初始化视频流失败: {str(e)}")
            return False
            
    def _read_frames(self):
        """帧读取循环(生产者)
        
        持续从视频流读取帧并更新缓存。使用锁保护帧数据的更新，
        通过事件通知消费者新帧可用。
        """
        import cv2
        while self._running:
            try:
                if self.cap is None:
                    break
                    
                ret, frame = self.cap.read()  # 读取一帧
                if not ret:
                    time.sleep(0.01)  # 读取失败时短暂等待
                    continue
                    
                with self._lock:  # 保护帧数据更新
                    self.frame_count += 1
                    self._current_frame = frame.copy()
                    self._frame_ready.set()  # 通知帧就绪
                    
            except Exception as e:
                self.logger.error(f"读取视频帧异常: {str(e)}")
                time.sleep(0.1)
                
    def read(self) -> Optional[np.ndarray]:
        """读取一帧(消费者)
        
        等待并返回最新的视频帧。如果超时或流已关闭则返回None。
        
        Returns:
            Optional[np.ndarray]: 视频帧数据，None表示读取失败
        """
        if not self.is_opened:
            return None
            
        # 等待新帧就绪
        self._frame_ready.wait(timeout=1.0)
        with self._lock:
            frame = self._current_frame.copy() if self._current_frame is not None else None
            self._frame_ready.clear()
        return frame
        
    @property
    def is_running(self) -> bool:
        """检查视频流是否正在运行
        
        Returns:
            bool: 视频流是否正在运行
        """
        return self._running and self._thread and self._thread.is_alive()
        
    def release(self):
        """释放视频流资源
        
        减少引用计数，当计数为0时释放底层资源。
        """
        with self._lock:
            self.ref_count -= 1
            if self.ref_count <= 0 and self.cap:
                self.cap.release()
                self.cap = None
        
    @property
    def is_opened(self) -> bool:
        """检查视频流是否打开
        
        Returns:
            bool: 视频流是否打开
        """
        return self.cap is not None and self.cap.isOpened()

class ResourceManager:
    """资源管理器
    
    负责管理所有视频流资源的生命周期，实现资源的创建、复用和释放。
    通过引用计数机制实现资源共享，避免重复创建相同的视频流。
    
    Attributes:
        streams (Dict[str, VideoStream]): 视频流字典，以URL为键
    """
    
    def __init__(self):
        """初始化资源管理器"""
        self.streams: Dict[str, VideoStream] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def get_video_stream(self, url: str) -> Optional[VideoStream]:
        """获取视频流实例
        
        如果指定URL的视频流已存在则复用，否则创建新的实例。
        
        Args:
            url (str): 视频流URL
            
        Returns:
            Optional[VideoStream]: 视频流实例，None表示创建失败
        """
        with self._lock:
            if url in self.streams:
                stream = self.streams[url]
                stream.ref_count += 1
                self.logger.info(f"复用已存在的视频流: {url}, 引用计数: {stream.ref_count}")
                return stream
            
            # 创建新的视频流
            stream = VideoStream(url)
            if stream.initialize():
                self.streams[url] = stream
                self.logger.info(f"成功创建新视频流: {url}, 引用计数: {stream.ref_count}")
                return stream
            return None
            
    def create_video_stream(self, url: str) -> Optional[VideoStream]:
        """创建新的视频流
        
        Args:
            url (str): 视频流URL
            
        Returns:
            Optional[VideoStream]: 视频流实例，None表示创建失败
        """
        return self.get_video_stream(url)
        
    def get_stream_status(self, url: str) -> Optional[dict]:
        """获取视频流状态信息
        
        Args:
            url (str): 视频流URL
            
        Returns:
            Optional[dict]: 状态信息字典，包含运行状态、引用计数等
        """
        with self._lock:
            if url not in self.streams:
                return None
                
            stream = self.streams[url]
            return {
                'url': url,
                'status': 'running' if stream.is_running else 'stopped',
                'ref_count': len([t for t in stream.callbacks]),
                'frame_count': stream.frame_count
            }