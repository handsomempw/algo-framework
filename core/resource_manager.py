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
    """视频流信息"""
    url: str = "http://192.168.100.137:30337/api/aisp-video-compute-manager/v1/cameras/7199616326262984705/_play"
    ref_count: int = 0
    last_active: datetime = datetime.now()
    
class VideoStream:
    """视频流类"""
    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.frame_count = 0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.ref_count = 1
        self.logger = logging.getLogger(__name__)
        # 添加帧缓存
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.callbacks = []
        
    def initialize(self) -> bool:
        """初始化视频流"""
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.url)
            # 设置 RTSP 缓冲
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
        """读取视频帧的循环"""
        import cv2
        while self._running:
            try:
                if self.cap is None:
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                    
                with self._lock:
                    self.frame_count += 1
                    self._current_frame = frame.copy()
                    self._frame_ready.set()
                    
            except Exception as e:
                self.logger.error(f"读取视频帧异常: {str(e)}")
                time.sleep(0.1)
                
    def read(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.is_opened:
            return None
            
        # 等待新帧
        self._frame_ready.wait(timeout=1.0)
        with self._lock:
            frame = self._current_frame.copy() if self._current_frame is not None else None
            self._frame_ready.clear()
        return frame
        
    @property
    def is_running(self) -> bool:
        return self._running and self._thread and self._thread.is_alive()
        
    def release(self):
        """释放资源"""
        with self._lock:
            self.ref_count -= 1
            if self.ref_count <= 0 and self.cap:
                self.cap.release()
                self.cap = None
        
    @property
    def is_opened(self) -> bool:
        """视频流是否打开"""
        return self.cap is not None and self.cap.isOpened()

class ResourceManager:
    """资源管理器
    
    负责管理视频流等资源的生命周期
    """
    def __init__(self):
        self.streams: Dict[str, VideoStream] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def get_video_stream(self, url: str) -> Optional[VideoStream]:
        """获取视频流，如果不存在则创建"""
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
        """创建新的视频流"""
        return self.get_video_stream(url)
        
    def get_stream_status(self, url: str) -> Optional[dict]:
        """获取视频流状态"""
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