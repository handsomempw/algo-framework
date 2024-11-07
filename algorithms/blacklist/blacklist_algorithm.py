# 标准库导入
from typing import Dict, Any
import numpy as np
import cv2
import base64
import logging
import time
from datetime import datetime

# 内部模块导入
from algorithms.base_algorithm import BaseAlgorithm, AlgorithmStatus
from algorithms.blacklist.face_api import FaceAPI

class BlacklistAlgorithm(BaseAlgorithm):
    """黑名单检测算法类
    
    继承自BaseAlgorithm,实现了黑名单人脸检测的核心功能。
    主要功能:
    1. 初始化人脸识别API连接
    2. 处理视频帧进行人脸检测
    3. 匹配黑名单库并返回结果
    """

    def __init__(self, algo_config: Dict[str, Any]):
        """
        初始化算法
        Args:
            algo_config: 算法配置,必须包含:
                - face_api_url: 人脸识别API地址
                - blacklist_db: 黑名单库名称
                - match_threshold: 匹配阈值
        """
        # 设置默认配置
        default_config = {
            'face_api_url': 'http://192.168.100.137:32316/api/aisp-face-center',
            'blacklist_db': 'aa4',
            'match_threshold': 0.7,
            'min_face_size': 100,
            'max_results': 5,
            'detection_threshold': 0.5,
            'api_timeout': (5, 30),
            'frame_interval': 1
        }
        
        # 更新配置
        self.algo_config = default_config.copy()
        self.algo_config.update(algo_config)
        
        # 添加性能统计属性
        self.performance_metrics = {
            'frame_count': 0,
            'process_times': [],
            'api_response_times': [],
            'face_counts': [],
            'match_counts': [],
            'start_time': datetime.now()
        }
        
        super().__init__(self.algo_config)
        self.logger = logging.getLogger(__name__)
        self.face_api = None
        self.match_threshold = self.algo_config.get('match_threshold', 0.7)
        self._frame_count = 0
        self._last_process_time = None
        
    def initialize(self) -> bool:
        """初始化算法资源"""
        try:
            # 验证必要配置
            if 'face_api_url' not in self.algo_config:
                raise ValueError("缺少face_api_url配置")
            if 'blacklist_db' not in self.algo_config:
                raise ValueError("缺少blacklist_db配置")
                
            # 初始化人脸识别API客户端
            self.face_api = FaceAPI(self.algo_config['face_api_url'])
            self.logger.info("黑名单算法初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.status = AlgorithmStatus.ERROR
            self.error_msg = str(e)
            return False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单帧图像"""
        start_time = time.time()
        self._frame_count += 1
        
        try:
            # 转换为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构造人脸搜索请求
            payload = {
                "db_name": self.algo_config['blacklist_db'],
                "image": image_base64,
                "detection_threshold": 0.5,
                "min_score": self.match_threshold,
                "min_size": self.algo_config.get('min_face_size', 100),
                "nprobe": 10,
                "top": self.algo_config.get('max_results', 5)
            }
            
            # 发送请求
            api_start_time = time.time()
            response = self.face_api.search_face(payload)
            api_time = time.time() - api_start_time
            
            # 更新API响应时间统计
            self.performance_metrics['api_response_times'].append(api_time)
            
            # 处理匹配结果
            matches = []
            if response and isinstance(response, list):
                for face in response:
                    match_info = self._process_face_result(face)
                    if match_info:
                        matches.append(match_info)
            
            # 计算处理时间
            process_time = time.time() - start_time
            self._last_process_time = process_time
            self.performance_metrics['process_times'].append(process_time)
            
            return {
                'matches': matches,
                'frame_info': {
                    'frame_id': self._frame_count,
                    'process_time': process_time,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"处理帧异常: {str(e)}")
            return {
                'matches': [],
                'frame_info': {
                    'frame_id': self._frame_count,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    def _log_performance_metrics(self):
        """输出性能统计信息"""
        metrics = self.performance_metrics
        elapsed_time = (datetime.now() - metrics['start_time']).total_seconds()
        
        self.logger.info(f"\n性能统计 (运行时间: {elapsed_time:.1f}秒):")
        self.logger.info(f"总处理帧数: {metrics['frame_count']}")
        self.logger.info(f"平均处理时间: {np.mean(metrics['process_times']):.3f}秒")
        self.logger.info(f"平均匹配数量: {np.mean(metrics['match_counts']):.2f}")

    def _process_face_result(self, face: dict) -> dict:
        """处理单个人脸识别结果"""
        try:
            # 解析人脸位置信息
            bbox = face.get('bbox', [])
            if not bbox or len(bbox) < 4:
                return None
                
            face_location = {
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3]),
                'confidence': face.get('confidence', 0)
            }
            
            # 获取匹配信息
            score = face.get('score', 0)
            if score >= self.match_threshold:
                return {
                    'face_id': face.get('id'),
                    'face_location': face_location,
                    'score': score,
                    'person_info': {
                        'name': face.get('name', ''),
                        'certificate_number': face.get('certificate_number', ''),
                        'info': face.get('info', {})
                    }
                }
            return None
            
        except Exception as e:
            self.logger.error(f"处理人脸结果异常: {str(e)}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """获取算法状态"""
        status = super().get_status()
        status.update({
            'frame_count': self._frame_count,
            'last_process_time': self._last_process_time,
            'match_threshold': self.match_threshold
        })
        return status

    def release(self) -> bool:
        """释放算法资源
        
        Returns:
            bool: 释放是否成功
        """
        try:
            # 清理人脸API连接
            if self.face_api:
                self.face_api = None
            
            # 重置计数器和时间
            self._frame_count = 0
            self._last_process_time = None
            
            self.logger.info("黑名单算法资源释放成功")
            return True
            
        except Exception as e:
            self.logger.error(f"释放资源失败: {str(e)}")
            return False