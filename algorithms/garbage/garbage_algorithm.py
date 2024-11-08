"""垃圾检测算法模块

该模块实现了基于Triton推理服务器的垃圾检测算法。
主要功能:
1. 初始化与Triton服务器的连接
2. 处理视频帧进行垃圾检测
3. 计算检测框与目标区域的IoU
4. 保存检测结果和性能指标
"""

import cv2
import time
import logging
import numpy as np
import tritonclient.grpc as grpcclient
from datetime import datetime
from typing import Dict, Any, List
from shapely.geometry import Polygon, box

from algorithms.base_algorithm import BaseAlgorithm, AlgorithmStatus
from algorithms.garbage.save_file import create_file
from algorithms.garbage.save_json import save_data

class GarbageAlgorithm(BaseAlgorithm):
    """垃圾检测算法类
    
    继承自BaseAlgorithm,实现了垃圾检测的核心功能。
    主要功能:
    1. 初始化Triton客户端连接
    2. 处理视频帧进行目标检测
    3. 计算IoU并过滤结果
    4. 保存检测结果图片和事件数据
    """
    
    def __init__(self, algo_config: Dict[str, Any]):
        """初始化算法
        
        Args:
            algo_config: 算法配置,必须包含:
                - triton_url: Triton服务器地址
                - score_threshold: 检测阈值
                - iou_threshold: IoU阈值
                - target_labels: 目标检测类别
                - roi_area: 感兴趣区域坐标
        """
        # 设置默认配置
        default_config = {
            'triton_url': '192.168.96.136:8942',
            'score_threshold': 0.5,
            'iou_threshold': 0.5,
            'target_labels': ['garbage'],
            'roi_area': [[0, 0], [1, 0], [1, 1], [0, 1]],
            'model_name': 'garbage-detection',
            'frame_interval': 1
        }
        
        # 更新配置
        self.algo_config = default_config.copy()
        self.algo_config.update(algo_config)
        
        # 添加性能统计属性
        self.performance_metrics = {
            'frame_count': 0,
            'process_times': [],
            'inference_times': [],
            'detection_counts': [],
            'start_time': datetime.now()
        }
        
        super().__init__(self.algo_config)
        self.logger = logging.getLogger(__name__)
        self.triton_client = None
        self._frame_count = 0
        self._last_process_time = None
        self._running = True
        
        # 初始化Triton输入输出
        self.inputs = None  # 动态创建输入
        self.outputs = [
            grpcclient.InferRequestedOutput('classes'),
            grpcclient.InferRequestedOutput('scores'),
            grpcclient.InferRequestedOutput('bboxes'),
            grpcclient.InferRequestedOutput("labels")
        ]

    def initialize(self) -> bool:
        """初始化算法资源"""
        try:
            # 验证必要配置
            if 'triton_url' not in self.algo_config:
                raise ValueError("缺少triton_url配置")
                
            # 初始化Triton客户端
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.algo_config['triton_url']
            )
            self.logger.info("垃圾检测算法初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            self.status = AlgorithmStatus.ERROR
            self.error_msg = str(e)
            return False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单帧图像
        
        Args:
            frame: 输入图像帧,numpy数组格式(H,W,C)
            
        Returns:
            Dict: 处理结果,包含:
                - detections: 检测结果列表
                - frame_info: 帧处理信息
        """
        if not self._running:
            return {
                'detections': [],
                'frame_info': {
                    'frame_id': self._frame_count,
                    'status': 'stopped',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        start_time = time.time()
        self._frame_count += 1
        
        try:
            # 打印输入帧的形状
            self.logger.info(f"输入帧形状: {frame.shape}")
            
            # 图像预处理 - 参考client_rtsp_1.py的处理方式
            image = frame.transpose((1, 0, 2))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.logger.info(f"预处理后形状: {image.shape}")
            
            # 动态创建输入
            self.inputs = [
                grpcclient.InferInput('image', image.shape, "UINT8"),
                grpcclient.InferInput('score', [1], "FP16")
            ]
            
            # 设置输入数据
            self.inputs[0].set_data_from_numpy(image)
            self.inputs[1].set_data_from_numpy(
                np.array([self.algo_config['score_threshold'] / 100], dtype=np.float16)
            )
            
            # 执行推理前打印信息
            self.logger.info(f"使用模型名称: {self.algo_config['model_name']}")
            
            # 执行推理
            infer_start = time.time()
            response = self.triton_client.infer(
                self.algo_config['model_name'],
                inputs=self.inputs,
                outputs=self.outputs
            )
            infer_time = time.time() - infer_start
            
            # 处理推理结果
            self.logger.info("开始处理推理结果")
            bboxes = response.as_numpy('bboxes')
            scores = response.as_numpy('scores')
            classes = response.as_numpy('classes')
            labels = response.as_numpy('labels')
            self.logger.info(f"检测框形状: {bboxes.shape if bboxes is not None else 'None'}")
            
            # 过滤结果
            detections = []
            for i in range(len(bboxes)):
                if labels[i].decode('utf-8') in self.algo_config['target_labels']:
                    iou = self._calculate_iou(
                        self.algo_config['roi_area'],
                        bboxes[i]
                    )
                    if iou > self.algo_config['iou_threshold']:
                        detections.append({
                            'bbox': bboxes[i].tolist(),
                            'score': float(scores[i]),
                            'label': labels[i].decode('utf-8'),
                            'iou': float(iou)
                        })
            
            # 保存检测结果
            if detections:
                self._save_detection_result(frame, detections)
            
            # 更新性能指标
            process_time = time.time() - start_time
            self._last_process_time = process_time
            self.performance_metrics['process_times'].append(process_time)
            self.performance_metrics['inference_times'].append(infer_time)
            self.performance_metrics['detection_counts'].append(len(detections))
            
            return {
                'detections': detections,
                'frame_info': {
                    'frame_id': self._frame_count,
                    'process_time': process_time,
                    'inference_time': infer_time,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"处理帧异常: {str(e)}")
            self.logger.error(f"异常详细信息: ", exc_info=True)
            return None

    def _calculate_iou(self, polygon: List[List[float]], bbox: List[float]) -> float:
        """计算多边形与矩形框的IoU
        
        Args:
            polygon: 多边形顶点坐标列表
            bbox: 矩形框坐标[x1,y1,x2,y2]
            
        Returns:
            float: IoU值
        """
        try:
            poly_shape = Polygon(polygon)
            rect_shape = box(*bbox)
            intersection = poly_shape.intersection(rect_shape)
            return intersection.area / rect_shape.area
        except Exception as e:
            self.logger.error(f"计算IoU异常: {str(e)}")
            return 0.0

    def _save_detection_result(self, frame: np.ndarray, detections: List[Dict]):
        """保存检测结果
        
        Args:
            frame: 原始图像帧
            detections: 检测结果列表
        """
        try:
            # 保存图片
            frame_name = str(time.time()).replace('.', '')
            result = create_file(frame_name, frame)
            
            # 构造事件数据
            event_data = {
                "events": [{
                    "id": result["id"],
                    "name": "garbage_detection",
                    "camera_id": self.algo_config.get('camera_id'),
                    "content": {
                        "algorithm_type": self.algo_config.get('algorithm_type'),
                        "score": detections[0]['score'],
                        "label": detections[0]['label'],
                        "iou": detections[0]['iou'],
                        "bounding_box": detections[0]['bbox'],
                        "image_path": f"{result['parent_path']}/{result['name']}",
                        "target_image_path": f"{result['parent_path']}/{result['name']}"
                    },
                    "config": {
                        "algorithm_type": self.algo_config.get('algorithm_type'),
                        "score": self.algo_config['score_threshold'],
                        "iou": self.algo_config['iou_threshold'],
                        "label": self.algo_config['target_labels'],
                        "bounding_box": self.algo_config['roi_area'],
                        "bounding_box_type": self.algo_config.get('bounding_box_type', '矩形')
                    },
                    "task_id": self.algo_config.get('task_id'),
                    "algorithm_type": self.algo_config.get('algorithm_type'),
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "analysis_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }]
            }
            
            # 保存事件数据
            save_data(event_data)
            
        except Exception as e:
            self.logger.error(f"保存检测结果异常: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """获取算法状态"""
        status = super().get_status()
        status.update({
            'frame_count': self._frame_count,
            'last_process_time': self._last_process_time,
            'performance_metrics': {
                'avg_process_time': np.mean(self.performance_metrics['process_times']),
                'avg_inference_time': np.mean(self.performance_metrics['inference_times']),
                'avg_detections': np.mean(self.performance_metrics['detection_counts'])
            }
        })
        return status

    def release(self):
        """释放算法资源"""
        try:
            self._running = False
            time.sleep(0.1)  # 等待最后一帧处理完成
            
            if self.triton_client:
                self.triton_client = None
                
            self._frame_count = 0
            self._last_process_time = None
            
            self.logger.info("垃圾检测算法资源释放成功")
            return True
            
        except Exception as e:
            self.logger.error(f"释放资源失败: {str(e)}")
            return False