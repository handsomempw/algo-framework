import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import threading
import logging
import time
from shapely.geometry import Polygon, box
from datetime import datetime
import json
import queue

from get_task import watch_tasks
from save_file import create_file
from save_json import save_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置管理类
class Config:
    TRITON_SERVERS = [
        # '192.168.96.136:8301',
        '192.168.96.136:8942',
    ]
    ALGORITHM_TYPES = [
        # 'image-object-detection',
        'garbage-detection',
    ]

# 任务类
class InferenceTask:
    def __init__(self, rtsp_path, score_threshold, triton_server_url, iou, label, bounding_box,
                 algorithm_type_name, camera_id, algorithm_type, bounding_box_type, task_id, created_at, stream_manager, index):
        """
        初始化对象

        Args:
            rtsp_path (str): RTSP路径
            score_threshold (float): 分数阈值
            triton_server_url (str): Triton服务器URL
            iou (float): 交并比阈值
            label (str): 标签
            bounding_box (list): 边界框坐标列表
            algorithm_type_name (str): 算法类型名称
            camera_id (int): 摄像头ID
            algorithm_type (int): 算法类型
            bounding_box_type (int): 边界框类型(矩形，多边形)
            task_id (int): 任务ID
            created_at (datetime): 任务创建时间
            stream_manager (StreamManager): 流管理器对象
            index (int): 同一个任务的rtsp流的索引

        Returns:
            None
        """
        self.rtsp_path = rtsp_path
        self.score_threshold = score_threshold
        self.triton_server_url = triton_server_url
        self.iou = iou
        self.label = label
        self.bounding_box = bounding_box
        self.index = index
        self.algorithm_type_name = algorithm_type_name
        self.camera_id = camera_id
        self.algorithm_type = algorithm_type
        self.bounding_box_type = bounding_box_type
        self.task_id = task_id
        self.created_at = created_at

        self.stream_manager = stream_manager

    def run(self):
        outputs = [
            grpcclient.InferRequestedOutput('classes'),
            grpcclient.InferRequestedOutput('scores'),
            grpcclient.InferRequestedOutput('bboxes'),
            grpcclient.InferRequestedOutput("labels")
        ]

        triton_client = grpcclient.InferenceServerClient(url=self.triton_server_url)
        # 等待帧队列准备
        self.stream_manager.frame_ready_event[self.rtsp_path].wait()
        while True:
            frame = self.stream_manager.frame_queues[self.rtsp_path].get()
            image = frame.transpose((1, 0, 2))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = [
                grpcclient.InferInput('image', image.shape, "UINT8"),
                grpcclient.InferInput('score', [1], "FP16")
            ]
            inputs[0].set_data_from_numpy(image)
            inputs[1].set_data_from_numpy(np.array([self.score_threshold / 100], dtype=np.float16))

            try:
                infer_result = triton_client.infer('base', inputs=inputs, outputs=outputs)
                self.process_results(infer_result, frame)
            except Exception as e:
                logging.error(f"Error during inference for {self.rtsp_path}: {e}")

    def get_current_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def convert_rectangle_to_vertices(self, boxes_xyxy):
        x1, y1, x2, y2 = boxes_xyxy
        vertices = [
            [x1, y1],  # 左上角
            [x2, y1],  # 右上角
            [x2, y2],   # 右下角
            [x1, y2],  # 左下角
        ]
        return vertices

    def process_results(self, infer_result, frame):
        bboxes = infer_result.as_numpy('bboxes')
        scores = infer_result.as_numpy('scores')
        classes = infer_result.as_numpy('classes')
        labels = infer_result.as_numpy('labels')

        # 保留大于iou阈值的数据
        remain_bboxes = []
        remain_labels = []
        remain_scores = []
        remain_ious = []
        for i in range(len(bboxes)):
            bboxes_iou = self.calculate_iou(self.bounding_box, bboxes[i])
            if bboxes_iou > self.iou / 100  and (labels[i].decode('utf-8') in self.label):
                remain_bboxes.append(bboxes[i])
                remain_labels.append(labels[i])
                remain_scores.append(scores[i])
                remain_ious.append(bboxes_iou)
                # logging.info(f"camera_index: {self.index}   label: ['{labels[i].decode('utf-8')}']    "
                #              f"class: [{classes[i]}]    score: [{round(scores[i], 4)}]    bbox: {bboxes[i]}")

        # 保存iou过滤后的图片
        if len(remain_bboxes) > 0:
            frame_name = str(time.time()).replace('.', '')
            create_file_result = create_file(frame_name, frame)
            logging.info(f"save image successfullly: {frame_name}")
        
            # 保存json数据
            data_json = {
                "events": [
                    {
                        "id": create_file_result["id"],
                        "name": "event",
                        "camera_id": self.camera_id,
                        "content": {
                            "algorithm_type": self.algorithm_type,
                            "score": round(float(remain_scores[0]), 2),
                            "label": remain_labels[0].decode('utf-8'),
                            "iou": round(float(remain_ious[0]) * 100, 2),
                            "bounding_box": self.convert_rectangle_to_vertices(remain_bboxes[0].tolist()),
                            "image_path": f"{create_file_result['parent_path']}/{create_file_result['name']}",
                            "target_image_path":f"{create_file_result['parent_path']}/{create_file_result['name']}"
                        },
                        "config": {
                            "algorithm_type": self.algorithm_type,
                            "score": self.score_threshold,
                            "iou": self.iou,
                            "label": self.label,
                            "bounding_box": self.bounding_box,
                            "bounding_box_type": self.bounding_box_type
                        },
                        "task_id": self.task_id,
                        "algorithm_type": self.algorithm_type,
                        "created_at": self.get_current_time(),
                        "analysis_at": self.get_current_time(),
                    }
                ]
            }
            save_data(data_json)

            filename = 'data.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_json, f, ensure_ascii=False, indent=4)

    def calculate_iou(self, polygon, rectangle):
        """
        计算两个几何形状的交并比（Intersection over Union, IoU）。

        Args:
            polygon (list of tuple): 表示多边形的顶点坐标列表，每个顶点坐标由(x, y)表示。
            rectangle (tuple): 表示矩形的坐标和尺寸，格式为(x_min, y_min, width, height)。

        Returns:
            float: 检测框和交集区域的交并比（IoU）。

        """
        poly_shape = Polygon(polygon)
        rect_shape = box(*rectangle)
        intersection = poly_shape.intersection(rect_shape)
        intersection_area = intersection.area
        rectangle_area = rect_shape.area
        iou = intersection_area / rectangle_area
        return iou

# RTSP流管理类
class RTSPStreamManager:
    """
    Attributes:
        capture_threads (dict): 用于存储捕获线程的字典，键为线程标识，值为线程对象。
        frame_queues (dict): 用于存储帧队列的字典，键为线程标识，值为队列对象。
        frame_ready_event (dict): 用于存储帧就绪事件的字典，键为线程标识，值为事件对象。
    """
    def __init__(self):
        self.capture_threads = {}
        self.frame_queues = {}
        self.frame_ready_event = {}

    def capture_frames(self, rtsp_path):
        """读取RTSP流帧并存储到共享队列中"""
        cap = cv2.VideoCapture(rtsp_path)
        logging.info(f"{rtsp_path}-opened successfully.")
        frame_queue = queue.Queue(maxsize=1)
        self.frame_queues[rtsp_path] = frame_queue
        self.frame_ready_event[rtsp_path].set()  # 标记RTSP流的队列已准备好
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 保证队列中只保存最新帧
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
        cap.release()

    def start_capture_thread(self, rtsp_path):
        """启动新的捕获线程，如果该RTSP流还没有被打开"""
        if rtsp_path not in self.capture_threads:
            self.frame_ready_event[rtsp_path] = threading.Event()  # 为每个RTSP流初始化事件
            capture_thread = threading.Thread(target=self.capture_frames, args=(rtsp_path,), daemon=True)
            self.capture_threads[rtsp_path] = capture_thread
            capture_thread.start()
    
    def stop_capture_thread(self, task_status):
        disabled_keys = [key for key, value in task_status.items() if all(item == '禁用' for item in value)]
        for stop_rtsp in disabled_keys:
            if stop_rtsp in self.capture_threads:
                self.frame_ready_event[stop_rtsp].clear()
                self.capture_threads[stop_rtsp].join(timeout=5)
                del self.capture_threads[stop_rtsp]

# 主程序
class MainProgram:
    """
    Attributes:
        all_threads (list): 存储所有任务线程的列表。
        task_status (dict): 存储任务状态的字典{rtsp:["状态"]}。
        all_threads_lock (threading.Lock): 用于控制对all_threads的访问的锁。
        watcher_thread_started (bool): 表示观察者线程是否已启动的标志。
        stream_manager (RTSPStreamManager): RTSP流管理器实例。
    """
    def __init__(self):
        self.all_threads = []
        self.task_status = {}
        self.all_threads_lock = threading.Lock()
        self.watcher_thread_started = False
        self.stream_manager = RTSPStreamManager()

    def update_tasks(self, rtsp_list, task_list, triton_server_url):
        """
        更新任务列表。

        Args:
            rtsp_list (list): RTSP流列表。
            task_list (list): 任务列表。
            triton_server_url (str): Triton推理服务器URL。

        Returns:
            None

        """
        with self.all_threads_lock:  # 使用锁保护all_threads列表
            for index, rtsp_path in enumerate(rtsp_list):
                task_id = task_list[index]["id"]
                # 更新任务状态
                if rtsp_path in self.task_status:
                    self.task_status[rtsp_path].append(task_list[index]["status"])
                else:
                    self.task_status[rtsp_path] = [task_list[index]["status"]]

                # 启动RTSP流线程
                if task_list[index]["status"] != "禁用":
                    self.stream_manager.start_capture_thread(rtsp_path)
                
                # 停止已禁用的rtsp线程
                self.stream_manager.stop_capture_thread(self.task_status)

                # 启动新的任务线程
                if not any(thread.name == task_id and thread.is_alive() for thread in self.all_threads) and \
                        task_list[index]["status"] == "启用":
                    score = task_list[index]["config"]["score"]
                    iou = task_list[index]["config"]["iou"]
                    label = task_list[index]["config"]["label"]
                    bounding_box = task_list[index]["config"]["bounding_box"]
                    algorithm_type_name = task_list[index]["name"]
                    camera_id = task_list[index]["camera_id"]
                    algorithm_type = task_list[index]["algorithm_type"]
                    bounding_box_type = task_list[index]["config"]["bounding_box_type"]
                    created_at = task_list[index]["created_at"]
                    task = InferenceTask(rtsp_path, score, triton_server_url, iou, label, bounding_box, 
                                        algorithm_type_name, camera_id, algorithm_type, bounding_box_type, 
                                        task_id, created_at, self.stream_manager, index)
                    thread = threading.Thread(target=task.run, name=task_id, daemon=True)
                    
                    self.all_threads.append(thread)
                    thread.start()

                # 停止禁用的任务线程
                elif any(thread.name == task_id for thread in self.all_threads) and task_list[index]["status"] == "禁用":
                    for thread in self.all_threads:
                        if thread.name == task_id:
                            task.stop()
                            thread.join(timeout=5)
                            self.all_threads.remove(thread)
                            logging.info(f"Stopped task: {task_id}")

    def start_watcher_threads(self):
        """启动监听线程读取任务列表"""
        if not self.watcher_thread_started:
            for algorithm_type, triton_server_url in zip(Config.ALGORITHM_TYPES, Config.TRITON_SERVERS):
                interval = 5
                watcher_thread = threading.Thread(
                    target=watch_tasks,
                    args=(algorithm_type, self.update_tasks, triton_server_url, interval))
                watcher_thread.start()
            self.watcher_thread_started = True

    def run(self):
        self.start_watcher_threads()

if __name__ == '__main__':
    program = MainProgram()
    program.run()
