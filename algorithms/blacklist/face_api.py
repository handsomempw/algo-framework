import requests
import logging
from typing import Dict, List, Optional, Union
import base64
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)

class FaceAPI:
    def __init__(self, base_url: str, timeout: tuple = (5, 10)):
        """
        初始化人脸识别API客户端
        :param base_url: API基础URL
        :param timeout: 请求超时设置，格式为 (连接超时, 读取超时)
        """
        if not base_url:
            raise ValueError("base_url 不能为空")
        
        # 验证URL格式
        parsed_url = urlparse(base_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("无效的 base_url 格式")
            
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def list_databases(self) -> List[Dict]:
        """获取人脸库列表"""
        try:
            url = f"{self.base_url}/v1/databases"
            response = requests.get(
                url, 
                headers=self.headers, 
                params={'page': 1, 'per_page': 50},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            logger.error(f"获取人脸库列表失败: {response.status_code}")
            return []
        except requests.Timeout:
            logger.error("获取人脸库列表超时")
            return []
        except Exception as e:
            logger.error(f"获取人脸库列表异常: {e}")
            return []

    def create_database(self, db_name: str, db_info: str = "") -> bool:
        """
        创建人脸库
        :param db_name: 人脸库名称
        :param db_info: 人脸库描述信息
        """
        if not db_name or not isinstance(db_name, str):
            raise ValueError("db_name 必须是非空字符串")
            
        try:
            url = f"{self.base_url}/v1/databases"
            payload = {"db_name": db_name, "db_info": db_info}
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"创建人脸库失败: {response.text}")
            return response.status_code == 200
        except requests.Timeout:
            logger.error("创建人脸库请求超时")
            return False
        except Exception as e:
            logger.error(f"创建人脸库异常: {e}")
            return False

    def delete_database(self, db_name: str) -> bool:
        """删除人脸库"""
        try:
            url = f"{self.base_url}/v1/databases/{db_name}"
            response = requests.delete(url, headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"删除人脸库异常: {e}")
            return False

    def add_face(self, db_name: str, image_path: str, person_info: Dict) -> Optional[str]:
        """
        添加人脸到指定库
        :param db_name: 人脸库名称
        :param image_path: 图片路径
        :param person_info: 人员信息字典
        """
        if not db_name or not isinstance(db_name, str):
            raise ValueError("db_name 必须是非空字符串")
        if not image_path or not isinstance(image_path, str):
            raise ValueError("image_path 必须是非空字符串")
        if not isinstance(person_info, dict):
            raise ValueError("person_info 必须是字典类型")
            
        try:
            url = f"{self.base_url}/v1/faces"
            
            # 读取并编码图片
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 构造请求数据
            payload = {
                "db_name": db_name,
                "detection_threshold": 0.5,
                "image": image_base64,
                "min_size": 100,
                "name": person_info.get('name', ''),
                "certificate_number": person_info.get('certificate_number', '000000000'),
                "info": person_info.get('info', '')
            }
            
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('id')
            logger.error(f"添加人脸失败: {response.text}")
            return None
        except requests.Timeout:
            logger.error("添加人脸请求超时")
            return None
        except Exception as e:
            logger.error(f"添加人脸异常: {e}")
            return None

    def search_face(self, payload: dict) -> List[Dict]:
        """搜索人脸
        
        Args:
            payload: 请求参数字典
                - db_name: 数据库名称
                - image: base64编码的图像数据
                - params: 搜索参数
        """
        try:
            url = f"{self.base_url}/v1/faces/_search"
            
            # 构造正确的请求格式
            search_payload = {
                "db_name": payload['db_name'],  # 必需参数
                "image": payload['image'],
                "detection_threshold": payload.get('detection_threshold', 0.5),
                "min_score": payload.get('min_score', 0.7),
                "min_size": payload.get('min_size', 100),
                "nprobe": payload.get('nprobe', 10),
                "top": payload.get('top', 5)
            }
            
            response = requests.post(
                url,
                json=search_payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('result', [])
                
            self.logger.error(f"搜索人脸失败: {response.status_code} - {response.text}")
            return []
            
        except Exception as e:
            self.logger.error(f"搜索人脸异常: {str(e)}")
            return []

    def get_face_info(self, face_id: str, db_name: str) -> Optional[Dict]:
        """获取人脸信息"""
        try:
            url = f"{self.base_url}/v1/faces/{face_id}/db_name/{db_name}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"获取人脸信息异常: {e}")
            return None

    def delete_face(self, face_id: str, db_name: str) -> bool:
        """删除人脸"""
        try:
            url = f"{self.base_url}/v1/faces/{face_id}/db_name/{db_name}"
            response = requests.delete(url, headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"删除人脸异常: {e}")
            return False 