import requests
import json
from datetime import datetime, timezone, timedelta

def get_beijing_time():
    """获取北京时间"""
    utc_now = datetime.now(timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    beijing_time = utc_now.astimezone(beijing_tz)
    return beijing_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def generate_snowflake_id():
    """生成雪花ID"""
    return str(int(datetime.now().timestamp() * 1000))

def save_data(data_json):
    """保存数据到指定服务器"""
    url = "http://192.168.100.137:30337/api/aisp-video-compute-manager/v1/events/_batch_create"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps(data_json)
    response = requests.request("POST", url, headers=headers, data=payload)
    print(f"保存结果: {response.status_code}")
    return response

if __name__ == "__main__":
    # 构造垃圾检测事件数据
    data_json = {
        "events": [{
      "id": "7261634464813305857",
      "name": "event123",
      "camera_id": "7199616326262984705",
      "camera_name": "观乾",
      "camera_city_code": "",
      "camera_address": "",
      "camera_gb28181_id": "34020000001310000001",
      "running_side": {
        "running_side": "云平台"
      },
      "content": {
        "score": 1900.920295715332,
        "label": "垃圾",
        "iou": 55.283808933002476,
        "bounding_box": [
          [
            24.0,
            571.0
          ],
          [
            2504.0,
            571.0
          ],
          [
            2504.0,
            1403.0
          ],
          [
            24.0,
            1403.0
          ]
        ],
        "target_image_path": "app/aisp-video-compute-manager/image/17313085709281695.jpg",
        "image_path": "app/aisp-video-compute-manager/image/17313085709281695.jpg",
        "algorithm_type": "garbage-detection"
      },
      "config": {
        "algorithm_type": "garbage-detection",
        "score": 10.0,
        "iou": 10.0,
        "label": [
          "垃圾"
        ],
        "bounding_box": [
          [
            0,
            0
          ],
          [
            1400,
            0
          ],
          [
            1400,
            1400
          ],
          [
            0,
            1400
          ]
        ],
        "bounding_box_type": "矩形"
      },
      "task_id": "7261541328882200577",
      "algorithm_type": "garbage-detection",
      "algorithm_type_name": "垃圾满溢检测",
      "created_at": "2024-11-11 15:02:51.205",
      "analysis_at": "2024-11-11 15:02:51.205"
    }
  ]
}
    
    # 保存数据并打印结果
    print("开始保存事件数据...")
    print(f"事件数据: {json.dumps(data_json, ensure_ascii=False, indent=2)}")
    response = save_data(data_json)
    print("事件数据保存完成")
