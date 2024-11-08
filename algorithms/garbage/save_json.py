# # # def draw(image, res, cls):
# # #     from PIL import Image, ImageDraw, ImageFont
# # #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #     pil_image = Image.fromarray(image)
# # #     draw = ImageDraw.Draw(pil_image)
# # #     font_path = "/workspace/workspace/wumh/STKAITI.TTF" 
# # #     font_size = 20
# # #     font = ImageFont.truetype(font_path, font_size)
# # #     for i, r in enumerate(res):
# # #         x0, y0 = int(r[0]), int(r[1])
# # #         x1, y1 = int(r[2]), int(r[3])
# # #         if x1 < x0:
# # #             x0, x1 = x1, x0
# # #         if y1 < y0:
# # #             y0, y1 = y1, y0
# # #         draw.rectangle([(x0, y0), (x1, y1)], outline="green", width=2)
# # #         text = "{}:{}".format(cls[i].decode('utf-8'), round(float(r[4]), 2))
# # #         draw.text((max(10, x0), max(20, y0) - font_size), text, font=font, fill=(0, 0, 255))
# # #     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
# # #     return image

# # #    # -------------------------------绘图-------------------------------------
# # #         res = []
# # #         for i in range(len(remain_bboxes)):
# # #             bbox = list(remain_bboxes[i])
# # #             bbox.append(float(remain_scores[i]))
# # #             res.append(bbox)
# # #         detection_res = self.draw(frame, res, remain_labels)
# # #         output_path = "/workspace/workspace/wumh/wuminghui/4_Garbage_overflow_detection/result/"
# # #         if len(remain_bboxes) > 0:
# # #             cv2.imwrite(f'{output_path}{self.task_id}.jpg', detection_res)
# # #         # -------------------------------------------------------------------------


import requests
import json
import time
from datetime import datetime


def generate_snowflake_id():
    """
    生成一个基于时间戳的简单雪花ID
    """
    return str(int(time.time() * 1000))

def get_current_time():
    """
    获取当前时间的ISO格式字符串
    格式:YYYY-MM-DD HH:mm:ss
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def save_data(data_json):
    """
    保存数据到指定服务器。

    Args:
        data_json (dict): 需要保存的数据，以字典形式给出。

    Returns:
        None
    """
    url = "http://192.168.100.137:30337/api/aisp-video-compute-manager/v1/events/_batch_create"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps(data_json)
    requests.request("POST", url, headers=headers, data=payload)

if __name__ == "__main__":
    data_json = {
        "events": [
            {
                # 必需字段
                "id": generate_snowflake_id(),  # 雪花ID
                "name": "test123",                        # 事件名称
                "camera_id": generate_snowflake_id(),      # 摄像头ID
                "content": {                               # 事件数据
                    "algorithm_type": "image-object-detection",
                    "score": 0.95,
                    "label": "person",
                    "iou": 0.8,
                    "bounding_box": [[390, 263], [1507, 263], [1507, 770], [390, 770]],
                    # "target_image_path": "app/aisp-video-compute-manager/image/1731032261840811.jpg",
                    "image_path": "app/aisp-video-compute-manager/image/1854721181547106304.jpg"
                },
                "config": {                               # 触发配置数据
                    "algorithm_type": "image-object-detection",
                    "score": 0.8,
                    "iou": 0.6,
                    "label": ["person"],
                    "bounding_box": [[852, 640], [1092, 640], [1092, 958], [852, 958]],
                    "bounding_box_type": "矩形"
                },
                "task_id": generate_snowflake_id(),       # 触发算法任务id
                "algorithm_type": "image-object-detection", # 算法类型
                "created_at": get_current_time(),          # 创建时间
                "analysis_at": get_current_time()          # 分析完成时间
            }
        ]
        }
    save_data(data_json)
