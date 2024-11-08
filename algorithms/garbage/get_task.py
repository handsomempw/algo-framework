import time
import hashlib
import http.client
import json

def get_task_data():
    conn = http.client.HTTPConnection("192.168.100.137", 30337)
    headers = {'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'}
    conn.request("GET", "/api/aisp-video-compute-manager/v1/tasks", '', headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data), headers


def watch_tasks(algorithm_type, callback, triton_server_url, interval=5):
    """
    监控指定算法类型的任务，并在任务发生变化时回调。

    Args:
        algorithm_type (str): 算法类型。
        callback (function): 回调函数，当任务发生变化时调用。回调函数接收两个参数：rtsp_list和task_list。
        interval (int, optional): 监控任务的间隔时间，默认为5秒。

    Returns:
        None

    """
    last_data_hash = None
    while True:
        data, headers = get_task_data()
        data_hash = hashlib.md5(json.dumps(data).encode()).hexdigest()
        
        if data_hash != last_data_hash:
            last_data_hash = data_hash
            rtsp_list, task_list = [], []
            for item in data:
                if item['algorithm_type'] == algorithm_type:
                    camera_id = item['camera_id']
                    task_list.append(item)
                    conn = http.client.HTTPConnection("192.168.100.137", 30337)
                    conn.request("POST", f"/api/aisp-video-compute-manager/v1/cameras/{camera_id}/_play", '', headers)
                    res = conn.getresponse()
                    res_data = res.read().decode()
                    try:
                        res_json = json.loads(res_data)
                        rtsp_url = res_json['rtsp_url']
                        rtsp_list.append(rtsp_url.replace("aisp-video-compute-manager:554", "192.168.96.136:30028"))
                    except KeyError:
                        print(f"Key 'rtsp_url' not found in the response data: {res_data}")
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Response data: {res_data}")
            callback(rtsp_list, task_list, triton_server_url)
        
        time.sleep(interval)


if __name__ == '__main__':
   def my_callback(rtsp_list, task_list):
      print("RTSP URLs:")
      for rtsp_url in rtsp_list:
         print(rtsp_url)
    #   print("\nTasks:")
    #   for task in task_list:
    #      print(task)

   algorithm_type = 'image-object-detection'
   interval = 5
   watch_tasks(algorithm_type, my_callback, interval)

