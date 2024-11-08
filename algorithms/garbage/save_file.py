import http.client
from codecs import encode
import cv2
import json
import numpy as np


def create_file(file_name, image):
    """
    将图像文件上传到指定的服务器并返回响应结果。

    Args:
        file_name (str): 要上传的文件名（不包含扩展名）。
        image (str): 要上传的图像文件的路径。

    Returns:
        str: 服务器返回的响应结果。
    """
    jwt_token  = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhaXNwIiwic3ViIjoiYWRtaW4iLCJhdWQiOiJhaS1zZXJ2aWNlLXBsYXRmb3JtLXNpdGUiLCJpZCI6IjcwOTMwNjkxMzIxNjU0MTkwMDkiLCJhcGlfc2VjcmV0X2lkIjoiNzI1OTA5NTk5MjY0MTYxNzkyMSIsIm5hbWUiOiJhZG1pbiIsImFjY291bnQiOiJhZG1pbiIsImVtYWlsIjoiYWRtaW5AYnl3aW4uY29tIiwibW9kZSI6IkFQSeWvhumSpSIsImlhdCI6MTczMDczMjE1MiwiZXhwIjo0ODUyNzk2MTUyfQ.K3GCNfW9cSDT10LIZg0YtpxMoYKfnbhUw_Bw8gGYDfY'
    conn = http.client.HTTPConnection("192.168.100.137", 30337)
    # 设置boundary用于multipart/form-data
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList = []

    # 读取图像文件并将其编码为二进制数据
    is_success, buffer = cv2.imencode('.jpg', image)
    if not is_success:
        raise ValueError("Could not encode image.")
    binary_data = np.array(buffer).tobytes()

    # 添加字段 "name"
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=name;'))
    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))  # 空行
    dataList.append(encode(file_name + ".jpg"))

    # 文件类型 "type"
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=type;'))
    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))  # 空行
    dataList.append(encode('文件'))

    # 父级路径
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=parent_path'))
    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))  # 空行
    dataList.append(encode('app/aisp-video-compute-manager/image'))

    # 文件字段，传入image "file"
    dataList.append(encode('--' + boundary))
    dataList.append(encode(f'Content-Disposition: form-data; name=file; filename={file_name}.jpg  '))
    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))  # 空行
    dataList.append(binary_data)

    # 结束 boundary
    dataList.append(encode('--' + boundary + '--'))
    dataList.append(encode(''))

    body = b'\r\n'.join(dataList)
    headers = {
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary),
        'Authorization': 'Bearer {}'.format(jwt_token)
    }
    conn.request("POST", "/api/aisp-storage-center/v1/files", body, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data)

if __name__ == '__main__':
    file_name = "111111"
    image = "/workspace/workspace/wumh/wuminghui/4_Garbage_overflow_detection/test/114.jpg"
    image = cv2.imread(image)
    resuslt = create_file(file_name, image)
    print(resuslt)
