import cv2
import numpy as np
import onnxruntime as ort
import socket
import pyaudio
import threading
# 配置
#server_address = ('192.168.105.93', 9999)
server_address = ('', )#填写你的ip和端口
confidence_thres = 0.35
iou_thres = 0.5
classes = {0: 'smoke', 1: 'fire'}
color_palette = np.random.uniform(100, 255, size=(len(classes), 3))
# 添加音频设置
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
def preprocess(img, input_width, input_height):
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, img_height, img_width

def calculate_iou(box, other_boxes):
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = box[2] * box[3]
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou

def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    if len(boxes) == 0:
        return []
    scores = np.array(scores)
    boxes = np.array(boxes)
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    if len(filtered_boxes) == 0:
        return []

    sorted_indices = np.argsort(filtered_scores)[::-1]
    indices = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        indices.append(current_index)
        if len(sorted_indices) == 1:
            break
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        iou = calculate_iou(current_box, other_boxes)
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[non_overlapping_indices + 1]

    return indices

def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box
    color = color_palette[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 3)
    label = f'{classes[class_id]}: {score:.2f}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                  cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
def postprocess(input_image, output, input_width, input_height, img_width, img_height):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    centers = []  # 新增列表用于保存中心点
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][:4]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

            # 计算中心点
            center_x = left + width // 2
            center_y = top + height // 2
            centers.append((center_x, center_y))  # 保存中心点

    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        draw_detections(input_image, box, score, class_id)

    return input_image, class_ids, centers  # 返回中心点

def init_detect_model(model_path):
    session = ort.InferenceSession(model_path)
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]
    return session, model_inputs, input_width, input_height

def detect_object(image, session, model_inputs, input_width, input_height):
    img_data, img_height, img_width = preprocess(image, input_width, input_height)
    outputs = session.run(None, {model_inputs[0].name: img_data})
    output_image, class_ids, centers = postprocess(image, outputs, input_width, input_height, img_width, img_height)
    return output_image, class_ids, centers  # 返回中心点

def audio_sender(socket_conn):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("语音发送线程启动...")
    while True:
        try:
            data = stream.read(CHUNK)
            if socket_conn.fileno() == -1:  # 检查连接是否仍然有效
                print("Socket 已关闭，停止发送音频。")
                break

            socket_conn.sendall(b'b' + len(data).to_bytes(4, byteorder='big') + data)
            print(f"语音发送")

        except Exception as e:
            print(f"语音发送错误: {e}")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
def detect_objects_from_camera(model_path):
    session, model_inputs, input_width, input_height = init_detect_model(model_path)
    cap = cv2.VideoCapture(0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Connecting to server...")
        s.connect(server_address)
        print("Connected to server.")

        # 启动语音发送线程
        audio_thread = threading.Thread(target=audio_sender, args=(s,), daemon=True)

        audio_thread.start()

        while True:

            print(".")
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # 进行对象检测
            output_image, class_ids, centers = detect_object(frame, session, model_inputs, input_width, input_height)

            if class_ids:
                print(f"Detected class IDs: {class_ids}")

                # 序列化输出图像和类别 ID
                _, encoded_image = cv2.imencode('.jpg', output_image)
                image_data = encoded_image.tobytes()
                image_size = len(image_data)

                class_data = ','.join(map(str, class_ids))  # 类别数据
                center_data = ';'.join(f"{x},{y}" for x, y in centers)  # 中心点数据
                # 将类别数据和中心点数据合并
                # 将字符串转换为字节
                combined_data = (
                        b"a" +
                        class_data.encode('utf-8') + b"\n" +  # 将 class_data 转换为字节
                        center_data.encode('utf-8') + b"\n" +  # 将 center_data 转换为字节
                        image_size.to_bytes(4, byteorder='big') +  # 图像大小作为字节
                        image_data + b"\n"  # 图像数据，假设 image_data 本身是字节类型
                ) # 使用换行符分隔
                # 发送合并后的数据
                s.sendall(combined_data)
                #print(f"Sent combined data: {combined_data}")
                print(f"Sent class IDs: {class_ids}, centers: {centers}")

                # # 发送图像数据长度
                # s.sendall(image_size.to_bytes(4, byteorder='big'))  # 发送图像数据大小
                print(f"Sent image size: {image_size} bytes   {image_size.to_bytes(4, byteorder='big')}")

                # # 发送图像数据
                # s.sendall(image_data)
                print("Sent image data.")
            else:
                # 发送空检测结果
                # s.sendall(b'a')

                class_ids = [2]  # 示例数据
                _, encoded_image = cv2.imencode('.jpg', frame)
                image_data = encoded_image.tobytes()
                image_size = len(image_data)

                # 发送类别数据和中心点
                class_data = ','.join(map(str, class_ids))  # 类别数据
                center_data = '0,0'  # 中心点数据

                combined_data = (
                        b"a" +
                        class_data.encode('utf-8') + b"\n" +  # 将 class_data 转换为字节
                        center_data.encode('utf-8') + b"\n" +  # 将 center_data 转换为字节
                        image_size.to_bytes(4, byteorder='big') +  # 图像大小作为字节
                        image_data + b"\n"  # 图像数据，假设 image_data 本身是字节类型
                )
                # 发送合并后的数据
                s.sendall(combined_data)
                #print(f"Sent combined data: {combined_data}")
                print(f"Sent class IDs: {class_ids}, centers: (0,0)")

                # # 发送图像数据长度
                # s.sendall(image_size.to_bytes(4, byteorder='big'))  # 发送图像数据大小
                print(f"Sent image size: {image_size} bytes   {image_size.to_bytes(4, byteorder='big')}")

                # # 发送图像数据
                # s.sendall(image_data)
                print("Sent image data.")



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 调用函数，传入模型路径
detect_objects_from_camera("best.onnx")
