import os
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# 上传和结果保存路径
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 设置文件上传大小限制为 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# 预设模型路径
MODEL_PATHS = {
    "yolov8s": "YOLO_model/yolov8s.pt",
    "yolov8n": "YOLO_model/yolov8n.pt",
    "CNN": "YOLO_model/CNN.pth"
}

# 定义类别名称列表
class_names = [
    'Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah',
    'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat',
    'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish',
    'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies',
    'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon',
    'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp',
    'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise',
    'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra'
]

# CNN模型预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model' not in request.form:
        return "缺少文件或模型参数", 400

    file = request.files['file']
    model_name = request.form['model']

    if file.filename == '':
        return "未选择文件", 400
    if model_name not in MODEL_PATHS:
        return "无效的模型选择", 400

    # 保存上传文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 处理图片
    if file.content_type.startswith("image"):
        if model_name == "CNN":
            # 加载 CNN 模型
            model = models.resnet50(weights=None)  # 不加载预训练权重
            model.fc = nn.Linear(model.fc.in_features, 80)  # 修改最后一层，输出类别数为 80
            model.load_state_dict(torch.load(MODEL_PATHS["CNN"], map_location=torch.device('cpu')))
            model.eval()

            # 预处理图片
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            # 推理
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()  # 获取预测的类别标签

            # 将预测的标签映射到类别名称
            predicted_class = class_names[predicted_label]

            # 在图片上添加预测标签
            draw = ImageDraw.Draw(img)
            try:
                # 加载更大的字体（需要系统中存在该字体文件）
                font = ImageFont.truetype("arial.ttf", 40)  # 字体大小为40
            except IOError:
                # 如果系统没有指定字体，则使用默认字体
                font = ImageFont.load_default()

            # 使用 textbbox 获取文本的边界框
            text = f"Predicted: {predicted_class}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]  # 计算文本宽度
            text_height = text_bbox[3] - text_bbox[1]  # 计算文本高度

            position = (10, 10)  # 文字位置 (x, y)
            draw.text(position, text, fill="red", font=font)

            # 保存结果
            result_path = os.path.join(RESULT_FOLDER, "result_" + filename)
            img.save(result_path)
            return f"/static/results/result_{filename}"  # 返回Web可访问的路径
        else:
            # 选择 YOLO 模型
            model = YOLO(MODEL_PATHS[model_name])  # 定义 YOLO 模型
            results = model(filepath)
            result_path = os.path.join(RESULT_FOLDER, "result_" + filename)
            results[0].save(filename=result_path)
            return f"/static/results/result_{filename}"  # 返回Web可访问的路径

    # 处理视频
    elif file.content_type.startswith("video"):
        if model_name == "CNN":
            return "CNN模型不支持视频处理", 400
        else:
            # 选择 YOLO 模型
            model = YOLO(MODEL_PATHS[model_name])  # 定义 YOLO 模型
            cap = cv2.VideoCapture(filepath)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 输出视频路径
            result_video_path = os.path.join(RESULT_FOLDER, "result_" + filename)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 修改编码器为 H.264
            out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                for result in results:
                    frame = result.plot()
                out.write(frame)

            cap.release()
            out.release()

            return f"/static/results/result_{filename}"  # 返回Web可访问的路径

    return "不支持的文件格式", 400

@app.route('/static/results/<filename>')
def get_result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)