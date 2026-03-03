from ultralytics import YOLO
import torch.nn as nn

model = YOLO(r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\cfg\yolov8m_mamba.yaml")  # build a YOLOv8n model from scratch

model.info()  # display model information

model.train(
    data=r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\LUNA16\Lesion2.yaml",
    epochs=300,
    name="LUNA16_yolov8_3.2",
    workers=0,
    imgsz=640,
    batch=4,
    device=0,
    pretrained=False,
    save_period=10,
    amp=False,
    project="C:/Users/chenjunzhou/Desktop/project/yolo_mamba-main/yolo_mamba-main/logs/LUNA16/LUNA16_yolov8_3.2",
)