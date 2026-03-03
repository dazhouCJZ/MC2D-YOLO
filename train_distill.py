import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# 定义蒸馏损失系数 (Distillation Weight)
DISTILL_WEIGHT = 2.0  # 可以根据效果调整，通常 1.0 - 5.0 之间


class KD_Trainer(DetectionTrainer):
    """
    自定义的训练器，继承自 YOLO 的 DetectionTrainer。
    增加了加载 Teacher 模型和计算蒸馏损失的功能。
    """

    def __init__(self, teacher_weights, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.teacher_weights = teacher_weights
        self.teacher = None
        self.distill_loss_fn = nn.MSELoss()  # 使用均方误差作为蒸馏损失

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        重写加载模型的方法，在此处顺便加载 Teacher 模型。
        """
        # 1. 加载正常的 Student 模型
        model = super().get_model(cfg, weights, verbose)

        # 2. 加载 Teacher 模型
        print(f"Loading Teacher model from {self.teacher_weights}...")
        # 注意：这里我们要加载完整的模型结构
        teacher_model = YOLO(self.teacher_weights).model

        # 将 Teacher 设为评估模式，并冻结参数（不更新 Teacher）
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        self.teacher = teacher_model.to(self.device)
        print("Teacher model loaded and frozen successfully.")

        return model

    def criterion(self, preds, batch):
        """
        重写损失计算函数：
        总损失 = 原生 YOLO 损失 + 蒸馏损失
        """
        # 1. 计算原本的 YOLO 损失 (Box, Cls, DFL)
        loss, loss_items = super().criterion(preds, batch)

        # 2. 计算蒸馏损失
        # 获取 Teacher 对当前 batch 图片的预测输出
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])

        # 计算特征层蒸馏损失 (Feature-based Distillation)
        # preds 和 teacher_preds 通常是包含 3 个不同尺度特征图的列表
        dist_loss = torch.tensor(0.0, device=self.device)

        # 遍历三个输出层 (P3, P4, P5)
        for stud_feat, teach_feat in zip(preds, teacher_preds):
            # 确保尺寸一致（如果是同构蒸馏，尺寸必然一致）
            if stud_feat.shape == teach_feat.shape:
                dist_loss += self.distill_loss_fn(stud_feat, teach_feat)

        # 3. 将蒸馏损失加权合并
        total_loss = loss + (DISTILL_WEIGHT * dist_loss)

        # (可选) 更新 loss_items 以便在日志中看到蒸馏带来的变化，这里简单处理直接返回
        return total_loss, loss_items


# --- 主执行代码 ---
if __name__ == "__main__":
    # 配置参数
    args = dict(
        model=r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\cfg\yolov8m_mamba.yaml",
        data=r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\LUNA16\Lesion2.yaml",
        epochs=300,
        name="Lesion_LUNA16_1.9",
        workers=0,
        imgsz=640,
        batch=4,
        device=0,
        project="C:/Users/chenjunzhou/Desktop/project/yolo_mamba-main/yolo_mamba-main/logs/LUNA16",
        amp=False,
        save_period=10,
    )

    # 指向你已经训练好的最好模型作为 Teacher
    TEACHER_PATH = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16\LUNA16_yolov8_1.8\LUNA16_yolov8_1.8\weights\best.pt"

    # 初始化自定义训练器
    trainer = KD_Trainer(teacher_weights=TEACHER_PATH, overrides=args)

    # 开始训练
    trainer.train()