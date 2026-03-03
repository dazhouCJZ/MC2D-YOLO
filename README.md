# MC2D-YOLO
基于低计算双向状态空间模型的低对比微小病灶检测算法
{PROJECT_DESCRIPTION}

## 🎯 项目概况

医学影像微小病灶检测对疾病的早期筛查、精准定位及临床决策具有重要意义。然而，针对比度低和微小病灶检测，现有方法在计算开销和全局语义建模方面仍存在不足。

对此，提出一种低计算双向状态空间模型的低对比微小病灶精准检测算法。首先，设计低计算双向状态空间模型，以线性复杂度获得长程依赖信息，并在输出端生成全局状态分量作为状态信号，解决病灶检测全局语义建模不足和计算开销大的问题。

然后，提出状态感知动态多尺度卷积，利用全局状态信号引导动态参数合成，并通过内容自适应的动态核聚合机制，提升低对比和小病灶的检测能力。此外，设计动态多尺度状态空间特征融合模块，实现局部多尺度特征与全局上下文语义的协同建模，增强模型在复杂医学影像场景下的检测性能。

实验结果表明，所提算法以低计算代价取得了良好的检测效果，并较好地权衡了检测精度与运行效率。

## 📋 要求

| 项目            | 配置                                              |
|---------------|-------------------------------------------------|
| **Python 版本** | 3.10                                            |
| **深度学习框架**    | PyTorch 2.1.1                                   |
| **操作系统**      | Ubuntu 11.4.0                                   |
| **CUDA 版本**   | 12.1                                            |
| **GPU**       | NVIDIA RTX 4090（24GB） × 1                       |
| **CPU**       | Intel(R) Xeon(R) Gold 5418Y                     |
| **主要依赖**      | Ultralytics（YOLO8 修改版）、TorchVision、mamba-ssm、NumPy、OpenCV|

## 🚀  快速入门

### 安装

conda create -n mamba-yolo python=3.10.0
conda activate mamba-yolo
pip install -r requirements.txt 

## 📊 数据集

### 数据集信息

- **名称**: DeepLesion
- **来源**: NIH临床中心公开数据集
- **尺寸**: 512*512
- **类别**: 8（ 0: bone,1: abdomen,2: mediastinum,3: liver,4: lung,5: kidney,6: soft tissue,7: pelvis）
- **分布**: 训练集 (70%) / 验证集 (10%) / 测试集 (20%)

- **名称**: LUNA16
- **来源**: LIDC-IDRI数据库筛选集
- **尺寸**: 285*285
- **类别**: 1（ 0: Lungnodule）
- **分布**: 训练集 (70%) / 验证集 (10%) / 测试集 (20%)

### 数据集结构

```
MC2D-YOLO
├── datasets
│   ├── DeepLesion
│   │   ├── images
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   ├── labels
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
│   ├── LUNA16
│   │   ├── images
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   ├── labels
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
```

## 🏗️ 模型架构

### 网络结构

```
# Backbone
Input (3, 640, 640)
    ↓
Conv [64, 3×3, stride=2] → (64, 320, 320)          # P1/2
    ↓
Conv [128, 3×3, stride=2] → (128, 160, 160)        # P2/4
    ↓
C2f_DMFV ×3 → (128, 160, 160)
    ↓
Conv [256, 3×3, stride=2] → (256, 80, 80)          # P3/8
    ↓
C2f_DMFV ×6 → (256, 80, 80)
    ↓
Conv [512, 3×3, stride=2] → (512, 40, 40)          # P4/16
    ↓
C2f_DMFV ×6 → (512, 40, 40)
    ↓
Conv [1024, 3×3, stride=2] → (1024, 20, 20)        # P5/32
    ↓
C2f_DMFV ×3 → (1024, 20, 20)
    ↓
SPPF [1024, k=5] → (1024, 20, 20)


# Head
# P5/32 → P4/16 融合路径
Backbone P5 [1024, 20, 20]
    ↓
Upsample (×2) → (1024, 40, 40)
    ↓
Concat with Backbone P4 [512, 40, 40]
    ↓
C2f_DMFV ×3 → (512, 40, 40)

# P4/16 → P3/8 融合路径
P4 Head Output [512, 40, 40]
    ↓
Upsample (×2) → (512, 80, 80)
    ↓
Concat with Backbone P3 [256, 80, 80]
    ↓
C2f_DMFV ×3 → (256, 80, 80)          ← P3 (Small)

# 自底向上增强路径
P3 [256, 80, 80]
    ↓
Conv [256, 3×3, stride=2] → (256, 40, 40)
    ↓
Concat with P4 Head [512, 40, 40]
    ↓
C2f_DMFV ×3 → (512, 40, 40)          ← P4 (Medium)

P4 [512, 40, 40]
    ↓
Conv [512, 3×3, stride=2] → (512, 20, 20)
    ↓
Concat with Backbone P5 [1024, 20, 20]
    ↓
C2f_DMFV ×3 → (1024, 20, 20)         ← P5 (Large)

# Detection Head
[P3, P4, P5] → Detect Head → {nc=1} classes
```

## 🎓 训练

### 训练脚本

```bash
python train.py/train_distill.py
```
> 💡 本项目设计并采用了两种训练策略以全面评估模型性能。第一种为标准监督训练策略，在不加载任何预训练权重的情况下，从零开始对改进后的YOLOv8-Mamba网络进行端到端训练，以验证网络结构本身在LUNA16病灶检测任务中的基础学习能力与收敛特性。第二种为基于知识蒸馏的训练策略，在标准训练的基础上引入性能更优的Teacher模型，通过特征层蒸馏方式对Student模型进行指导，在YOLO的多尺度输出特征上对齐Teacher与Student的特征表示，并将蒸馏损失与原生检测损失进行联合优化，从而在不增加推理复杂度的前提下进一步提升模型的特征表达能力与检测精度。

### 超参数

```yaml
# cfg.py
model:
  MODEL_YAML: yolov8_mamba.yaml
  #dataset YAML file
  data: Lesion.yaml/Lesion2.yaml

training:
  device: 0
  imgsz: 640
  batch: 4
  epochs: 300
  patience: 50
  workers: 0
  amp: False
  save_period: 10
  resume: False

  # ===== Knowledge Distillation Specific Settings =====
  distillation: True                 # 启用知识蒸馏训练
  teacher_weights: best.pt           # Teacher模型权重路径
  distill_type: feature              # 蒸馏方式：feature-based
  distill_loss: MSELoss              # 蒸馏损失函数
  distill_weight: 2.0                # 蒸馏损失权重λ
  distill_layers: [P3, P4, P5]       # 参与蒸馏的多尺度特征层
  freeze_teacher: True               # Teacher模型参数冻结
```

## 🔮 验证与评估

### 运行测试与可视化

```bash
python test_DeepLesion.py/test_Luna16.py
```
> 💡该代码首先动态修复DeepLesion/LUAN16的类别映射，然后进行指标计算，输出Precision,Recall,F1-Score,mAP50,mAP50-95等关键数据，在predict_visual目录下生成带有边界框和置信度的测试集预测图片，并且生成predictions.json，供后续COCO评估使用。

### 预测结果格式修复

```bash
python coco_fix_prediction_json_DeepLession.py/coco_fix_prediction_json_LUNA16.py
```
> 💡由于YOLO导出的predictions.json中的image_id通常与标准COCO标注的数字ID不匹配，需要运行该代码进行修复，其修复逻辑就是将预测文件中的文件名ID映射回COCO标注文件中的整数ID。

### 标准COCO指标与TIDE误差分析

```bash
python coco.py
```
> 💡最后运行coco.py脚本进行权威的性能评估，该步骤集成了pycocotools和TIDE工具，通过输出最严谨的mAP@[.5:.95]指标并自动分析模型误差来源如分类错误、定位偏移及背景误检等生成分析柱状图，为论文提供深度的定量与定性数据支撑。

