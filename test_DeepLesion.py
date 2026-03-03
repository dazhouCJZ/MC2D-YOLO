import matplotlib

matplotlib.use('Agg')  # ← 关键修复
import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    # model_path = './output_dir/mscoco/mambayolo_n/weights/best.pt'
    model_path = r"C:\Users\chenjunzhou\Desktop\研究生\论文（曼巴病灶）\logs\DeepLesion\mamba yolov8 16 C2DMSC xiaogai_ZN（4300）tbest（good）\best.pt"
    model = YOLO(model_path)  # 选择训练好的权重路径

    # 只适合DeepLession
    # --- 核心修复：绕过只读属性，直接修改底层字典 ---
    custom_names = {
        0: 'bone', 1: 'abdomen', 2: 'mediastinum', 3: 'liver',
        4: 'lung', 5: 'kidney', 6: 'soft tissue', 7: 'pelvis'
    }

    # 方法 A: 修改模型类实例的内部属性 (通常有效)
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        model.model.names = custom_names

    # 方法 B: 针对 Ultralytics 内部实例的属性赋值
    # 使用 setattr 的方式或者直接修改底层存储
    model.ckpt['model'].names = custom_names if hasattr(model, 'ckpt') else custom_names

    # 最关键的一步：同步到 predictor 中
    model.predictor = None  # 重置预测器，强制它下次运行时重新加载 model.names

    # 打印确认
    print(f"类别名已更新为: {custom_names}")
    # -----------------------------------------------

    # 打印确认一下，不再使用 model.overrides['names'] = ...
    print(f"成功注入类别名映射: {model.names}")

    result = model.val(
        data=r'C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\DeepLesion\Lesion.yaml',
        # cache=False,  # 添加这个参数
        split='test',  # split可以选择train、val、test 根据自己的数据集情况来选择.
        imgsz=640,
        batch=4,
        save_json=True,  # if you need to cal coco metrice
        project='logs/DeepLesion_t/DeepLesion_t_2.27',
        name='DeepLesion_t_2.27',
        # plots=False,  # ← 强烈推荐
        )

    # --- 新增：为展示效果生成独立预测图 ---
    print("正在生成独立可视化预测图...")
    model.predict(
        source=r'C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\DeepLesion\images\test',
        # 指向你的测试集图片文件夹
        imgsz=640,
        conf=0.25,  # 置信度阈值，可调
        save=True,  # ← 关键：保存带标签的图片
        project='logs/DeepLesion_t/DeepLesion_t_2.27',  # 保存到和 val 相同的目录下
        name='predict_visual',
        line_width=2,  # 框的粗细
        show_labels=True,
        show_conf=True
    )

    if model.task == 'detect':  # 仅目标检测任务适用
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图",
                                        "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75",
                                           "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row([
                cls_name,
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.box.ap[idx]:.4f}"
            ])
        model_metrice_table.add_row([
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))

        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)