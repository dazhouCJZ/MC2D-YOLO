import json
import os


def convert_prediction_json(pred_file, anno_file, output_file=None):
    """
    将YOLOv8生成的prediction.json转换为COCO标准格式

    Args:
        pred_file: YOLOv8生成的prediction.json路径
        anno_file: COCO标注文件路径（instances_val.json）
        output_file: 输出文件路径，如果为空则自动生成
    """

    # 确保输出文件是.json文件
    if output_file:
        # 如果output_file是目录，添加文件名
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, "predictions_fixed.json")
        # 确保扩展名是.json
        elif not output_file.endswith('.json'):
            output_file = output_file + '.json'
    else:
        # 默认在pred_file同目录下创建_fixed版本
        dir_name = os.path.dirname(pred_file)
        base_name = os.path.basename(pred_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(dir_name, f"{name_without_ext}_fixed.json")

    print(f"预测文件: {pred_file}")
    print(f"标注文件: {anno_file}")
    print(f"输出文件: {output_file}")

    # 1. 加载COCO标注文件
    try:
        with open(anno_file, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)
    except Exception as e:
        print(f"加载标注文件失败: {e}")
        return None

    # 2. 创建文件名到图片ID的映射
    filename_to_id = {}
    for img in anno_data['images']:
        # 获取纯文件名（去掉路径和扩展名）
        filename = os.path.basename(img['file_name'])
        filename_no_ext = os.path.splitext(filename)[0]
        # 也保存带扩展名的版本
        filename_to_id[filename_no_ext] = img['id']
        filename_to_id[filename] = img['id']

    print(f"标注文件包含 {len(anno_data['images'])} 张图片，建立了 {len(filename_to_id)} 个映射")

    # 3. 加载预测文件
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"加载预测文件失败: {e}")
        return None

    print(f"预测文件包含 {len(pred_data)} 个预测")

    # 4. 转换预测文件中的image_id
    converted_count = 0
    failed_count = 0
    failed_examples = []

    for i, pred in enumerate(pred_data):
        original_id = str(pred['image_id'])
        matched = False

        # 尝试不同的匹配方式
        match_attempts = [
            original_id,  # 原始ID
            os.path.basename(original_id),  # 去掉路径
            os.path.splitext(os.path.basename(original_id))[0],  # 去掉路径和扩展名
        ]

        for attempt in match_attempts:
            if attempt in filename_to_id:
                pred['image_id'] = filename_to_id[attempt]
                converted_count += 1
                matched = True
                break

        if not matched:
            failed_count += 1
            if failed_count <= 10:  # 只记录前10个失败的例子
                failed_examples.append(original_id)

    # 5. 保存转换后的文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

        print(f"\n转换完成:")
        print(f"  成功转换: {converted_count}/{len(pred_data)}")
        print(f"  失败: {failed_count}/{len(pred_data)}")

        if failed_examples:
            print(f"\n前{len(failed_examples)}个失败的匹配:")
            for example in failed_examples:
                print(f"  - {example}")

            # 尝试猜测可能的原因
            print(f"\n可能的解决方案:")
            print("1. 检查预测文件中的image_id格式")
            print("2. 检查标注文件中的file_name格式")
            print("3. 可能需要自定义匹配逻辑")

        print(f"\n输出文件已保存: {output_file}")
        return output_file

    except Exception as e:
        print(f"保存文件失败: {e}")
        return None


if __name__ == "__main__":
    # 使用正确的路径 - 修改这里！
    pred_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16_t\LUNA16_1.8_t\LUNA16_1.8_t2\predictions.json"
    anno_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\LUNA16\luna16_test.json"

    # 输出到明确的位置
    output_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16_t\LUNA16_1.8_t\LUNA16_1.8_t2"

    # 或者让脚本自动决定输出位置
    # output_file = None  # 这将创建 predictions_fixed.json 在预测文件同目录

    result = convert_prediction_json(pred_file, anno_file, output_file)

    if result:
        print(f"\n转换成功！固定文件路径: {result}")
    else:
        print("\n转换失败！")