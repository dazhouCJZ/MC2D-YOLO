import json
import os
import re


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

    # 2. 创建多种映射方式
    # 映射1: 图片ID的直接映射（字符串和整数格式）
    id_mapping = {}
    for img in anno_data['images']:
        img_id = img['id']
        # 字符串形式
        id_mapping[str(img_id)] = img_id
        # 整数形式
        id_mapping[img_id] = img_id

    # 映射2: 从文件名提取数字并映射
    filename_to_id = {}
    for img in anno_data['images']:
        filename = os.path.basename(img['file_name'])
        filename_no_ext = os.path.splitext(filename)[0]

        # 直接文件名映射
        filename_to_id[filename_no_ext] = img['id']
        filename_to_id[filename] = img['id']

        # 从文件名提取数字并映射
        numbers = re.findall(r'\d+', filename_no_ext)
        for num in numbers:
            num_int = int(num)
            filename_to_id[str(num_int)] = img['id']
            filename_to_id[num_int] = img['id']

            # 尝试去掉前导零
            stripped = num.lstrip('0')
            if stripped:
                stripped_int = int(stripped)
                filename_to_id[str(stripped_int)] = img['id']
                filename_to_id[stripped_int] = img['id']

    print(f"标注文件包含 {len(anno_data['images'])} 张图片")
    print(f"建立的映射总数: {len(id_mapping)} (ID映射) + {len(filename_to_id)} (文件名映射)")

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

    # 记录原始ID类型的分布
    id_types = {}

    for i, pred in enumerate(pred_data):
        original_id = pred['image_id']

        # 记录ID类型
        id_type = type(original_id).__name__
        id_types[id_type] = id_types.get(id_type, 0) + 1

        matched = False
        new_id = None

        # 尝试多种匹配方式，按优先级排序

        # 方法1: 直接ID映射
        if original_id in id_mapping:
            new_id = id_mapping[original_id]
            matched = True
        # 方法2: 字符串形式的ID映射
        elif str(original_id) in id_mapping:
            new_id = id_mapping[str(original_id)]
            matched = True
        # 方法3: 文件名映射
        elif original_id in filename_to_id:
            new_id = filename_to_id[original_id]
            matched = True
        # 方法4: 字符串形式的文件名映射
        elif str(original_id) in filename_to_id:
            new_id = filename_to_id[str(original_id)]
            matched = True
        # 方法5: 尝试从预测ID中提取数字
        else:
            str_id = str(original_id)
            # 提取数字
            numbers = re.findall(r'\d+', str_id)
            for num in numbers:
                num_int = int(num)
                # 尝试作为ID
                if num_int in id_mapping:
                    new_id = id_mapping[num_int]
                    matched = True
                    break
                # 尝试去掉前导零
                stripped = num.lstrip('0')
                if stripped:
                    stripped_int = int(stripped)
                    if stripped_int in id_mapping:
                        new_id = id_mapping[stripped_int]
                        matched = True
                        break

        if matched and new_id is not None:
            pred['image_id'] = new_id
            converted_count += 1
        else:
            failed_count += 1
            if len(failed_examples) < 10:
                failed_examples.append({
                    'index': i,
                    'original_id': original_id,
                    'type': type(original_id).__name__
                })

    # 5. 输出转换统计信息
    print(f"\n📊 转换统计:")
    print(f"  总预测数: {len(pred_data)}")
    print(f"  成功转换: {converted_count}")
    print(f"  失败: {failed_count}")

    if id_types:
        print(f"\n📊 原始ID类型分布:")
        for type_name, count in id_types.items():
            print(f"  {type_name}: {count} ({count / len(pred_data) * 100:.1f}%)")

    if failed_examples:
        print(f"\n❌ 前{len(failed_examples)}个失败的匹配:")
        for example in failed_examples:
            print(f"  索引{example['index']}: {example['original_id']} (类型: {example['type']})")

        # 分析可能的解决方案
        print(f"\n🔍 问题分析和解决方案:")

        # 检查是否所有预测都有相同的问题
        if failed_count == len(pred_data):
            print("1. 所有预测都失败了，可能是完全错误的映射关系")
            print("2. 建议检查预测生成时的image_id是如何设置的")
            print("3. 可能需要查看YOLO预测代码，了解image_id的来源")
        else:
            print("1. 部分预测失败，可能是混合了多种ID格式")
            print("2. 可能需要更复杂的匹配逻辑")
            print("3. 建议输出更多调试信息来了解失败原因")

        print("\n💡 建议的调试步骤:")
        print("1. 查看标注文件的前几个图片ID和文件名")
        print("2. 查看预测文件的前几个image_id")
        print("3. 检查是否有系统性的偏移或转换错误")

    # 6. 保存转换后的文件（即使有失败）
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 输出文件已保存: {output_file}")

        # 如果完全失败，创建空文件并提示
        if converted_count == 0:
            print("\n⚠️ 警告: 没有成功转换任何预测！")
            print("   输出的文件可能无法用于评估")
            print("   请检查映射逻辑和原始数据格式")

        return output_file

    except Exception as e:
        print(f"保存文件失败: {e}")
        return None


def analyze_files_before_conversion(pred_file, anno_file):
    """
    在转换前分析文件格式
    """
    print("🔍 转换前分析:")
    print("-" * 50)

    # 加载预测文件
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        print(f"预测文件: {pred_file}")
        print(f"预测数量: {len(pred_data)}")

        # 分析前10个预测
        print("\n前10个预测的image_id:")
        for i, pred in enumerate(pred_data[:10]):
            original_id = pred.get('image_id', 'MISSING')
            print(f"  {i}: {original_id} (类型: {type(original_id).__name__})")

        # 统计不同类型的ID
        id_types = {}
        for pred in pred_data[:100]:  # 只分析前100个以节省时间
            original_id = pred.get('image_id', 'MISSING')
            id_type = type(original_id).__name__
            id_types[id_type] = id_types.get(id_type, 0) + 1

        print(f"\nID类型统计 (前100个):")
        for type_name, count in id_types.items():
            print(f"  {type_name}: {count}")

    except Exception as e:
        print(f"分析预测文件失败: {e}")

    print("-" * 50)

    # 加载标注文件
    try:
        with open(anno_file, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)

        print(f"\n标注文件: {anno_file}")

        if 'images' in anno_data:
            print(f"图片数量: {len(anno_data['images'])}")

            # 显示前5个图片的信息
            print("\n前5个图片信息:")
            for i, img in enumerate(anno_data['images'][:5]):
                print(f"  {i}: ID={img.get('id')}, 文件名={img.get('file_name')}")

        if 'annotations' in anno_data:
            print(f"标注数量: {len(anno_data['annotations'])}")

    except Exception as e:
        print(f"分析标注文件失败: {e}")


if __name__ == "__main__":
    # 使用正确的路径
    pred_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16_t\LUNA16_1.8_t\LUNA16_1.8_t\predictions.json"
    anno_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\LUNA16\luna16_test.json"
    output_file = r"C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16_t\LUNA16_1.8_t\LUNA16_1.8_t"

    # 先分析文件
    print("=" * 60)
    print("文件格式分析")
    print("=" * 60)
    analyze_files_before_conversion(pred_file, anno_file)

    print("\n" + "=" * 60)
    print("开始转换")
    print("=" * 60)

    # 执行转换
    result = convert_prediction_json(pred_file, anno_file, output_file)

    if result:
        print(f"\n✅ 转换完成！固定文件路径: {result}")
    else:
        print("\n❌ 转换失败！")