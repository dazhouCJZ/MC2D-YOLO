import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets
import matplotlib
matplotlib.use('Agg')   # ← 关键修复

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\datasets\LUNA16\luna16_test.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default=r'C:\Users\chenjunzhou\Desktop\project\yolo_mamba-main\yolo_mamba-main\logs\LUNA16_t\LUNA16_1.8_t\LUNA16_1.8_t\predictions_fixed.json', help='data yaml path')

    return parser.parse_known_args()[0]

if __name__ == '__main__':
    plots = False,  # ← 强烈推荐
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    anno = COCO(anno_json)  # init annotations api

    # ================== 新增修复代码 ==================
    # 检查是否存在 'info' 键，如果没有则手动添加一个空的
    if 'info' not in anno.dataset:
        anno.dataset['info'] = {'description': 'my custom dataset'}
    # =================================================

    pred = anno.loadRes(pred_json)  # init predictions api

    # 后续代码保持不变
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # TIDE 部分也不受影响
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='result')
