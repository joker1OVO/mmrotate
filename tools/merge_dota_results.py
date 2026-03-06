import argparse
import os
import sys

# 添加 DOTA_devkit 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools/DOTA_devkit'))
from ResultMerge import mergebypoly

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', help='裁剪图像目录', default='data/split_ss_dota/test')
    parser.add_argument('--result_dir', help='检测结果目录', default='runs/DOTA/test_results')
    parser.add_argument('--merge_dir', help='合并结果目录', default='runs/DOTA/merged_results')
    parser.add_argument('--iou_thresh', type=float, default=0.1, help='IOU 阈值')
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(args.merge_dir, exist_ok=True)
    
    # 调用合并函数
    mergebypoly(args.split_dir, args.result_dir, args.merge_dir, args.iou_thresh)

if __name__ == '__main__':
    main()