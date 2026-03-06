import os
from pathlib import Path

# 配置路径
img_dir = Path('data/split_ss_dota/test/images')
ann_dir = Path('data/split_ss_dota/test/annfiles')

# 确保目录存在
ann_dir.mkdir(parents=True, exist_ok=True)

# 标准标签内容 (DOTA格式)
label_content = """
717.0 76.0 726.0 78.0 722.0 95.0 714.0 90.0 small-vehicle 0
"""

# 为每个图像生成标签文件
for img_file in img_dir.glob('*.png'):
    ann_file = ann_dir / f"{img_file.stem}.txt"
    
    with open(ann_file, 'w') as f:
        f.write(label_content)
    
    print(f"生成: {ann_file}")

print(f"\n成功为 {len(list(img_dir.glob('*.png')))} 张图像创建标签文件")