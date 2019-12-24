# coding: utf-8
import os
import glob

DATA_ROOT = '/Data1/DL-project/VoxNet.pytorch/data/ModelNet10'

CLASSES = {'bathtub', 'chair', 'dresser', 'night_stand', 'sofa', 'toilet', 'bed', 'desk', 'monitor', 'table'}

for c in CLASSES:
    for split in ['test', 'train']:
        for off in glob.glob(os.path.join(DATA_ROOT, c, split, '*.off')):
            # 判断是否存在
            binname = os.path.join(DATA_ROOT, c, split, os.path.basename(off).split('.')[0] + '.binvox')
            if os.path.exists(binname):
                print(binname, "exits, continue...")
                continue
            os.system(f'./binvox -d 32 -cb -pb {off}')
