import os
import json
import sys
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from utils.pcutils import normalize, resample_pcd, make_holes_pcd2, show_pcd

shapenet_part_dir = 'data/shapenetcore_part'
class ShapeNetDataset(Dataset):
    def __init__(self, root_dir=shapenet_part_dir, npoints=2048, class_choice=None,
                 split='train', hole_size=0.35, classification=False):
        self.npoints = npoints
        self.root = root_dir
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.labels = {}
        self.npoints = npoints
        self.normalize = True
        self.hole_size = hole_size
        self.classification = classification

        with open(self.catfile, 'r') as f:
            for i, line in enumerate(f):
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
                self.labels[ls[0]] = i

        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                sys.exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),
                                        self.cat[item], token))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3], self.labels[item]))

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        if self.normalize:
            point_set = normalize(point_set, unit_ball=True)

        label = fn[-1]
        if self.classification:
            return resample_pcd(point_set, self.npoints), np.array([label])
        else:
            partial, hole = make_holes_pcd2(point_set, hole_size=self.hole_size)
            partial = resample_pcd(partial, self.npoints)
            gt = resample_pcd(point_set, self.npoints)
            return partial, gt

    def __len__(self):
        return len(self.datapath)

class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file = shapenet_part_dir, npoints=2048, batch_size = 32,
                  hole_size = 0.35, classification=False):
        super().__init__()
        self.dataset_file = dataset_file
        self.npoints = npoints
        self.batch_size = batch_size
        self.hole_size = hole_size
        self.classification=classification

    def setup(self, stage=None):
        if stage =='fit' or stage is None:
            self.train_dataset = ShapeNetDataset(root_dir=self.dataset_file, npoints=self.npoints, split='train',
                                                 hole_size=self.hole_size, classification=self.classification)

            self.val_dataset = ShapeNetDataset(root_dir=self.dataset_file, npoints=self.npoints, split='val',
                                               hole_size=self.hole_size, classification=self.classification)

        if stage == 'test' or stage is None:
            self.test_dataset = ShapeNetDataset(root_dir=self.dataset_file, npoints=self.npoints, split='test',
                                                hole_size=self.hole_size, classification=self.classification)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, drop_last=True)

if __name__ == '__main__':
    shapenet_part_dir = 'shapenetcore_part'
    data = ShapeNetDataset(root_dir=shapenet_part_dir, npoints=2048,split='train')
    print(f'the number of point clouds: {len(data)} \nthe shape of point cloud is: ', *[tuple(pc.shape) for pc in data[0]])

