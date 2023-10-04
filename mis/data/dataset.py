import os.path as osp
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from mis.ops import top_down_sampling


def sample_mask(data, img, threshold=0.05, decay=0.9):
    tree, (h, w) = data['tree'], data['size']
    size = h * w

    for _ in range(100):
        seed = np.random.randint(2**31)
        descendent_indices = top_down_sampling(tree, decay, seed)
        if len(descendent_indices) / size > threshold:
            break

    mask = np.zeros(size)
    mask[descendent_indices] = 1
    mask = mask.reshape(h, w)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask.astype(np.uint8)


class SBDFewDataset(ISDataset):

    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, ratio=0.05, **kwargs):
        super(SBDFewDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._insts_path = self.dataset_path / 'inst'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

        n_samples = int(len(self.dataset_samples) * ratio)
        self.dataset_samples = self.dataset_samples[:n_samples]

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, objects_ids=instances_ids, sample_id=index)

    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask


class SBDUnsupervisedDataset(ISDataset):

    def __init__(self, dataset_path, proposal_path, split='train', decay=0.9, **kwargs):
        super(SBDUnsupervisedDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._proposal_path = Path(proposal_path)
        self.decay = decay

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(osp.join(self._proposal_path, f'{osp.splitext(image_name)[0]}.pkl'), 'rb') as f:
            tree_data = pkl.load(f)

        instances_mask = sample_mask(tree_data, image.copy(), decay=self.decay)
        instances_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, objects_ids=instances_ids, sample_id=index)
