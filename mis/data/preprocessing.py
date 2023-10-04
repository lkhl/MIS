import os.path as osp
import pickle
from math import ceil
from multiprocessing import Lock, Process, Queue, Semaphore
from threading import Thread

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from mis.model.feature_extractor import build_feature_extractor, build_transform
from mis.ops import bottom_up_merging


class Resize(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, img):
        w, h = img.size
        if w % self.stride != 0:
            w = w + self.stride - w % self.stride
        if h % self.stride != 0:
            h = h + self.stride - h % self.stride
        return img.resize((w, h))


def build_model(model_size, patch_size):
    assert model_size in ('small', 'base')
    transform = transforms.Compose([
        Resize(stride=patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    if model_size == 'small':
        model_name = f'dino_vits{patch_size}'
    else:
        model_name = f'dino_vits{patch_size}'

    model = torch.hub.load('facebookresearch/dino:main', model_name)
    model.eval()

    return model, transform


class Preprocessor(object):

    def __call__(self, files, out_dir, model_size='small', patch_size=8, **kwargs):
        model = build_feature_extractor(model_size, patch_size)
        model.eval()
        model.cuda()
        transform = build_transform(scale_factor=patch_size / 8.0, patch_size=patch_size)
        for img_path in tqdm(files, desc='Preprocessing'):
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).cuda()
            h, w = img.shape[-2:]
            feat_h, feat_w = h // patch_size, w // patch_size
            with torch.no_grad():
                feats = model(img)[0]
            tree = bottom_up_merging(feats.cpu().numpy(), feat_h, feat_w)
            data = dict(tree=tree, size=(feat_h, feat_w))
            with open(osp.join(out_dir, f'{osp.splitext(osp.basename(img_path))[0]}.pkl'),
                      'wb') as f:
                pickle.dump(data, f)


class ParallelPreprocessor(object):

    def __init__(self, max_queue_size=50):
        self._task_queue = Queue(max_queue_size)
        self._featurizing_counter = Semaphore(0)
        self._merging_counter = Semaphore(0)
        self._loading_lock = Lock()

    def __call__(self,
                 files,
                 out_dir,
                 model_size='small',
                 patch_size=8,
                 n_featurizing_workers=1,
                 n_merging_workers=1):
        processes = []

        files_per_worker = ceil(len(files) / n_featurizing_workers)
        for i in range(n_featurizing_workers):
            cur_files = files[i * files_per_worker:(i + 1) * files_per_worker]
            process = Process(
                target=self._featurizing_worker,
                args=(cur_files, model_size, patch_size, self._task_queue,
                      self._featurizing_counter, self._loading_lock, f'cuda:{i}'))
            process.daemon = True
            processes.append(process)

        for i in range(n_merging_workers):
            process = Process(
                target=self._merging_worker,
                args=(out_dir, self._task_queue, self._merging_counter))
            process.daemon = True
            processes.append(process)

        featurizing_counting_thread = Thread(
            target=self._counting_worker,
            args=('Featurizing', self._featurizing_counter, len(files)))
        merging_counting_thread = Thread(
            target=self._counting_worker, args=('Merging', self._merging_counter, len(files)))
        featurizing_counting_thread.setDaemon(True)
        merging_counting_thread.setDaemon(True)
        featurizing_counting_thread.start()
        merging_counting_thread.start()

        for process in processes:
            process.start()

        merging_counting_thread.join()

    @staticmethod
    def _counting_worker(desc, counter, total):
        for _ in tqdm(range(total), desc=desc):
            counter.acquire()

    @staticmethod
    def _featurizing_worker(files, model_size, patch_size, task_queue, counter, lock, device):
        lock.acquire()
        model = build_feature_extractor(model_size, patch_size)
        model.eval()
        model.to(device)
        lock.release()
        transform = build_transform(scale_factor=patch_size / 8.0, patch_size=patch_size)
        for img_path in files:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            h, w = img.shape[-2:]
            feat_h, feat_w = h // patch_size, w // patch_size
            with torch.no_grad():
                feats = model(img)[0]
            task_queue.put(
                dict(
                    name=osp.splitext(osp.basename(img_path))[0],
                    features=feats.cpu().numpy(),
                    size=(feat_h, feat_w)))
            counter.release()

    @staticmethod
    def _merging_worker(out_dir, task_queue, counter):
        while True:
            task = task_queue.get()
            tree = bottom_up_merging(task['features'], *task['size'])
            data = dict(tree=tree, size=task['size'])
            with open(osp.join(out_dir, f'{task["name"]}.pkl'), 'wb') as f:
                pickle.dump(data, f)
            counter.release()
