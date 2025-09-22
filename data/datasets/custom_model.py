import os.path as osp
import re

from .bases import BaseImageDataset

class CustomModel(BaseImageDataset):
    """
    Training-only dataset that reads image paths from a text file and
    infers PID from the parent folder name (e.g., .../90005/90005d1.jpg -> pid=90005).
    If a second column exists in the line, it is used as PID instead.

    Expects:
      data/datasets/custom_train/train_list.txt
        - either:  /abs/or/rel/path/to/90005/90005d1.jpg
        - or:      /abs/or/rel/path/to/90005/90005d1.jpg 90005

    Notes:
      - Single-camera setup: camid is fixed to 0 (indexing from 0).
      - For validation compatibility, we set query=gallery=train.
        Metrics will be meaningless but training runs fine.
    """
    dataset_dir = 'custom_train'

    def __init__(self, root='data/datasets', list_name='train_list.txt', camid_value=0, verbose=True, **kwargs):
        super(CustomModel, self).__init__()
        self.dataset_root = osp.join(root, self.dataset_dir)
        self.list_path = osp.join(self.dataset_root, list_name)
        self.camid_value = int(camid_value)

        self._check_before_run()
        train = self._process_list(self.list_path, relabel=True)

        # Training-only: reuse train as val to keep pipeline happy
        query = train
        gallery = train

        if verbose:
            print("=> CustomModel (train-only) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_root):
            raise RuntimeError("'{}' is not available".format(self.dataset_root))
        if not osp.exists(self.list_path):
            raise RuntimeError("'{}' is not available".format(self.list_path))

    def _infer_pid_from_path(self, img_path):
        # Parent folder name is PID, e.g., .../90005/90005d1.jpg -> '90005'
        parent = osp.basename(osp.dirname(img_path))
        if not re.fullmatch(r'\\d{5,}', parent):
            raise RuntimeError("Parent folder '{}' is not a 5+ digit PID".format(parent))
        return int(parent)

    def _process_list(self, txt_path, relabel=False):
        with open(txt_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        entries = []
        pid_container = set()

        # First pass: collect PIDs
        for ln in lines:
            parts = ln.split()
            img_path = parts[0]
            if not osp.isabs(img_path):
                img_path = osp.join(self.dataset_root, img_path)
            if len(parts) >= 2:
                pid = int(parts[1])
            else:
                pid = self._infer_pid_from_path(img_path)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        # Second pass: build dataset tuples
        dataset = []
        for ln in lines:
            parts = ln.split()
            img_path = parts[0]
            if not osp.isabs(img_path):
                img_path = osp.join(self.dataset_root, img_path)
            if len(parts) >= 2:
                pid = int(parts[1])
            else:
                pid = self._infer_pid_from_path(img_path)

            camid = self.camid_value  # single camera
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset