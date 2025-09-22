import re
import os.path as osp

from .bases import BaseImageDataset


class Veri776(BaseImageDataset):
    """
    VeRi-776 vehicle re-identification dataset
    Reference:
      Liu et al. Large-scale vehicle re-identification in urban surveillance videos. ICME 2016.

    Directory structure (official):
      veri_776/
        image_train/
        image_test/
        image_query/
        name_train.txt
        name_test.txt
        name_query.txt
        ...

    Filenames follow the pattern:
      <pid>_c<camid>_XXXXXXXX_X.jpg
      e.g., 0002_c002_00030600_0.jpg -> pid=2, camid=2
    """

    dataset_dir = 'veri_776'

    def __init__(self, root='your_dataset_path', verbose=True, **kwargs):
        super(Veri776, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')

        self.list_train_path = osp.join(self.dataset_dir, 'name_train.txt')
        self.list_test_path = osp.join(self.dataset_dir, 'name_test.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'name_query.txt')

        self._check_before_run()

        train = self._process_txt(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_txt(self.query_dir, self.list_query_path, relabel=False)
        gallery = self._process_txt(self.gallery_dir, self.list_test_path, relabel=False)

        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.list_train_path):
            raise RuntimeError("'{}' is not available".format(self.list_train_path))
        if not osp.exists(self.list_query_path):
            raise RuntimeError("'{}' is not available".format(self.list_query_path))
        if not osp.exists(self.list_test_path):
            raise RuntimeError("'{}' is not available".format(self.list_test_path))

    def _process_txt(self, dir_path, txt_path, relabel=False):
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        pid_container = set()
        filename_pattern = re.compile(r'^(\d+)_c(\d{3})_')

        for name in lines:
            m = filename_pattern.match(name)
            if m is None:
                raise RuntimeError("Filename '{}' does not match VeRi format".format(name))
            pid = int(m.group(1))
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for name in lines:
            m = filename_pattern.match(name)
            pid = int(m.group(1))
            camid = int(m.group(2))
            assert 1 <= camid <= 999
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            img_path = osp.join(dir_path, name)
            dataset.append((img_path, pid, camid))

        return dataset