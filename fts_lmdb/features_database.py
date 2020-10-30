# Copyright (c) Facebook, Inc. and its affiliates.
import os
import torch
from multiprocessing.pool import ThreadPool
from fts_lmdb.image_database import ImageDatabase
from fts_lmdb.feature_readers import FeatureReader

class FeaturesDatabase(ImageDatabase):
    def __init__(
        self, path, annotation_db=None, feature_path=None, *args, **kwargs
    ):
        super().__init__(path, annotation_db, *args, **kwargs)
        self.feature_readers = []
        self.feature_dict = {}
        self.feature_key = feature_path
        self._fast_read = False

        path = path.split(",")

        for image_feature_dir in path:
            feature_reader = FeatureReader(
                base_path=image_feature_dir,
                depth_first=False,
                max_features=100
            )
            self.feature_readers.append(feature_reader)

        self.paths = path
        self.annotation_db = annotation_db
        self._should_return_info = True

    def _threaded_read(self):
        elements = [idx for idx in range(1, len(self.annotation_db))]
        pool = ThreadPool(processes=4)

        for i, _ in enumerate(pool.imap_unordered(self._fill_cache, elements)):
            if i % 100 == 0:
                pass

        pool.close()

    def _fill_cache(self, idx):
        feat_file = self.annotation_db[idx]["feature_path"]
        features, info = self._read_features_and_info(feat_file)
        self.feature_dict[feat_file] = (features, info)

    def _read_features_and_info(self, feat_file):
        features = []
        infos = []
        for feature_reader in self.feature_readers:
            feature, info = feature_reader.read(feat_file)
            # feature = torch.from_numpy(feature).share_memory_()

            features.append(feature)
            infos.append(info)

        if not self._should_return_info:
            infos = None
        return features, infos

    def _get_image_features_and_info(self, feat_file):
        assert isinstance(feat_file, str)
        image_feats, infos = self.feature_dict.get(feat_file, (None, None))

        if image_feats is None:
            image_feats, infos = self._read_features_and_info(feat_file)

        return image_feats, infos

    def __len__(self):
        self._check_annotation_db_present()
        return len(self.annotation_db)

    def __getitem__(self, idx):
        self._check_annotation_db_present()
        image_info = self.annotation_db[idx]
        return self.get(image_info)

    def get(self, item):
        feature_path = item.get(self.feature_key, None)

        if feature_path is None:
            feature_path = self._get_feature_path_based_on_image(item)

        return self.from_path(feature_path)

    def from_path(self, path):
        assert isinstance(path, str)

        if "genome" in path and path.endswith(".npy"):
            path = str(int(path.split("_")[-1].split(".")[0])) + ".npy"

        features, infos = self._get_image_features_and_info(path)

        item = {}
        for idx, image_feature in enumerate(features):
            item["image_feature_%s" % idx] = image_feature
            if infos is not None:
                # infos[idx].pop("cls_prob", None)
                item["image_info_%s" % idx] = infos[idx]

        return item

    def _get_feature_path_based_on_image(self, item):
        image_path = self._get_attrs(item)[0]
        feature_path = ".".join(image_path.split(".")[:-1]) + ".npy"
        return feature_path

if __name__ == "__main__":
    db = FeaturesDatabase(path="detectron.lmdb", annotation_db=None, feature_path="features")
    keys = iter(os.listdir("features"))
    for k in keys:
        f = f"{k}"
        item = db.from_path(f)
        print(item["image_feature_0"].shape)
        print(item["image_info_0"].keys())
        print(item["image_info_0"]["num_boxes"])
        print(item["image_info_0"]["image_height"])
        print(item["image_info_0"]["image_width"])
        print(item["image_info_0"]["objects"].shape)
        print(item["image_info_0"]["bbox"].shape)
        print(item["image_info_0"]["max_features"].item())
        print(torch.max(torch.FloatTensor(item["image_info_0"]["cls_prob"]), -1).indices.shape)
        print(torch.max(torch.FloatTensor(item["image_info_0"]["cls_prob"]), -1).values.shape)
        break
