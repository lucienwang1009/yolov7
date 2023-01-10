import torch
import cv2
import numpy as np
import random
import logging

from utils.fewshot.gen_support_pool import gen_support_pool


logger = logging.getLogger(__name__)


class FewShotGenerator(object):
    def __init__(self, img_dir, img_size=320, ways=None, shots=1, classes=None):
        # NOTE: For Few-shot
        logger.info('Generate support dataset for few-shot learning')
        self.support_df = gen_support_pool(img_dir, img_size)
        logger.info('Done')
        self.categories = set(self.support_df.category_id)
        self.shots = shots
        self.ways = ways
        if classes is not None:
            self.categories = self.categories.intersection(classes)
        print(self.categories)

    def generate(self, labels=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if labels is not None:
            cls = torch.unique(torch.cat(
                [label[:, 1] for label in labels]
            )).type(torch.int).tolist()
        else:
            cls = []

        if self.ways is not None:
            random.shuffle(cls)
            complement_cats = list(self.categories.difference(cls))
            random.shuffle(complement_cats)
            cls += complement_cats[:max(self.ways - len(cls), 0)]
            # cls = cls[:self.ways]
        else:
            cls = list(self.categories)

        imgs = []
        labels = []
        for c in cls:
            samples = self.support_df[
                self.support_df.category_id == c
            ]
            shots = self.shots
            if len(samples) < shots:
                shots = len(samples)

            samples = samples.sample(n=shots, replace=False)
            for _, s in samples.iterrows():
                labels.append(torch.tensor([s.category_id, *s.support_box]))
                imgs.append(torch.from_numpy(
                    np.ascontiguousarray(cv2.imread(
                        s.file_path).transpose((2, 0, 1))[::-1])
                ))
        return imgs, labels
