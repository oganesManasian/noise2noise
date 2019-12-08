from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator


class TrainGenerator(Sequence):
    def __init__(self, image_dir, batch_size=4, image_size=512):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.src_image_paths = [p for p in sorted(Path(image_dir + "src").glob("**/*")) if p.suffix.lower() in image_suffixes]
        self.trg_image_paths = [p for p in sorted(Path(image_dir + "trg").glob("**/*")) if p.suffix.lower() in image_suffixes]
        self.image_num = len(self.src_image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size

        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # For images with one color channel
        # x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        # y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)

        last_img_used_ind = batch_size * idx

        for sample_ind in range(batch_size):
            image_ind = last_img_used_ind + sample_ind
            x[sample_ind] = cv2.imread(str(self.src_image_paths[image_ind]))
            y[sample_ind] = cv2.imread(str(self.trg_image_paths[image_ind]))
            # For images with one color channel
            # x[sample_ind] = np.expand_dims(cv2.imread(str(self.image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)
            # y[sample_ind] = np.expand_dims(cv2.imread(str(self.image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)

        return x, y


class TestGenerator(Sequence):
    def __init__(self, image_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        src_image_paths = [p for p in sorted(Path(image_dir + "src").glob("**/*")) if p.suffix.lower() in image_suffixes]
        trg_image_paths = [p for p in sorted(Path(image_dir + "trg").glob("**/*")) if p.suffix.lower() in image_suffixes]
        self.image_num = len(src_image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_ind in range(self.image_num):
            x = cv2.imread(str(src_image_paths[image_ind]))
            y = cv2.imread(str(trg_image_paths[image_ind]))
            # For images with one color channel
            # x = np.expand_dims(cv2.imread(str(image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)
            # y = np.expand_dims(cv2.imread(str(image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)

            # expand_dims for creating the 4th dimension
            self.data.append((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)))

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]

