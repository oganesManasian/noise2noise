from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator


class TrainGenerator(Sequence):
    def __init__(self, image_dir, source_noise_model, target_noise_model, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // (2 * self.batch_size)

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size

        # x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # For images with one color channel
        x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)

        last_img_used_ind = 2 * batch_size * idx

        for sample_ind in range(batch_size):
            image_ind = last_img_used_ind + 2 * sample_ind
            # x[sample_ind] = cv2.imread(str(self.image_paths[image_ind]))
            # y[sample_ind] = cv2.imread(str(self.image_paths[image_ind + 1]))
            # For images with one color channel
            x[sample_ind] = np.expand_dims(cv2.imread(str(self.image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)
            y[sample_ind] = np.expand_dims(cv2.imread(str(self.image_paths[image_ind + 1]), cv2.IMREAD_GRAYSCALE), axis=2)

        return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir, val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        n_pairs = self.image_num // 2
        for pair_ind in range(n_pairs):
            image_ind = pair_ind * 2
            # x = cv2.imread(str(image_paths[image_ind]))
            # y = cv2.imread(str(image_paths[image_ind + 1]))
            # For images with one color channel
            x = np.expand_dims(cv2.imread(str(image_paths[image_ind]), cv2.IMREAD_GRAYSCALE), axis=2)
            y = np.expand_dims(cv2.imread(str(image_paths[image_ind + 1]), cv2.IMREAD_GRAYSCALE), axis=2)

            # expand_dims for creating the 4th dimension
            self.data.append((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)))

    def __len__(self):
        return self.image_num // 2

    def __getitem__(self, idx):
        return self.data[idx]

