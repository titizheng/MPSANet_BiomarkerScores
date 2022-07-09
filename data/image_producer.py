import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa

from data.annotation import Annotation  # noqa
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    数据生成器，生成一个正方形网格，例如3x3，补丁及其
来自预采样图像的相应标签。
    """
    def __init__(self, data_path, image_name_list, img_size, patch_size,json_path,
                 crop_size=224, normalize=True):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path ##patch path
        self.image_name_list=image_name_list
        self._json_path = json_path ##file for  data
        self._img_size = img_size   ##786
        self._patch_size = patch_size   ##256
        self._crop_size = crop_size     ##224
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        self._pids = list(map(lambda x: x.strip('.json'),
                              os.listdir(self._json_path)))


        self._annotations = {}
        for pid in self._pids:
            pid_json_path = os.path.join(self._json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self._annotations[pid] = anno

        self._coords = [] #


        f=self.image_name_list

        # for line in f:
        #     pid, x_center, y_center, idpng = line.strip('\n').split('-')[0:4]#
        #     x_center, y_center = int(x_center), int(y_center)
        #     self._coords.append((pid, x_center, y_center, idpng))  ##不旋转

        for line in f:
            xzt = line.strip('\n').split('-')
            pid = xzt[0]+'-'+xzt[1]+'-'+xzt[2] + '-'+xzt[3]+'-'+xzt[4]+'-'+xzt[5]
            x_center, y_center, idpng = xzt[-3],xzt[-2],xzt[-1]#分别对应着，名字，x，y，序号.png
            x_center, y_center = int(x_center), int(y_center)
            self._coords.append((pid, x_center, y_center, idpng))  ##不旋转


            self._coords.append((pid, x_center, y_center, idpng,0))
            self._coords.append((pid, x_center, y_center, idpng,1))
            self._coords.append((pid, x_center, y_center, idpng,2))
            self._coords.append((pid, x_center, y_center, idpng, 3))


        self._num_image = len(self._coords)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        # pid, x_center, y_center,idpng,ds = self._coords[idx]


        pid, x_center, y_center, idpng= self._coords[idx]

        x_top_left = int(x_center - self._img_size / 2)
        y_top_left = int(y_center - self._img_size / 2)

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # (x, y) is the center of each patch
                x = x_top_left + int((x_idx + 0.5) * self._patch_size)
                y = y_top_left + int((y_idx + 0.5) * self._patch_size)

                if self._annotations[pid].inside_polygons((x, y), True):
                    label = 1
                else:
                    label = 0

                # extracted images from WSI is transposed with respect to
                # the original WSI (x, y)
                label_grid[y_idx, x_idx] = label
        name1=pid+'-'+str(x_center)+'-'+str(y_center)+'-'+idpng

        img = Image.open(os.path.join(self._data_path, name1))

        img = self._color_jitter(img)


        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_grid = np.fliplr(label_grid)

        # use rotate

        num_rotate = np.random.randint(0, 4)

        # num_rotate = ds
        # num_rotate = 0  ##

        img = img.rotate(90 * num_rotate)
        label_grid = np.rot90(label_grid, num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1

        return (img_flat, label_flat)
