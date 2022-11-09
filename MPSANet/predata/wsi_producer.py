import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
import torchvision.models as models

np.random.seed(0)


class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, mask_path, image_size=768, patch_size=256,
                 crop_size=224, normalize=True, flip='NONE', rotate='NONE'):
        """
        Initialize the predata producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path = wsi_path ##全切片的路径
        self._mask_path = mask_path ###提取的组织区域的mask
        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._preprocess()

    def _preprocess(self):
        self._mask = np.load(self._mask_path)  ##加载mask文件
        self._slide = openslide.OpenSlide(self._wsi_path) ##打开wsi文件

        X_slide, Y_slide = self._slide.level_dimensions[0] ##找到最大维度的值
        X_mask, Y_mask = self._mask.shape ##获得mask文件的形状

        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self._resolution = X_slide * 1.0 / X_mask ##获得leavl0下和mask的leval下的比率，缩小了几倍，缩小的倍数是2的几倍
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self._resolution))

        # all the idces for tissue region from the tissue mask
        self._X_idcs, self._Y_idcs = np.where(self._mask)##返回的是，所有mask中不为0的部分，也就是切的patch都是去除背景的部分

        self._idcs_num = len(self._X_idcs)
        print('self._X_idcs, self._Y_idcs',len(self._X_idcs), len(self._Y_idcs))
        print('输出self._mask',self._mask.shape)
        print('输出self._mask的值', self._mask)

        if self._image_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._image_size, self._patch_size))
        self._patch_per_side = self._image_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

    def __len__(self):
        return self._idcs_num  ##这个应该是要获得的像素点的个数

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)

        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)

        ##--------------------------------------------这个就是切patch的过程，但是这里的level开始设置为了0，应该是要和mask保持一致吧

        ##这个含义应该是，基本上mask内部的所有的像素点，都要取出来一个patch文件，然后这样每一个patch都会得到一个概率值，之后再把这个概率值给像素点，保存成图片
        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')

        #---------------------------------------------------------------------------------------------------------------

        ##实际上都没有进行这些操作，都为0了
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)

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

                idx += 1

        return (img_flat, x_mask, y_mask)
