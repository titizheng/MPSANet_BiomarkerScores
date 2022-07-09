
from PIL import Image
import numpy as  np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000
import os

import cv2 as cv
def blend_two_images():
    retu_path=r'D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\test_mask\retu_tupian'
    yuantu_path=r'D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\test_mask\yuantu_tupian'
    name='66650(CD30).png'
    i=os.path.join(yuantu_path,name)
    j=os.path.join(retu_path,name)
    alpha=0.9
    img1 = cv.imread(i)
    img2 = cv.imread(j)
    img_add = cv.addWeighted(img1, alpha,img2, beta=0.4, gamma=0)


    cv.imwrite(r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\test_mask\hebingtu/{}.png".format(name),img_add)
  # return
blend_two_images()





# img = Image.blend(Image.open(i).convert('RGBA'), Image.open(j).convert('RGBA'), alpha=0.6)