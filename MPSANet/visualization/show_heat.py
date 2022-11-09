import matplotlib.pyplot as plt
import  numpy as np
import openslide
import pandas as  pd
import  os
import seaborn as sns
level=2 #

def breast_thumbnail(slide_path,slide_name):
    slide = openslide.OpenSlide(slide_path)
    print(slide.level_dimensions[level])

    slide_shape = slide.level_dimensions[level]
    slide_w = slide_shape[0]
    slide_h = slide_shape[1]
    slide_region = slide.read_region((0, 0), level, (slide_w - 1, slide_h - 1))
    slide_npy = np.array(slide_region)  ##

    thumbnail = slide.get_thumbnail((slide_w, slide_h))  #

    thumbnail_npy = np.array(thumbnail) #

    plt.figure(figsize=(slide_w / 100, slide_h / 100), dpi=100)
    plt.imshow(thumbnail_npy)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig("/home/omnisky/sata_16tb1/zttdata/TCGA/breastHer2/HER2_status/debug/4masknpy/level0/image/{}.png".format(slide_name))
    print("3.Successfully saved thumbnail named {}".format(slide_name))
    plt.show()
    return slide_w, slide_h

def get_prob_map(npy_path, slide_name, slide_w, slide_h):
    image = np.load(npy_path) #
    image = np.transpose(image, [1, 0])
    print(image.shape)
    #######----------------------------------

    # image=pd.DataFrame(image)
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.5,dark=0,light=1, as_cmap=True)
    ax=sns.heatmap(image,cmap=cmap,cbar=True,xticklabels=False,yticklabels=False)  #YlGnBu，，rainbow
    print("npyslide_w:", image.shape[0], "npyslide_h:", image.shape[1])
    # plt.figure(figsize=(image.shape[0] / 100, image.shape[1] / 100), dpi=100)
    ax.figure.set_size_inches(slide_w / 100,slide_h / 100)

    scatter_fig = ax.get_figure()


    scatter_fig.savefig(
        "/home/omnisky/sata_16tb1/zttdata/TCGA/breastHer2/HER2_status/debug/4masknpy/level0/image/{}.png".format(slide_name))

    print("4.Probability graph of successfully saving file named {}".format(slide_name))
    plt.show()


slide_path_root =r"/home/omnisky/sata_16tb1/zttdata/TCGA/breastHer2/HER2_status/debug/1SVS/Her2Neg"
npy_path_root = r"/home/omnisky/sata_16tb1/zttdata/TCGA/breastHer2/HER2_status/debug/4masknpy/level0/Her2Neg" ##


for slide in os.listdir(slide_path_root):
    slide_path = os.path.join(slide_path_root, slide)
    (slide_name, extension) = os.path.splitext(slide)
    npy_path = os.path.join(npy_path_root, slide_name + ".npy")
    print("1.Obtained slice named {} successfully".format(slide_name))
    print("2.slide_path:",  slide_path, "npy_path:", npy_path)
    slide_w, slide_h = breast_thumbnail(slide_path, slide_name)
    get_prob_map(npy_path, slide_name, slide_w, slide_h)





