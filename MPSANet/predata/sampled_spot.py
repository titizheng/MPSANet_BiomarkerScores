import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")

parser.add_argument("--patch_number", default=150, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=0, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")



class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path ##
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)

        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)



        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points


def run(args, mask_path, txt_path):
    sampled_points = patch_point_in_mask_gen(mask_path, args.patch_number).get_patch_point() ##args.patch_number
    sampled_points = (sampled_points * 2 ** args.level).astype(np.int32) # make sure the factor

    mask_name = os.path.split(mask_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))

    with open(txt_path, "a") as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main(mask_path, txt_path):
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args, mask_path, txt_path)


if __name__ == "__main__":

    mask_path_root =r" "
    txt_path_root = r" "


    mask_name='62832(CD30).npy'
    mask_path = os.path.join(mask_path_root, mask_name)
    print("1.mask path{}".format(mask_path))
    (file_name, extension) = os.path.splitext(mask_name)
    txt_path = os.path.join(txt_path_root, file_name + ".txt")
    main(mask_path, txt_path)

