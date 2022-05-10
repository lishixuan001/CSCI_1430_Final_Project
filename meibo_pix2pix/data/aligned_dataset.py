import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
from skimage import img_as_bool
import numpy as np
from pdb import set_trace as st
from matplotlib import pyplot as plt


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def generate_occlusion(self, output_size, min_rate, max_rate):
        oclu_min, oclu_max = int(round(output_size * min_rate)), int(round(output_size * max_rate))
        ocl_height, ocl_width = np.random.randint(low=oclu_min, high=oclu_max, size=2)
        ocl_point_lft = np.random.randint(low=0, high=int(output_size-ocl_width))
        ocl_point_up = np.random.randint(low=0, high=int(output_size-ocl_height))
        return ocl_height, ocl_width, ocl_point_lft, ocl_point_up
        
    def change_gray_scale(self, image):
        """
        Change image from [-1, 1] scale to [0, 1] scale
        """
        return (image - image.min()) / (image.max() - image.min())
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A) # [1, 256, 256]
        B = B_transform(B) # [1, 256, 256]
        
        num_occlusions = 2
        for j in range(num_occlusions):
            ocl_height, ocl_width, ocl_point_lft, ocl_point_up = self.generate_occlusion(output_size=A.shape[-1], 
                                                                                         min_rate=1/5, 
                                                                                         max_rate=1/3)
            A[:, ocl_point_lft:(ocl_point_lft+ocl_height), ocl_point_up:(ocl_point_up+ocl_width)] = -1.0
        
#         plt.imsave("/home/lishixuan001/ICSI/A.png", self.change_gray_scale(A.squeeze(0)), cmap='gray')
#         plt.imsave("/home/lishixuan001/ICSI/B.png", self.change_gray_scale(B.squeeze(0)), cmap='gray')
        
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
