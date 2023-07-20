import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tqdm import tqdm

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


class EdgeDetection():

    def __init__(self, images_path):

        self.images_path = images_path


    def read_image(self):

        '''To read an image from a given path'''

        for image_file in os.listdir(self.images_path):
            image_path = os.path.join(self.images_path, image_file)
            image = cv2.imread(image_path)
            yield image

    def convert_to_gray(self, image):

        '''To convert an image to grayscale'''

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_laplacian_filter(self, image,kernel_sizes):

        '''To apply laplacian filter to an image with multiple kernel sizes'''

        results = []  # List to store the filtered images

        for ksize in kernel_sizes:
            laplacian_img = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
            laplacian_img_ = np.uint8(np.absolute(laplacian_img))
            results.append(laplacian_img_)

        return results

    def apply_custom_laplacian_filter(self, image):

        '''To apply custom laplacian filter to an image'''

        # create a kernel
        kernel = np.array([[0, -1, 0], 
                           [-1, 4, -1], 
                           [0, -1, 0]])
        
        # apply the kernel to the image
        return cv2.filter2D(image, -1, kernel) # -1 is the depth of the destination image

        

    def apply_filter_on_images(self):

        '''To apply filters to the images'''

        image_idx = 1 # Counter variable for image index

        # read the image
        images = list(self.read_image())

        # define the kernel sizes
        kernel_sizes = [1, 3, 5]

        # iterate over the images
        for i, img in tqdm(enumerate(images)):

            # convert the image to grayscale
            gray_img = self.convert_to_gray(img)

            # apply laplacian filter to the image with multiple kernel sizes
            laplacian_imgs = self.apply_laplacian_filter(gray_img, kernel_sizes)

            # apply custom laplacian filter to the image
            custom_laplacian_img = self.apply_custom_laplacian_filter(gray_img)

            # calculate the number of subplots
            num_subplots = len(laplacian_imgs) + 3 # 3 (original, grayscale, custom laplacian) + number of laplacian images = 6

            # calculate the figure size based on the number of subplots
            figsize = (3 * num_subplots, 5)  # (width, height) # 3 * num_subplots because each subplot has width 3

            # create a figure and a set of subplots
            fig, axs = plt.subplots(nrows=1, ncols=num_subplots, figsize=figsize)

            # plot the original image
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image', fontsize=9)
            axs[0].axis('off')

            # plot the grayscale image
            axs[1].imshow(gray_img, cmap='gray')
            axs[1].set_title('Grayscale Image')
            axs[1].axis('off')

            # plot the images with laplacian filter applied
            for i, laplacian_img in enumerate(laplacian_imgs):
                axs[i+2].imshow(laplacian_img, cmap='gray')
                axs[i+2].set_title(f'Laplacian Filter, Kernel Size: {kernel_sizes[i]}', fontsize=9)
                axs[i+2].axis('off')

            # plot the image with custom laplacian filter applied
            axs[-1].imshow(custom_laplacian_img, cmap='gray') # -1 is the last index
            axs[-1].set_title('Custom Laplacian Filter', fontsize=9)
            axs[-1].axis('off')

            # create a directory to save the results
            os.makedirs('./edge_detection', exist_ok=True)

            # save the figure
            fig.savefig(f'./edge_detection/edge_detection_{image_idx}.png')
            image_idx += 1 # increment the image index



    


