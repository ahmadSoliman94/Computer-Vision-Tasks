import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize, thin


class MorphologicalTransformations:

    ''' 
    Pipeline for morphological transformations:
    1. Convert the image to binary image
    2. Convert Iimage ti boolean.
    3. Apply skeletonization: reduces binary objects to 1 pixel wide representations. 
    This can be useful for feature extraction, and/or representing an object’s topology.
    4. Apply Opening: erosion followed by dilation.  Opening can remove small bright spots (i.e. “salt”)
    5. Apply Closing: dilation followed by erosion. Closing can remove small dark spots (i.e. “pepper”)
    '''


    def __init__(self, image_folder_path):

        self.image_folder_path = image_folder_path


    def load_images(self):

        ''' Load the images. '''

        for image in os.listdir(self.image_folder_path):
            image_path = os.path.join(self.image_folder_path, image)
            img = cv2.imread(image_path)

            yield img


    def convert_to_binary(self, img):

        ''' Convert the image to binary image. '''

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)

        # Apply thresholding
        ret,bin_img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Convert to boolean - this is required by remove_small_objects

        return bin_img
    



    def apply_skeletonization(self, img_binary):

        ''' Apply skeletonization. '''

        skeleton = skeletonize(img_binary, method='lee')

        # Convert back to uint8
        skeleton = skeleton.astype(np.uint8)

        return skeleton
    

    def apply_closing(self, img_binary):

        ''' Apply Closing. '''

        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

        return closing
    
    def apply_opening(self, img_binary):

        ''' Apply Opening. '''

        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

        return opening
    






    def apply_opreation(self):

        ''' Apply the operations. '''

        # Load the images
        images = list(self.load_images())

        for idx,image in tqdm(enumerate(images)):

            # Convert to binary
            img_binary = self.convert_to_binary(image)

            # Apply skeletonization
            skeleton = self.apply_skeletonization(img_binary)

            # Apply closing
            closing = self.apply_closing(img_binary)

            # Apply opening
            opening = self.apply_opening(img_binary)

            # Apply Combination of Opening and Closing
            opening_closing = self.apply_closing(opening)

            # Apply combination of skeletonization and closing and opening
            selk_clos_op  = self.apply_skeletonization(opening_closing) 


            # create a folder to save the images
            if not os.path.exists('./morphological_output'):
                os.makedirs('./morphological_output')


            # Save the images
            cv2.imwrite(f'./morphological_output/{idx}_skeleton.png', skeleton)
            cv2.imwrite(f'./morphological_output/{idx}_closing.png', closing)
            cv2.imwrite(f'./morphological_output/{idx}_opening.png', opening)
            cv2.imwrite(f'./morphological_output/{idx}_opening_closing.png', opening_closing)
            cv2.imwrite(f'./morphological_output/{idx}_selk_clos_op.png', selk_clos_op)
            
