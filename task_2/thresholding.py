import os
import cv2
import numpy as np
from tqdm import tqdm



class Image_Thresholding():


    '''
    Pipline:

    1. Load the images.
    2. Convert the image to grayscale.
    3. Apply Gaussian Blur to reduce noise.
    4. Apply Contrast enhancement.
    5. Apply Global Thresholding.
    6. Apply Adaptive Thresholding.
    7. Apply Otsu Thresholding.
    8. Save the images.
    '''

    def __init__(self, images_folder_path):
        self.images_folder_path = images_folder_path


    def load_images(self):

        ''' Load the images. '''

        for image in os.listdir(self.images_folder_path):
            image_path = os.path.join(self.images_folder_path, image)
            img = cv2.imread(image_path)

            ''' 
            return: it immediately terminates the function and returns the value to the caller function.
            yield: it pauses the function saving all its states and later continues from there on successive calls.
            '''
            yield img

    def apply_contrast_enhancement(self, img):

        ''' Apply Contrast enhancement. '''

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split the image into its channels
        l, a, b = cv2.split(img)

        # Apply CLAHE to the luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)

        # Merge the channels back together
        img_clahe = cv2.merge((l_clahe, a, b))

        # Convert back to BGR
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        
        return img_clahe
    
    
    def apply_gaussian_blur(self, img):

        ''' Apply Gaussian Blur to reduce noise. '''

        img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    
    def convert_to_grayscale(self, img):

        ''' Convert the image to grayscale. '''

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def apply_gaussian_blur(self, img):

        ''' Apply Gaussian Blur to reduce noise. '''

        img = cv2.GaussianBlur(img, (5,5), 0)
        return img
    
    
    def apply_global_thresholding(self, img):

        ''' Apply Global Thresholding. '''

        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        return img
    
    def apply_adaptive_thresholding(self, img):

        ''' Apply Adaptive Thresholding. '''

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        return img
    

    def apply_otsu_thresholding(self, img):

        ''' Apply Otsu Thresholding. '''

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img
    

    def apply_on_muliple_images(self):

        ''' Apply the thresholding methods on multiple images. '''

        # load the images
        images = list(self.load_images())

        # iterate over the images
        for i, img in tqdm(enumerate(images)):

            # apply contrast enhancement
            enh_img = self.apply_contrast_enhancement(img)

            
            # apply gaussian blur
            img_blur = self.apply_gaussian_blur(enh_img)

            # convert the image to grayscale
            img_gray = self.convert_to_grayscale(img_blur)


            # apply global thresholding
            global_thresholding = self.apply_global_thresholding(img_gray)

            # apply adaptive thresholding
            adaptive_thresholding = self.apply_adaptive_thresholding(img_gray)

            # apply otsu thresholding
            otsu_thresholding = self.apply_otsu_thresholding(img_gray)

            # create thresholding_output directory if not exists
            os.makedirs("./thresholding_output", exist_ok=True)


            cv2.imwrite(f"./thresholding_output/global_thresholding_{i}.jpg", global_thresholding)
            cv2.imwrite(f"./thresholding_output/adaptive_thresholding_{i}.jpg", adaptive_thresholding)
            cv2.imwrite(f"./thresholding_output/otsu_thresholding_{i}.jpg", otsu_thresholding)

        print(f"Image saved successfully.")



        