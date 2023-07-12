import os
import cv2
import numpy as np
from tqdm import tqdm


class Image_Transfer:

    '''
      Pipeline for Image Transfer:
        1. calculate the mean and the deviation of the images.
        2. read random input images and random template images.
        3. convert images to LAB color space.
        4. get the channels of the images.
        5. calculate the mean and the deviation of the channels.
        6. convert the input images from LAB to BGR.
    '''

    def __init__(self, images_folder_path, templates_folder_path):
        self.images_folder_path = images_folder_path
        self.templates_folder_path = templates_folder_path

    def get_mean_std(self,img):

        ''' Calculate the mean and the standard deviation of the image. '''

        x_mean, x_std = cv2.meanStdDev(img)

        x_mean = np.hstack(np.around(x_mean, decimals=2)) # hstack: Stack arrays in sequence horizontally (column wise).
        x_std = np.hstack(np.around(x_std, decimals=2))

        return x_mean, x_std
    
    def load_images(self):

        ''' Load the images and convert them to LAB color space. '''

        for image in os.listdir(self.images_folder_path):
            image_path = os.path.join(self.images_folder_path, image)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            yield img  # yeild: returns a generator object

    def load_templates(self):

        ''' Load the templates and convert them to LAB color space. '''

        for template in os.listdir(self.templates_folder_path):
            template_path = os.path.join(self.templates_folder_path, template)
            temp = cv2.imread(template_path)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2LAB)
            yield temp 


    def transfer(self):
        ''' Transfer the images. '''
        # load the images
        images = list(self.load_images())
        
        # load the templates
        templates = list(self.load_templates())
        
        # create output directory if not exists
        os.makedirs("./output", exist_ok=True)

        # iterate over each template
        for i, template in tqdm(enumerate(templates)):

            # Calculate the mean and the standard deviation of the templates.
            templates_mean, templates_std = self.get_mean_std(template)

            # iterate over each image
            for j, input_image in tqdm(enumerate(images)):

                # Calculate the mean and the standard deviation of the images.
                images_mean, images_std = self.get_mean_std(input_image)

                # get the channels of the images.
                h, w, c = input_image.shape

                # iterate over each channel
                for row in range(h):
                    for col in range(w):
                        for channel in range(c):
                            # transfer the images
                            x = input_image[row, col, channel] 
                            x = ((x - images_mean[channel]) * (templates_std[channel] / images_std[channel])) + templates_mean[channel] # transfer the images
                            x = round(x) # round the values

                            # check if the values are in the range
                            x = 0 if x < 0 else x # if x < 0 then x = 0 else x = x
                            x = 255 if x > 255 else x # if x > 255 then x = 255 else x = x

                            # assign the new values to the images
                            input_image[row, col, channel] = x

                # convert the images from LAB to BGR
                input_image = cv2.cvtColor(input_image, cv2.COLOR_LAB2BGR)

                # save the images
                cv2.imwrite(f"./output/transfer_template{i+1}_input{j+1}.jpg", input_image)
        
        print("Images have been transferred successfully!")

