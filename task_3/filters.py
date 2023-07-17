import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import disk
from skimage.filters import gaussian, median
from skimage.restoration import denoise_bilateral


# ignore warnings
import warnings
warnings.filterwarnings('ignore')

class ImageFilter:
    def __init__(self, images_path):
        self.images_path = images_path
        self.filters = {
            'Blur': {'func': cv2.blur, 'args': {'ksize': (5, 5)}},
            'Gaussian': {'func': gaussian, 'args': {'sigma': 1}},
            'Median': {'func': median, 'args': {}},
            'Bilateral': {'func': denoise_bilateral, 'args': {'sigma_color': 0.05, 'sigma_spatial': 15,'channel_axis':-1}}
        }

    def load_images(self):

        '''To load images from a folder'''

        
        
        for image_file in os.listdir(self.images_path):
            image_path = os.path.join(self.images_path, image_file)
            image = cv2.imread(image_path)
            yield image

    def apply_filters(self):

        '''To apply filters to the images'''

        image_index = 1  # Counter variable for image index

        
        for image in tqdm(self.load_images()):
            # normalize image for skimage filters
            norm_image = image / 255.

            # Create a figure and a set of subplots.
            fig, axs = plt.subplots(nrows=len(self.filters) + 1, ncols=1, figsize=(10, 10 * (len(self.filters) + 1)))

            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            # Iterate over the filters
            for i, (filter_name, filter_) in tqdm(enumerate(self.filters.items())):
                func = filter_['func']
                args = filter_['args']

                # Apply the filter
                processed_image = func(norm_image, **args) if func in [median, denoise_bilateral] else func(image, **args)

                # Convert the processed image to RGB if it's grayscale
                if len(processed_image.shape) == 2:
                    processed_image = cv2.cvtColor((processed_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

                # Plot the processed image
                axs[i + 1].imshow(processed_image)
                axs[i + 1].set_title(filter_name + ' Filter')
                axs[i + 1].axis('off')

            
            plt.tight_layout()
            plt.show()

            # create a folder to save the filtered images
            os.makedirs('results', exist_ok=True)

            # Save the figure
            fig.savefig(f'./results/filtered_image_{image_index}.png')
            image_index += 1  # Increment the image index

