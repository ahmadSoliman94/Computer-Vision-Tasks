import os
import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ignore warnings
import warnings
warnings.filterwarnings('ignore')



class Color_Recognition():

    ''' 
    Pipline:
    1. Load input images: Read the input images from the folder
    2. Convert to HSV: Convert the images from RGB to HSV
    3. Segment the images: Segment the images based on the k-means clustering algorithm.
    4. annotate output images: Visualize the results by drawing bounding boxes and write the color names on the images.
    5. extract feature data: Extract the feature data from the segmented images.
    6. classify the colors: Classify the colors based on the feature data.
    7. evaluate the model: Evaluate the model using the test data.
    '''

    def __init__(self, folder_path):

        '''Initialize the folder path'''

        self.folder_path = folder_path



    def load_images(self):

        '''1. Load input images: Read the input images from the folder'''

        images = []

        # Iterate through the images in the folder
        for filename in os.listdir(self.folder_path):

            # check the file extension
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.webp'):

                # Read the image
                img = cv2.imread(os.path.join(self.folder_path, filename))
                
                # Check if the image is not None
                if img is not None:

                    # Append the image to the list
                    images.append(img)
        
        print('Number of images loaded: ', len(images))
        
        return images
    
    def process(self, images):

        '''
        2. Convert to HSV: Convert the images from RGB to HSV
        3. Segment the images
        4. annotate output images: Visualize the results by drawing bounding boxes and write the color names on the images.
        5. extract feature data: Extract the feature data from the segmented images.
        '''
        segmented_images = []

        # Store the feature data
        feature_data = []


        # Dictionary of color ranges
        color_ranges = {'Red': [(0, 50, 50), (10, 255, 255)], 
                        'Green': [(50, 50, 50), (90, 255, 255)], 
                        'Blue': [(90, 50, 50), (130, 255, 255)],
                        'Yellow': [(20, 50, 50), (40, 255, 255)],}

        # Iterate through the images
        for image in images:

            # Convert the image from RGB to HSV
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for each color and combine them into a single mask,
            mask_total = np.zeros(image.shape[:2], dtype="uint8")

            # Create a kernel for morphological operations
            kernel = np.ones((5, 5), np.uint8)

            # Iterate through the color ranges
            for color, (lower, upper) in color_ranges.items():

                # Create a mask for each color
                mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))

                # Perform morphological operations to smooth the mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # Add the mask to the total mask
                mask_total += mask

            # Segment the image using the total mask
            segmented = cv2.bitwise_and(image, image, mask=mask_total)

            # Append the segmented image to the list
            segmented_images.append(segmented)

            # Find the contours in the mask
            contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Iterate through the contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Dictionary to store the color count
                color_count = {color: 0 for color in color_ranges.keys()} # 0 for each color

                # Iterate through the color ranges
                for color, (lower, upper) in color_ranges.items():

                    # Create a mask for each color
                    mask = cv2.inRange(image_hsv[y:y+h, x:x+w], np.array(lower), np.array(upper))

                    # Count the number of non-zero pixels within the bounding box
                    color_count[color] += cv2.countNonZero(mask)

                # Determine the color with the maximum count
                max_color = max(color_count, key=color_count.get)

                # Append the bounding box coordinates, color name, and contour area to the feature data
                feature_data.append([x, y, w, h, cv2.contourArea(contour),max_color])
                
                # Write the color name on the image
                cv2.putText(image, max_color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            # Create a DataFrame from the feature data
            df = pd.DataFrame(feature_data, columns=['x', 'y', 'width', 'height', 'area','color'])

            # Save the DataFrame as a CSV file
            df.to_csv('./feature_data.csv', index=False)

            # check if the folder exists
            if not os.path.exists('./predected_images'):
                os.makedirs('./predected_images')
            else:
                cv2.imwrite('./predected_images/image_' + str(len(segmented_images)) + '.jpg', image)


        return segmented_images
    
    def classify(self):

        ''' 
        6. classify the colors: Classify the colors based on the feature data.
        7. evaluate the model: Evaluate the model using the test data.
        '''

        # Read the feature data
        df = pd.read_csv('./feature_data.csv')

        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[['x', 'y', 'width', 'height', 'area']], df['color'], test_size=0.2, random_state=42)

        # Create a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the classifier
        knn.fit(X_train, y_train)

        print('-'*20)

        # Evaluate the classifier
        print('Accuracy: ', knn.score(X_test, y_test))

        print('-'*20)

        # pirnt classification report
        print(classification_report(y_test, knn.predict(X_test)))

        print('-'*20)

        # plot confusion matrix
        cm = confusion_matrix(y_test, knn.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=knn.classes_, yticklabels=knn.classes_)

        # save the plot
        plt.savefig('./confusion_matrix.png')

        # Predict the color of a new data point    
        print('Predicted color of a new data point: ', knn.predict([[0, 0, 100, 100, 1000]]))

        



            
    


    
 


    




    