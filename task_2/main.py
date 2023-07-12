import color_recognition as cr
import reinhard as rh

def main():


    # =================== COLR RECOGNITION =================== #

    # IMAGES_FOLDER_PATH = './images'

    # # Create an instance of the Color_Recognition class
    # color_recognition = cr.Color_Recognition(IMAGES_FOLDER_PATH)

    # # Load the images
    # images = color_recognition.load_images()


    # # Segment the images
    # color_recognition.process(images)

    # # Classify the colors
    # color_recognition.classify()


    

  # ===================  REINHARD COLOR TRANSFER Reinhard =================== #

    IMAGES_FOLDER_PATH = './input'
    TEMPLATES_FOLDER_PATH = './template'

    # Create an instance of the Reinhard class
    transfer = rh.Image_Transfer(IMAGES_FOLDER_PATH, TEMPLATES_FOLDER_PATH) 

    # Transfer the images
    transfer.transfer()




if __name__ == '__main__':

    main()

    
