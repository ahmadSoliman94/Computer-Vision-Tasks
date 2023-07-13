import reinhard as rh
import thresholding as th
import color_recognition as cr


def main():


    # =================== COLR RECOGNITION =================== #

    # IMAGES_FOLDER_PATH = './images'

    # # Create an instance of the Color_Recognition class
    # color_recognition = cr.Color_Recognition(IMAGES_FOLDER_PATH)

    # # Load the images
    # images = color_recognition.load_images()


    # # detect the images
    # color_recognition.process(images)

    # # Classify the colors
    # color_recognition.classify()


    

  # ===================  REINHARD COLOR TRANSFER  =================== #

    # IMAGES_FOLDER_PATH = './input'
    # TEMPLATES_FOLDER_PATH = './template'

    # # Create an instance of the Reinhard class
    # transfer = rh.Image_Transfer(IMAGES_FOLDER_PATH, TEMPLATES_FOLDER_PATH) 

    # # Transfer the images
    # transfer.transfer()

  
  # ===================  IMAGE THRESHOLDING =================== #

    IMAGES_FOLDER_PATH = './thresholding_input'

    # Create an instance of the Image_Thresholding class
    thresholding = th.Image_Thresholding(IMAGES_FOLDER_PATH)

    # apply thresholding
    thresholding.apply_on_muliple_images()




if __name__ == '__main__':

    main()

    
