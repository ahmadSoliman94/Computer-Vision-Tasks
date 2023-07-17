import filters



def main():

    # Create an instance of the ImageFilter class
    image_filter = filters.ImageFilter('./images')


    # Apply filters
    image_filter.apply_filters()



if __name__ == '__main__':
    main()