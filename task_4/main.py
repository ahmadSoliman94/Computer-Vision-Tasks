import laplacian_filter as lf


def main():
    # create an object of the LaplacianFilter class
    lf_obj = lf.EdgeDetection(images_path='./images')

    # apply filters to the images
    lf_obj.apply_filter_on_images()




if __name__ == '__main__':
    main()