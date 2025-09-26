from dip import analysis as aly
import cv2

if __name__ == '__main__':
    # Load the image first
    image = cv2.imread('input_images/testimage.jpg')
    
    if image is not None:
        aly.generate_histogram_from_image(image, name='test')
    else:
        print("Error: Could not load image")