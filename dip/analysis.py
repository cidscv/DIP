import matplotlib.pyplot as plt
import os
import cv2

def generate_histogram_from_image(image, name=""):
    """
    Generate a histogram from an input image 
    using matplotlib for chart generation and cv2 to define histogram
    """

    # Generate our output path, may need to fix this to not be relative path. Can use os.path.abspath
    histogram_output = "../histograms"

    if os.path.exists('../histograms') == False:
        print("Creating histogram output path 'histograms'")
        os.makedirs('../histograms')

    # Set out plot figures
    plt.figure(figsize=(10, 6))
    
    # Check of image is colour or b/w
    if len(image.shape) == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black', label='Intensity')
    
    # Set name of plot to image filename if no name provided
    if name == "":
        name = image

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram - {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save to histogram folder
    plt.savefig(f"{histogram_output}/{name}", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()