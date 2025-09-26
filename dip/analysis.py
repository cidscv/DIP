import matplotlib.pyplot as plt
import os
import cv2

def generate_histogram_from_image(image, name=""):
    """
    Generate a histogram from an input image 
    using matplotlib for chart generation and cv2 to define histogram
    """

    # Generate our output path using absolute path
    histogram_output = os.path.abspath("histograms")

    if not os.path.exists(histogram_output):
        print("Creating histogram output path 'histograms'")
        os.makedirs(histogram_output)

    # Set our plot figures
    plt.figure(figsize=(10, 6))
    
    # Check if image is colour or b/w
    if len(image.shape) == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black', label='Intensity')
    
    # Handle case where no name provided
    if name == "":
        name = "unnamed_image"

    plt.xlabel('Light Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram - {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save to histogram folder
    output_file = os.path.join(histogram_output, f"{name}_histogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {output_file}")
    return output_file