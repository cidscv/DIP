# Digital Image Processing - DIP

Two image processing pipelines designed for low-light images which typically have poor contrast, low brightness, noisy backgrounds, and motion blur due to longer exposure needed. Two pipelines are available for use; a baseline pipeline which peforms basic filters and enhancements for low-light scenes, and a customizeable pipeline that allows the user to select which filters and enhancement they would like to use so that they can enhance the image based on the specific issues present.

Some images may have lots of noise which require blur techniques to smooth over, other images may have inbalanced contrast levels that require adjustments and CLAHE (Contrast Limited Adaptive Histogram Equalization), and others may need light touchups to sharpen edges and reduce motion blur.

Program can be run following the instructions below. Choose from either pipeline A (baseline) or pipeline B (customized) when running the program based on desired output.

This program was built as part of Assignment 1 for Toronto Metropolitan University's CP8325 - Digital Image Processing MSc Computer Sciene class, taken in Fall 2025.

## DIP

### Baseline Processing

- Adjust black/white point levels, midtone adjustment
- Gamma transformation
- Apply Median filter for light blur
- Unsharp mask

All values will be pre-defined and consistent for every image.

### Custom Processing

Full choice of following filtering and enhancement techniques:

1. Adjust Levels and Curves,
2. Apply Gamma Transform,
3. Apply Gaussian Filter,
4. Apply Median Filter,
5. Apply CLAHE,
6. High Boost Filter,
7. Unsharp Mask,
8. White Colour Balance Correction

### Analysis & Comparisions

- Show histogram of image (histograms)
- Generate EXIF Data for each image (<output_dir>/exif_data)
- Detect clipping in histogram (analysis_output/)

## To Run

1. Create virtual python environment

`python -m venv .venv`

2. Install dependancies

`pip install -r requirements.txt`

3. Run program

```
usage: python main.py [-h] --image_dir imagedir --out output_dir --pipeline pipeline

Digital Image Processing (DIP) pipelines - processes images using different pipeline methods

options:
  -h, --help            show this help message and exit
  --image_dir imagedir  Directory containing input images to process
  --out output_dir      Directory where processed images will be saved
  --pipeline pipeline   Pipeline to use for processing (choices: A, B)
```

## Developer Notes

All filters and enhancement technique functions are in the filters.py file. The baseline_proc.py is responsible for the baseline processing pipeline A and runs automatically when chosen by the user. The custom_proc.py is where the custom pipeline B is defined and accepts user input based on the desired filter/enhancement. Pipeline B runs one image at a time and will only move on to the next image in the specified input directory once the user has selected 9: Exit. This step applies the filter and generates the output image in <output_dir>. There are some util functions for generating the exif data and clipping detection as well as the histogram generation function in analysis.py.

Program was written on Arch Linux OS but should work with other operating systems. Check package versions are available for current system if you run into trouble and ensure pip is updated to the latest version in your python virtual environment.

_Generative AI Disclaimer_ - Some code was generated using Claude AI based on my input. This was used to speed up the development process for functions which would have been written similary by me as these functions won't have any variance due to established, commonly accepted, methods for implementation. The design of this program was 100% originally created by me without the help of generative AI.

## Credits

Made by Owen Reid (owen.reid@torontomu.ca) | 2025
