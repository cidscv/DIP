# Digital Image Processing - DIP

An image processing pipeline that analyizes different techniques, comparing between baseline adjustments such as histogram equalization of intensity values to balance contrast, light blur techniques (Gaussian, Box, Median), logarithmic transformations to compress dynamic range. Additionally, a more configurable pipeline so that input images can have specific filters and transformations applied based on the issues present in the image. By using both pipelines, we can compare the output images from both to verify what the problem domain of each image is and why we decided to use specific techniques in order to imrove the image quality and show detail.

Finally, we will use resampling techniques when downsizing the image size so that we can output each image at different resolutions while maintaining detail and quality.

## DIP

### Baseline Processing

- Adjust black/white point levels, midtone adjustment
- Gamma transformation
- Apply Median filter for light blur
- Unsharp mask

All values will be pre-defined and consisten for every image.

### Custom Processing

- Contrast-limited adaptive histogram equalization
- Ability to choose between box, gaussian, or median blur techniques
- Colour balance

All values will be set by the user when input image is provided.

### Analysis & Comparisions

- Show histogram of image
- Show basic properties of image
- Detect clipping in histogram
- Apply resolution adjustments, with anti-aliasing techniques
- Ablation study
- Determine issues with image based on computed factors

## To Run

1. Create virtual python environment

`python -m venv .venv`

2. Install dependancies

`pip install -r requirements.txt`

3. Run program

`python main.py -i <input_image> -p <pipeline={baseline, custom}> [-c <custom_options>] -o <output_image>`

## Credits

Made by Owen Reid (owen.reid@torontomu.ca) for CP8325 - Digital Image Processing
