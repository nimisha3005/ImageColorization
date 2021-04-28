# Image Colorization with RGB plates

The goal of this project is to colorize images taken by Prokudin-Gorskii a 100 years ago. The images are provided in the form of digitized glass plates that requires application of image processing techniques to produce a colored image.

This project is taken from [Dr. Alexei Efros’](https://inst.eecs.berkeley.edu/~cs194-26/fa20/hw/proj1/) class at the University of California, Berkeley. The output of the code on different images can be viewed in Final_Images folder.

## Data

The Prokudin-Gorskii collection of images is in the public domain, and these images can be downloaded from the [Library of Congress](http://www.loc.gov/pictures/collection/prok/).

## Installation

Step 1: Clone github repository
```
git@github.com:nimisha3005/ImageColorization.git
```
Step 2: Change working directory
```
cd ImageColorization
```
Step 3: Install requirements for the project
```
pip install -r requirements.txt
```
Step 4: Create the directory of source images where RGB images are present

Step 5: Run file on terminal and parse locations of input and output images
```
python img_color.py -i INPUT_LOCATION -o OUTPUT_LOCATION
```

##References

- [Project Report (Berkeley)] (https://sauravmittal.github.io/computational-photography/colorizing-photos/)
- [Image Pyramid and Alignment] (https://github.com/jmecom/image-pyramid)
