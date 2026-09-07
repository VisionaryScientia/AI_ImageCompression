Title of Paper: Image compression approach on autoencoder architecture

Date: February 26, 2026 

Authors: Oleksandr Kis (o.kis@duikt.edu.ua), Gennadiy Kis

In brief, it's the experimental autoencoder NN for loseless image compression for educational and scientific purpose.

-   The basic idea is similar to [png](https://en.wikipedia.org/wiki/PNG): pixel filtering by known context and futher
    compression with DEFLATE algorithm
-   The prediction filter is based on autoencoder convolutional network
-   Two modes are supported - latent space representation and inpaint prediction


## Contents & Organization
The Root Directory contains:
- \*.bat: execution scripts for Windows (setup.bat, train.bat, test.bat)
- \*.py: Python source core  (compress.py, image_cnn.py etc.).
- images: Dataset of training and test images.
- weights_...: Pre-trained NN weights for different mask sizes.
- requirements.txt: Python dependencies.
- LICENSE: GNU license text.

## Installation


The following prerequisites required (see requirements.txt):
Python 3.8
with following packages:
zlib
numpy
bitstring
Tensorflow:  2.6.2
Keras:  2.6.0
opencv (required for the result visualization)
matplotlib (required for the result visualization)

```sh
git clone https://github.com/VisionaryScientia/AI_ImageCompression
cd ./AI_ImageCompression
setup.bat
```
Setup batch script installs needed python packages and run test

## Usage
For the compression operating the CNN weights is required. Pretrained weights are available in the repository.
for training use image_cnn.py script. It is tuned to load the latest available checkpoint from the corresponding weights folder.
The weight folder should follow pattern 'weights_<block_size>_<inpaint_size>',
For example, weights_24_0 is used for autoencoder input 24 without inpainting (pure latent space encoding).
The training/inference parameters is adjusted immediately in the script image_cnn.py.
The class settings specifies the dimensionality of the latent space and training parameters:

```
class settings():
    batch_size = 64
    inpaint_size = 8 # 0 - for latent compression, 8 and 12 - supported inpanting mask size
    input_size = 24
    # max processing size, if an input image has bigger dims it is resized for faster processing
    max_dim = 640
    # grayscale images only
    channels = 1
    # use good block mask
    zero_diff = 0
    # loss can be 'binary_crossentropy', "mean_squared_error" or empty string for inference mode
    loss = 'mean_squared_error'

```

Image datasets are expectped in the "./images" "train", "test" and "validate" subfolder respectively

For training use
```
image_cnn.py [epoch number] [TensorBoard]
```
Or
```
train.bat
```

By default single training epoch passed. If added TensorBoard then corresponding callback is added.
For compression\decompression use compress.py
Synopsis: compress.py (-c|-d|-t|-v) "input file|folder" "output file|folder"
-c compress creates compressed images with .bin extension, also .png files are created for the reference,
 becase of the original image is cropped to be aligned with inpaint block size for simplicity.
The input folder is looked up for images (gif, jpg, png)
-d decompress the .bin files into png
-t generates test output: CNN blocks mask, residual image and residuals histogram
-v the same as test mode but the output is immediately visualized 

## Running tests
Every module contains small test that is run by default
Use setup.bat to install needed packages and run needed tests. If they are passed then the environment is configured correctly.
To run compression/decompression test:
```
test.bat
```
This script compressed a file to the 'compressed' folder then restore it back into 'decompressed' folder and show the result image.

## LICENSE  
Copyright (c) 2026 GENNADIY KIS, OLEKSANDR KIS
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, merge, publish, and distribute the Software solely for NON-COMMERCIAL, EDUCATIONAL, and SCIENTIFIC RESEARCH purposes, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. Any publication, paper, or scientific work that utilizes this Software must provide proper academic attribution referencing this repository.
3. For commercial licensing inquiries or permission to use the Software outside the scope of this license, please contact the authors.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
