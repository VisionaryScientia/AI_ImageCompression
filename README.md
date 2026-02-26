Title of Paper: Image compression approach on autoencoder architecture

Link: http://www.ipol.im/ 
Version: Preprint 1.0
Date: February 26, 2026 

Authors: Oleksandr Kis (o.kis@duikt.edu.ua), Gennadiy Kis

In brief, it's the experimental autoencoder NN for loseless image compression for educational and scientific purpose.

-   The basic idea the same as [png](https://en.wikipedia.org/wiki/PNG) exploites: pixel filtering by known context and futher
    compression with DEFLATE algorithm
-   The prediction filter is based on autoencoder inpaint convolutional network
-   Two modes are supported - latent space representation and inpaint prediction


## Contents & Organization
Root Directory: Core execution scripts (compress.py, image_cnn.py).
images/: Dataset of gray-level test images (512x512).
weights_.../: Pre-trained NN weights (4K params) for different mask sizes.
requirements.txt: Python dependencies.
LICENSE: GNU license text.

## Installation
The following prerequisites required:
Python 3.6.6
with following packages:
zlib
numpy
Tensorflow:  2.6.2
Keras:  2.6.0

```sh
git clone https://github.com/VisionaryScientia/AI_ImageCompression
cd ./AI_ImageCompression
```

## Usage
For operating the tranied weights is required. Pretrained weights are available in the repository.
for training use image_cnn.py script. It is tuned to load the latest available checkpoint from the corresponding weights folder
The training parameters is adjusted immediately in the script image_cnn.py class settings specifies the dimensionality of the latent space and training parameters:

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

For compression\uncompression use compress.py
Synopsis: (-c|-u|-t) "input file|folder" "output file|folder"
-c compress creates compressed images in with .bin extension, also created .bmp and .png files for the reference
The input folder is looked up for .jpg images
-u uncompress the .bin files into png

## Running tests
Every module contains small test that is run by default

## LICENSE  
GNU AGPL
Copyright (c) 2026 GENNADIY KIS, OLEKSANDR KIS

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
