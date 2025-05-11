# AI coding

In summary, its the experimental autoencoder NN for loseless image compression for educational and scientific purpose.

-   The basic idea the same as [png](https://en.wikipedia.org/wiki/PNG) exploites: pixel prediction filtering by known context and futher
    compression with DEFLATE algorithm
-   The difference in prediction filter based on autoencoder inpaint convolutional network.
-   Two modes are supported - latent space representation and inpaint prediction


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
for training use image_cnn.py script. It is tuned to load the latest available checkpoint from the weights folder
The training parameters is adjusted immediately in the script after the comment line
 "Specifying the dimensionality of the latent space and training parameters".

Image datasets are expectped in the "./images" "train", "test" and "validate" subfolder respectively

For compression\uncompression use compress.py
Synopsis: (-c|-u|-t) "input file|folder" "output file|folder"
-c compress creates compressed images in with .bin extension, also created .bmp and .png files for the reference
The input folder is looked up for .jpg images
-u uncompress the .bin files into png

## Running tests
Every module contains small test that is run by default

## Resources

-   
-   
