import sys
import re
import os
from image_cnn import make_model
from evaluate_image import evaluate_image,\
  restore_image, prepare_image, compress_image, write_encoded_datadiff,\
  restore_image_from_file

import tensorflow as tf

batch_size = 64
data_path = "./images"

def load(mode="inpaint"):
    # Specifying the dimensionality of the latent space
    autoencoder, input_shape = make_model()
    print(input_shape)
    checkpoint_path = "weights_" + mode + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
      print("Loading weights from " +  latest + "...")
    autoencoder.load_weights(latest)
    autoencoder.build((batch_size,) + input_shape)
    autoencoder.summary()
    return autoencoder, input_shape

MAX_DIM = 640
block_size = 6
if len( sys.argv ) < 4:
   print( "Synopsis: (-c|-u|-t) <input file|folder> <output file|folder>")
else:
  autoencoder, input_shape = load()
  data_path = sys.argv[2]
  if not os.path.isdir(data_path):
    print ("Not a folder " + data_path)
    exit(-1)
  out_path = sys.argv[3]
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  elif not os.path.isdir(out_path):
    print ("Not a folder " + out_path)
    exit(-1)

  if sys.argv[1] in ("-c", "-C", "--compress"): # compress mode
    print("compress")
    files = tf.data.Dataset.list_files(os.path.join(data_path,"*.jpg"))
    for image_path in files:
      image = prepare_image(image_path, MAX_DIM, input_shape, block_size)
      compressed = compress_image(autoencoder, image, input_shape, block_size)
      base = os.path.basename(image_path.numpy().decode())
      print(base)
      with open(os.path.join(out_path, base + ".bin"), mode="wb") as output_file:
        write_encoded_datadiff(output_file, compressed)
      cv.imwrite(os.path.join(out_path, base + ".png"), image)
      cv.imwrite(os.path.join(out_path, base + ".bmp"), image)
  elif sys.argv[1] in ("-u", "-U", "--uncompress"): # uncompress mode
    print("decompress")
    files = tf.data.Dataset.list_files(os.path.join(data_path,"*.bin"))
    for compressed_path in files:
      image = restore_image_from_file(autoencoder, compressed_path.numpy(), input_shape, block_size)
      data = tf.io.encode_png(image)
      base = os.path.basename(compressed_path.numpy().decode())
      tf.io.write_file(os.path.join(out_path, base + ".png"),data)

  elif sys.argv[1] in ("-t", "-T"): # test mode
    print("test")
    import cv2 as cv
    files = tf.data.Dataset.list_files(os.path.join(data_path,"*.jpg"))
    for image_path in files:
      image = prepare_image(image_path, MAX_DIM, input_shape, block_size)
      compressed = compress_image(autoencoder, image, input_shape, block_size)
      restored = restore_image(autoencoder, compressed, input_shape, block_size)
      cv.imshow("compressed", compressed)
      cv.imshow("original", image)
      cv.imshow("restored", restored)
      if 27 == cv.waitKey():
          break
  else:
    print("Unknown mode switch " + sys.argv[0])
