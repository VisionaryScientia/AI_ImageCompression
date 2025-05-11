import numpy as np
import tensorflow as tf
import os
import cv2
import scipy
from image_zlib import read_encoded_datadiff, write_encoded_datadiff

def restore_image(model, diff : np.array, input_shape, block_size):
    image_shape = diff.shape
    h, w = image_shape[:2]
    bh, bw = input_shape[:2]
    result_image = diff.astype(np.float32)
    result_image /= 255.
    y_offset = bh - block_size
    x_offset = bw - block_size
    result_image[y_offset:,x_offset:] -= 0.5
    result_image = result_image[np.newaxis, ...]
    #delim = np.ones([bh,1,1], np.float32)

    for y in range(y_offset, h, block_size):
        for x in range(x_offset, w, block_size):
            block = result_image[:, y - y_offset: y + block_size, x - x_offset : x + block_size, :].copy()
            mean_v = 0.5 * (block[:, : -block_size,:].mean() + block[:, :, : -block_size].mean())
            block[:,-block_size:, -block_size:] = mean_v
            result = model(block)
            result_image[:, y: y + block_size, x: x + block_size, :] += \
                result[:, -block_size:, -block_size:, :]
            #v = np.hstack([block[0], delim, result[0], delim,
            #   result_image[0, y - y_offset: y + block_size, x - x_offset : x + block_size]]) * 255
            #v = scipy.ndimage.zoom(v, 4, order=0)
            #cv2.imshow("preview", np.clip(v, 0, 255).astype(np.uint8))
            #cv2.waitKey()

    result_image = np.clip(result_image[0] * 255., 0, 255).astype(np.uint8)
    return result_image


def restore_image_from_file(model, compressed_file_path : str, input_shape, block_size):
    """
    :param model:
    :param compressed_file_path:
    :param input_shape:
    :param block_size:
    :return: restored image
    """
    diff = None
    with open(compressed_file_path, "rb") as f:
        diff = read_encoded_datadiff(f)
    return restore_image(model, diff[...,np.newaxis], input_shape, block_size)


def compress_image(model, image : np.array, input_shape, block_size):
    h, w = image.shape[:2]
    bh, bw = input_shape[:2]
    image_blocks = []

    image_n = tf.cast(image, tf.float32)
    image_n *= 1 / 255.

    y_offset = bh - block_size
    x_offset = bw - block_size
    #delim = np.ones([bh,1,1], np.float32)
    for y in range(y_offset, h, block_size):
        for x in range(x_offset, w, block_size):
            block = image_n[y - y_offset:y + block_size, x - x_offset:x + block_size].numpy()
            v = 0.5 * (block[: -block_size, :].mean() + block[:, : -block_size].mean())
            block[-block_size:, -block_size:] = v
            image_blocks.append(block)
    batch_size = 64
    test_block_ds = tf.data.Dataset.from_tensor_slices(image_blocks)
    test_ds = test_block_ds.batch(batch_size)
    result = model.predict(test_ds)
    predict = np.ones(image.shape, np.float32) * 0.5
    idx = 0
    for y in range(y_offset, h, block_size):
        for x in range(x_offset, w, block_size):
            predict[y:y + block_size, x:x + block_size] = \
                result[idx][-block_size:, -block_size:]
            #v = image_n[y - y_offset:y + block_size, x - x_offset:x + block_size]
            #v = np.hstack([result[idx], delim, image_blocks[idx], delim, v]) * 255
            #v = scipy.ndimage.zoom(v, 4, order=0)
            #cv2.imshow("preview", np.clip(v, 0, 255).astype(np.uint8))
            #cv2.waitKey()
            idx += 1
    diff = np.clip((0.5 + image_n - predict).numpy() * 255, 0, 255).astype(np.uint8)
    #diff = (0.5 + image_n - predict).numpy() * 255
    return diff

def compress_image_to_file(model, image: np.array,
                           compressed_file_path : str, input_shape, block_size):
    """
    :param model:
    :param compressed_file_path:
    :param input_shape:
    :param block_size:
    :return: None
    """
    diff = compress_image(model, image, input_shape, block_size)
    with open(compressed_file_path, "wb") as f:
        write_encoded_datadiff(f, diff)


def prepare_image(image_path : str, max_dim : int, block_shape, block_size):
    """
    Load image of needed size adjusted for block shape, values is normalized to [0..1]
    :param image_path:
    :param max_dim:
    :param block_shape:
    :return:
    """
    data = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(data, channels=block_shape[2])
    h, w = image.shape[:2]

    if h > max_dim and w > max_dim:
        if h > w:
            w = w * max_dim // h
            h = max_dim
        else:
            h = h * max_dim // w
            w = max_dim
    h = (h - block_shape[0]) // block_size * block_size + block_shape[0]
    w = (w - block_shape[1]) // block_size * block_size + block_shape[1]
    image = tf.image.resize(image, (h, w))
    return image.numpy().astype(np.uint8)


def evaluate_image(model, image_path, block_size, max_dim, input_shape, batch_size=64):
   data = tf.io.read_file(image_path)
   image = tf.io.decode_jpeg(data, channels=input_shape[2])
   h,w = image.shape[:2]
   
   if h > max_dim and w > max_dim:
       if h > w:
           w = w * max_dim // h 
           h = max_dim
       else:
           h = h * max_dim // w 
           w = max_dim
       h = h // input_shape[0] * input_shape[0]
       w = w // input_shape[1] * input_shape[1]
       image = tf.image.resize(image, (h, w))
   image = tf.cast(image, tf.float32)
   image *= 1/255.
   h,w = image.shape[:2]
   image_blocks = []
   if block_size > 0: 
      for y in range(0, h - input_shape[0], block_size):
         for x in range(0, w - input_shape[1], block_size):
             block = image[y:y+input_shape[0], x:x+input_shape[1]].numpy()
             block[-block_size:, -block_size:] = block.mean()
             image_blocks.append(block)
   else:
     for y in range(0, h, input_shape[0]):
         for x in range(0, w, input_shape[1]):
             block = image[y:y+input_shape[0], x:x+input_shape[1]].numpy()
             image_blocks.append(block)

   test_block_ds = tf.data.Dataset.from_tensor_slices(image_blocks)
   test_ds = test_block_ds.batch(batch_size)
   latent = model.encoder.predict(test_ds)
   latent_block_ds = tf.data.Dataset.from_tensor_slices(latent.astype(np.float16))
   latent_ds = latent_block_ds.batch(batch_size)
   result = model.decoder.predict(latent_ds)
   idx = 0
   result_image = np.empty(image.shape, np.float32)
   original_image = np.empty(image.shape, np.float32)
   if block_size > 0:
      for y in range(0, h - input_shape[0], block_size):
         for x in range(0, w - input_shape[1], block_size):
             result_image[y:y+block_size, x:x+block_size] = result[idx][-block_size:, -block_size:]
             original_image[y:y+block_size, x:x+block_size] = image_blocks[idx][-block_size:, -block_size:]
             idx += 1  
   else:
      for y in range(0, h, input_shape[0]):
         for x in range(0, w, input_shape[1]):
             result_image[y:y+input_shape[0], x:x+input_shape[1]] = result[idx][:, :]
             original_image[y:y+input_shape[0], x:x+input_shape[1]] = image_blocks[idx][:, :]
             idx += 1  

   result_image *= 255.
   original_image *= 255.
   
   base = os.path.basename(image_path.numpy().decode())
   print(base)
   tf.io.write_file(os.path.join("./output", base + "_result.png"), tf.io.encode_png(result_image.astype(np.uint8)))
   tf.io.write_file(os.path.join("./output", base + "_original.png"), tf.io.encode_png(original_image.astype(np.uint8)))
   return latent, result_image - original_image
