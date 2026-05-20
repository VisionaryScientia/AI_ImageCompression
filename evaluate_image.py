import numpy as np
import math
import tensorflow as tf
import os
from differentiate import *
from image_zlib import \
    decode_datadiff, encode_datadiff,\
    decode_latent, encode_latent

DEBUG_VIEW = False
VERBOSE = True
if DEBUG_VIEW:
    import cv2 as cv


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (np.ndarray): The first image (e.g., original).
        img2 (np.ndarray): The second image (e.g., compressed or denoised).
        max_pixel_value (float): The maximum possible pixel value (default is 255.0 for 8-bit images).

    Returns:
        float: The PSNR value in decibels (dB).
    """
    # Ensure images have float64 data type for accurate calculations
    img1 = np.array(img1, np.float64)
    img2 = np.array(img2, np.float64)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is 0, the images are identical, and PSNR is considered infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR using the formula
    psnr_value = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr_value


def entropy(labels):
    """ Computes entropy of label distribution. """
    labels_num = len(labels)
    if labels_num <= 1:
        return 0
    values, counts = np.unique(labels, return_counts=True)
    if len(values) <= 1:
        return 0
    probs = counts / labels_num
    return -(probs * np.log2(probs)).sum()


def restore_image(model, diff : np.array, mask: np.array, input_shape, block_size):
    image_shape = diff.shape
    h, w = image_shape[:2]
    bh, bw = input_shape[:2]
    y_offset = bh - block_size
    result_image = np.array(diff, np.float32)
    mask.reshape(h//block_size, w//block_size)
    x_offset = bw - block_size
    if DEBUG_VIEW:
        delim = np.ones([bh,1,1], np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if mask[y//block_size,x//block_size]:
                block = result_image[y - y_offset: y + block_size, x - x_offset: x + block_size, :].copy()
                mean_v = (block[: -block_size, :].mean() + block[:, : -block_size].mean()) / 2
                block[-block_size:, -block_size:] = mean_v
                predict = model(block[np.newaxis, ...]/255.)[0,...]
                result_image[ y: y + block_size, x: x + block_size, :] = int_image(diff[y: y + block_size, x: x + block_size, :]) + \
                    np.clip(predict[ -block_size:, -block_size:, :] * 255, 0, 255)
            else:
                shift_x = 1 if x > 0 else 0
                shift_y = 1 if y > 0 else 0
                int_crop_inplace(result_image[y - shift_y: y + block_size, x - shift_x: x + block_size, :])
            if DEBUG_VIEW:
                v = np.hstack([block, delim,
                   result_image[y - y_offset: y + block_size, x - x_offset : x + block_size]]) * 255
                v = cv.resize(v, (0,0), fx=4, interpolation=cv.INTER_NEAREST)
                cv.imshow("preview", np.clip(v, 0, 255).astype(np.uint8))
                cv.waitKey()

    return result_image.astype(np.uint8)


def restore_image_from_file(model, compressed_file_path : str, input_shape, block_size, batch_size):
    """
    :param model:
    :param compressed_file_path:
    :param input_shape:
    :param block_size:
    :return: restored image
    """
    with open(compressed_file_path, "rb") as f:
        buf = f.read()
    if block_size > 0:
        diff, mask = decode_datadiff(buf)
        return restore_image(model, diff[...,np.newaxis], mask, input_shape, block_size)
    else:
        latent, diff = decode_latent(buf)
        return latent_restore_image(model, latent, diff[...,np.newaxis], input_shape, batch_size)


def compress_image(model, image : np.array, input_shape, block_size, batch_size, zero_diff):
    h, w = image.shape[:2]
    bh, bw = input_shape[:2]
    image_blocks = []
    diff = tf.cast(image,np.float32).numpy()
    image_copy = image.numpy()
    predicted_image = np.zeros((h,w), float)
    y_offset = bh - block_size
    x_offset = bw - block_size
    delim = np.ones([bh,1,1], np.float32) * 255
    mask = np.zeros((h//block_size, w//block_size),np.uint8)
    for y in range(y_offset, h, block_size):
        for x in range(x_offset, w, block_size):
            block = diff[y - y_offset:y + block_size, x - x_offset:x + block_size].copy()
            mean_v = (block[: -block_size, :].mean() + block[:, : -block_size].mean())/2
            block[-block_size:, -block_size:]= mean_v
            image_blocks.append(block / 255.)
    test_block_ds = tf.data.Dataset.from_tensor_slices(image_blocks)
    test_ds = test_block_ds.batch(batch_size)
    predicted = model.predict(test_ds)
    idx = 0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if y < y_offset or x < x_offset:
                shift_x = 1 if x > 0 else 0
                shift_y = 1 if y > 0 else 0
                diff[y:y + block_size, x:x + block_size] = diff_crop(
                    image_copy[y - shift_y:y + block_size, x - shift_x:x + block_size])[shift_y:, shift_x:]
                continue
            predicted_block = predicted[idx][-block_size:, -block_size:] * 255
            predicted_image[y:y + block_size, x:x + block_size] = np.squeeze(predicted_block, axis=-1)
            # check prediction quality
            image_block = diff[y:y + block_size, x:x + block_size]
            block_diff = image_block - predicted_block
            clipping_good = tf.reduce_min(block_diff) >= -128 and tf.reduce_max(block_diff) <= 127
            prediction_good = False
            if clipping_good:
                diff_var = tf.math.reduce_variance(block_diff)
                image_var = tf.math.reduce_variance(image_block)
                prediction_good = image_var > diff_var
            if prediction_good or zero_diff == 1 and clipping_good:
                mask[y // block_size, x // block_size] = 1
                diff[y:y + block_size, x:x + block_size] = diff_image(tf.cast(block_diff, np.int16).numpy())
            else:
                diff[y:y + block_size, x:x + block_size] =\
                    diff_crop(image_copy[y-1:y + block_size, x-1:x + block_size])[1:,1:]
            if DEBUG_VIEW and prediction_good:
                v = image[y - y_offset:y + block_size, x - x_offset:x + block_size]
                v = np.hstack([predicted[idx] * 255, delim, image_blocks[idx] * 255, delim, v]).astype(np.uint8)
                v = cv.resize(v, (0,0), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
                cv.imshow("preview", np.clip(v, 0, 255).astype(np.uint8))
                cv.waitKey()
            idx += 1
    return diff.astype(np.int16), mask, np.clip(predicted_image, 0, 255).astype(np.uint8)



def compress_image_to_file(model, image: np.array,
                           compressed_file_path : str, input_shape, block_size, batch_size, zero_diff = 0):
    """
    :param model:
    :param compressed_file_path:
    :param input_shape:
    :param block_size:
    :return: diff - for statistics evaluation
    """
    if block_size > 0:
        diff, mask, _ = compress_image(model, image, input_shape, block_size, batch_size, zero_diff)
        if VERBOSE:
            print("CNN predicted blocks", mask.sum(), "of", mask.shape[0] * mask.shape[1],
                "ratio", mask.sum() / (mask.shape[0] * mask.shape[1]))
        compressed_data = encode_datadiff(diff, mask)
    else:
        latent, diff = latent_compress_image(model, image, input_shape, batch_size)
        if zero_diff == 1:
           diff.fill(0)
        with open(compressed_file_path, "wb") as f:
            compressed_data = encode_latent(latent, diff)

    total_bytes = image.shape[0] * image.shape[1]
    if VERBOSE:
        print(len(compressed_data), "CR", (1. * total_bytes) / len(compressed_data))
    with open(compressed_file_path, "wb") as f:
        f.write(compressed_data)
    return diff



def prepare_image(image_path : str, max_dim : int, block_shape : np.array, inpaint_block_size : int):
    """
    Load image of needed size adjusted for block shape, values is normalized to [0..1]
    :param image_path:
    :param max_dim: the image is resized to this size
    :param block_shape: input size to autoencoder
    :param inpaint_block_size: if 0 no inpainting is applied, should be less than block_shape dimensions
    :return:
    """
    data = tf.io.read_file(image_path)
    _, extension = os.path.splitext(image_path if type(image_path) is str else image_path.numpy().decode("utf-8"))
    extension = extension.lower()
    if extension == ".jpg":
        image = tf.io.decode_jpeg(data, channels=block_shape[2])
    elif extension == ".png":
        image = tf.io.decode_png(data, channels=block_shape[2])
    elif extension == ".bmp":
        image = tf.io.decode_bmp(data, channels=block_shape[2])
    elif extension == ".gif":
        image = tf.io.decode_gif(data)[0]
        image = tf.image.rgb_to_grayscale(image)
    else:
        return None
    h, w = image.shape[:2]
    h_orig, w_orig = h, w
    if h > max_dim and w > max_dim:
        if h > w:
            w = w * max_dim // h
            h = max_dim
        else:
            h = h * max_dim // w
            w = max_dim

    # padding
    if inpaint_block_size > 0:
        pad_h = (h - block_shape[0] + inpaint_block_size) // inpaint_block_size * inpaint_block_size + block_shape[0] - inpaint_block_size
        pad_w = (w - block_shape[1] + inpaint_block_size) // inpaint_block_size * inpaint_block_size + block_shape[1] - inpaint_block_size
    else:
        pad_h = h // block_shape[0] * block_shape[0]
        pad_w = w // block_shape[1] * block_shape[1]

    if h_orig != h or w_orig != w:
        image = tf.image.resize(image, (h, w))
        if VERBOSE:
            print("Input is resized to max acceptable dimension")
    if pad_w > w or pad_h > h:
        image = tf.image.pad_to_bounding_box(image, 0, 0, pad_h, pad_w)
        if VERBOSE:
            print("Input is padded to fit block shape")
    if pad_w < w or pad_h < h:
        image = tf.image.crop_to_bounding_box(image, 0, 0, pad_h, pad_w)
        if VERBOSE:
            print("Input is cropped to fit block shape")
    return tf.cast(image, np.uint8), tf.shape(image)[:2]


scale = 127.
def latent_compress_image(model, image_n : np.array, input_shape, batch_size):
    image = tf.cast(image_n, tf.float32)
    image *= 1 / 255.
    h, w = image.shape[:2]
    image_blocks = []
    for y in range(0, h, input_shape[0]):
        for x in range(0, w, input_shape[1]):
            block = image[y:y + input_shape[0], x:x + input_shape[1]]
            image_blocks.append(block)

    test_block_ds = tf.data.Dataset.from_tensor_slices(image_blocks)
    test_ds = test_block_ds.batch(batch_size)
    latent = model.encoder.predict(test_ds).astype(np.float16)
    latent_block_ds = tf.data.Dataset.from_tensor_slices(latent)
    latent_ds = latent_block_ds.batch(batch_size)
    result = model.decoder.predict(latent_ds)
    idx = 0
    result_image = np.empty(image.shape, np.float32)
    for y in range(0, h, input_shape[0]):
        for x in range(0, w, input_shape[1]):
            result_image[y:y + input_shape[0], x:x + input_shape[1]] = result[idx][:, :]
            idx += 1
    diff = (image - result_image) * scale
    return latent, np.clip(diff, -scale-1, scale).astype(np.int8)


def latent_restore_image(model, latent: np.array, diff : np.array, input_shape, batch_size):
    image_shape = diff.shape
    h, w = image_shape[:2]
    bh, bw = input_shape[:2]
    result_image = tf.Variable(diff.astype(np.float32)/scale)
    latent_block_ds = tf.data.Dataset.from_tensor_slices(latent)
    latent_ds = latent_block_ds.batch(batch_size)
    result = model.decoder.predict(latent_ds)
    idx = 0
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            s = result_image[y:y + bh, x:x + bw] + result[idx][:, :]
            result_image[y:y + bh, x:x + bw].assign(s)
            idx += 1
    result_image = tf.clip_by_value(result_image * 255, 0, 255)
    result_image = tf.cast(result_image, np.uint8)
    return result_image.numpy()


def evaluate_image(model, image_path, inpaint_block_size, max_dim, input_shape, batch_size=64):
    """
    Calculates difference between the original and autoencoder output image.
    This can be useful for estimation of the residual statistics.
    :param model: CNN autoencoder
    :param image_path: test image path
    :param inpaint_block_size: discarded subblock size for inpaining
    :param max_dim: maximum image processing size
    :param input_shape: CNN input shape
    :param batch_size: batch for inference
    :return: reconstructed and original image (resized to maximum size)
    """
    image, (h, w) = prepare_image(image_path, inpaint_block_size=inpaint_block_size, max_dim=max_dim, block_shape=input_shape)
    image = image.numpy() / 255.
    image_blocks = []
    if inpaint_block_size > 0:
        for y in range(0, h - input_shape[0], inpaint_block_size):
            for x in range(0, w - input_shape[1], inpaint_block_size):
                block = image[y:y + input_shape[0], x:x + input_shape[1]]
                block[-inpaint_block_size:, -inpaint_block_size:] = block.mean()
                image_blocks.append(block)
    else:
        for y in range(0, h, input_shape[0]):
            for x in range(0, w, input_shape[1]):
                block = image[y:y + input_shape[0], x:x + input_shape[1]]
                image_blocks.append(block)

    test_block_ds = tf.data.Dataset.from_tensor_slices(image_blocks)
    test_ds = test_block_ds.batch(batch_size)
    latent = model.encoder.predict(test_ds)
    latent_block_ds = tf.data.Dataset.from_tensor_slices(latent)
    latent_ds = latent_block_ds.batch(batch_size)
    result = model.decoder.predict(latent_ds)
    idx = 0
    result_image = np.empty(image.shape, np.float32)
    original_image = np.empty(image.shape, np.float32)
    if inpaint_block_size > 0:
        for y in range(0, h - input_shape[0], inpaint_block_size):
            for x in range(0, w - input_shape[1], inpaint_block_size):
                result_image[y:y + inpaint_block_size, x:x + inpaint_block_size] = result[idx][-inpaint_block_size:, -inpaint_block_size:]
                original_image[y:y + inpaint_block_size, x:x + inpaint_block_size] = image_blocks[idx][-inpaint_block_size:, -inpaint_block_size:]
                idx += 1
    else:
        for y in range(0, h, input_shape[0]):
            for x in range(0, w, input_shape[1]):
                result_image[y:y + input_shape[0], x:x + input_shape[1]] = result[idx][:, :]
                original_image[y:y + input_shape[0], x:x + input_shape[1]] = image_blocks[idx][:, :]
                idx += 1

    result_image *= 255.
    original_image *= 255.
    return result_image.astype(np.uint8), original_image.astype(np.uint8)


if __name__ == "__main__":
    image, (h, w) = prepare_image("./images/train/Test_chart_11.jpg", max_dim=512, block_shape=(28,28,1), inpaint_block_size=8)

    if tf.shape(image)[0] == h and tf.shape(image)[1] == w:
        print("Image prepared correctly")
    else:
        print("Image preparation failed")