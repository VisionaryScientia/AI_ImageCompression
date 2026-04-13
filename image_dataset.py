import os
import random

import numpy as np
import tensorflow as tf


def augment_block(input_block: np.ndarray):
    k = random.randint(0, 3)
    if k > 0:
        input_block = tf.image.rot90(input_block, k)
    if random.randint(0, 3) < 2:
        input_block = tf.image.flip_left_right(input_block)
    if random.randint(0, 3) < 2:
        input_block = tf.image.flip_up_down(input_block)
    return input_block


def list_image_files(data_dir: str):
    return [os.path.join(data_dir, os.fsdecode(file)) for file in os.listdir(data_dir)
        if os.fsdecode(file).lower().endswith(".jpg")]


def count_blocks_in_image(filename: str, inpaint_size: int, shape: (int, int, int)):
    data = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(data, channels=shape[2])
    h, w = image.shape[:2]
    if inpaint_size > 0:
        y_count = max(0, (h - shape[0]) // inpaint_size + 1)
        x_count = max(0, (w - shape[1]) // inpaint_size + 1)
    else:
        step = shape[0]
        y_count = max(0, (h - shape[0]) // step + 1)
        x_count = max(0, (w - shape[1]) // step + 1)
    return y_count * x_count


def count_dataset_blocks(data_dir: str, inpaint_size: int, input_shape: (int, int, int)):
    return sum(count_blocks_in_image(filename, inpaint_size, input_shape) for filename in list_image_files(data_dir))


def read_image(filename: str, inpaint_size: int, shape: (int, int, int), augment: bool):
    """
    Load the raw data from the image and split into blocks
    """
    data = tf.io.read_file(filename)
    image = tf.cast(tf.io.decode_jpeg(data, channels=shape[2]), tf.float32)
    image *= 1/255.
    h, w = image.shape[:2]
    y_limit = h - shape[0]
    x_limit = w - shape[1]

    if inpaint_size > 0:
        step = inpaint_size
        y_offset = random.randint(0, max(0, step - 1)) if y_limit > 0 else 0
        x_offset = random.randint(0, max(0, step - 1)) if x_limit > 0 else 0
        coords = [(y, x)
            for y in range(y_offset, y_limit + 1, step)
            for x in range(x_offset, x_limit + 1, step)]
        for y, x in coords:
            a = image[y:y + shape[0], x:x + shape[1]]
            if augment:
                a = augment_block(a)
            b = a.numpy()
            b[-inpaint_size:, -inpaint_size:] = np.mean(a)
            yield (tf.constant(b), tf.constant(a))
    else:
        y_offset = random.randint(0, max(0, shape[0] - 1)) if y_limit > 0 else 0
        x_offset = random.randint(0, max(0, shape[1] - 1)) if x_limit > 0 else 0
        coords = [(y, x)
            for y in range(y_offset, y_limit + 1, shape[0])
            for x in range(x_offset, x_limit + 1, shape[1])]
        for y, x in coords:
            a = image[y:y + shape[0], x:x + shape[1]]
            if augment:
                a = augment_block(a)
            yield (tf.constant(a), tf.constant(a))


def process_path(data_dir: str, inpaint_size: int, shape: (int, int, int), augment: bool):
    """
    :param data_dir: imput directory with train images
    :param inpaint_size: inpainted block dimensions, right down corner of the context
    :param shape: context block dimensions (32,32,3)
    :return:
    """
    dir_path = data_dir.decode()
    if dir_path.lower().endswith(".jpg"):
        for r in read_image(dir_path, inpaint_size, shape, augment):
            yield r
    else:
        files = list_image_files(dir_path)
        random.shuffle(files)
        for filename in files:
            for r in read_image(filename, inpaint_size, shape, augment):
                yield r


def create_dataset(data_dir: str, batch_size: int,
                   inpaint_size:int, input_shape:(int,int,int)):
    buffer_size = 1000
    train_steps = max(1, count_dataset_blocks(data_dir + "/train", inpaint_size, input_shape) // batch_size)
    validate_steps = max(1, count_dataset_blocks(data_dir + "/validate", inpaint_size, input_shape) // batch_size)
    train_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/train", inpaint_size, input_shape, True],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    train_ds=train_block_ds.shuffle(buffer_size=buffer_size).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    test_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/validate", inpaint_size, input_shape, False],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    test_ds=test_block_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds, train_steps, validate_steps


if __name__ == "__main__":
    import cv2 as cv
    data_dir="./images"
    inpaint_size = 16
    batch_size = 8
    shape = (32,32,3)
    train_ds, validate_ds, _, _ = create_dataset(data_dir, batch_size, inpaint_size, shape)
    shuffled_ds = train_ds.shuffle(100)
    count = 0
    VIEW_BLOCK = False
    COUNT_N = batch_size * 100
    key = 0
    for im0,im1 in shuffled_ds:
        for k in range(batch_size):
            if VIEW_BLOCK:
                v = (np.hstack([im0[k].numpy(), im1[k].numpy()]) * 255).astype(np.uint8)
                cv.imshow("test", v)
                key = cv.waitKey(0)
        count += 1
        if count == COUNT_N or key == 27:
            break
    print("Batch Blocks generation test:", "succeeded" if count == COUNT_N else "failed")
