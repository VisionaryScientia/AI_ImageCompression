import numpy as np
import tensorflow as tf
import os
import random

def read_image(filename:str, inpaint_size:int, shape:(int,int,int), augment:bool=False):
    # Load the raw data from the file
    data = tf.io.read_file(filename)
    image = tf.cast(tf.io.decode_jpeg(data, channels=shape[2]), tf.float32)
    image *= 1/255.
    h,w = image.shape[:2]
    if augment:
        if inpaint_size > 0:
            x0 = random.randint(0, inpaint_size)
            y0 = random.randint(0, inpaint_size)
            step = inpaint_size
        else:
            x0 = random.randint(0, shape[1])
            y0 = random.randint(0, shape[0])
            step = np.min(shape[:2])//2
        y1 = h - shape[0] - step
        x1 = w - shape[1] - step
    else:
        x0 = 0
        y0 = 0
        x1 = w - shape[1]
        y1 = h - shape[0]
        step = inpaint_size if inpaint_size > 0 else (np.min(shape[:2])//2)

    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            a = image[y:y+shape[0], x:x+shape[1]].numpy()
            b = a.copy()
            if inpaint_size > 0:
                b[-inpaint_size:,-inpaint_size:] = np.mean(b)
            yield (tf.constant(b), tf.constant(a))


def process_path(data_dir:str, inpaint_size: int, shape:(int,int,int), augment:bool=False):
    """
    :param data_dir: imput directory with train images
    :param inpaint_size: inpainted block dimensions, right down corner of the context
    :param shape: context block dimensions (32,32,3)
    :return:
    """
    dir_path = data_dir.decode()
    if dir_path.lower().endswith(".jpg"):
        for r in read_image(dir_path, inpaint_size, shape):
             yield r
    else:
        for file in os.listdir(dir_path):
            filename = os.fsdecode(file)
            filename = os.path.join(dir_path, filename)
            if filename.lower().endswith(".jpg"):
                for r in read_image(filename, inpaint_size, shape, augment):
                    yield r


def create_dataset(data_dir: str, batch_size: int,
                   inpaint_size:int, input_shape:(int,int,int)):
    buffer_size = 10000
    train_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/train", inpaint_size, input_shape, False],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    train_ds=train_block_ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    test_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/validate", inpaint_size, input_shape],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    test_ds=test_block_ds.batch(batch_size)
    return train_ds, test_ds


if __name__ == "__main__":
    import cv2 as cv
    data_dir="./images"
    inpaint_size = 16
    batch_size = 8
    shape = (32,32,3)
    train_ds, test_ds = create_dataset(data_dir, batch_size, inpaint_size, shape)
    shuffled_ds = train_ds.shuffle(100)
    count = 0
    view = False
    COUNT_N = batch_size * 100
    key = 0
    for im0,im1 in shuffled_ds:
        for k in range(batch_size):
            if view:
                v = (np.hstack([im0[k].numpy(), im1[k].numpy()]) * 255).astype(np.uint8)
                cv.imshow("test", v)
                key = cv.waitKey(0)
        count += 1
        if count == COUNT_N or key == 27:
            break

    print("Batch Blocks test:", "succeeded" if count == COUNT_N else "failed")