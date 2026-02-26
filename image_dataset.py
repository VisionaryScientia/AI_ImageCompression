import numpy as np
import tensorflow as tf
import os

def read_image(filename:str, inpaint_size:int, shape:(int,int,int)):
    # Load the raw data from the file
    data = tf.io.read_file(filename)
    image = tf.cast(tf.io.decode_jpeg(data, channels=shape[2]), tf.float32)
    image *= 1/255.
    h,w = image.shape[:2]
    if inpaint_size > 0:
        for y in range(0, h - shape[0], inpaint_size):
            for x in range(0, w - shape[1], inpaint_size):
                a = image[y:y+shape[0], x:x+shape[1]].numpy()
                b = a.copy()
                b[-inpaint_size:,-inpaint_size:] = np.mean(b)
                yield (tf.constant(b), tf.constant(a))
    else:
        for y in range(0, h - shape[0], shape[0]):
            for x in range(0, w - shape[1], shape[1]):
                a = image[y:y+shape[0], x:x+shape[1]].numpy()
                b = a.copy()
                yield (tf.constant(b), tf.constant(a))


def process_path(data_dir:str, inpaint_size: int, shape:(int,int,int)):
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
            for r in read_image(filename, inpaint_size, shape):
              yield r

def create_dataset(data_dir: str, batch_size: int,
                   inpaint_size:int, input_shape:(int,int,int)):
    buffer_size = 10000
    train_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/train", inpaint_size, input_shape],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    train_ds=train_block_ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    test_block_ds=tf.data.Dataset.from_generator(process_path, args = [data_dir + "/test", inpaint_size, input_shape],
       output_types=(tf.float32, tf.float32), output_shapes=(input_shape, input_shape))
    test_ds=test_block_ds.batch(batch_size)
    return train_ds, test_ds


if __name__ == "__main__":
    import cv2 as cv
    data_dir="./images/train"
    inpaint_size = 16
    shape = (32,32,3)
    block_ds = tf.data.Dataset.from_generator(process_path, args = [data_dir, inpaint_size, shape],
            output_types=(tf.float32, tf.float32), output_shapes = (shape, shape))
    shuffled_ds = block_ds.shuffle(100)
    count = 0
    for im0,im1 in shuffled_ds:
       v = (np.hstack([im0.numpy(), im1.numpy()]) * 255).astype(np.uint8)
       cv.imshow("test", v)
       if 27 == cv.waitKey(0):
           break
       count += 1
    print("Blocks", count)