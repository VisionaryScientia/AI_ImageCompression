import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import zlib

from image_cnn import make_model, settings
from evaluate_image import \
  restore_image, prepare_image, compress_image,\
  restore_image_from_file, latent_compress_image, latent_restore_image, \
  compress_image_to_file, calculate_psnr

data_path = "./images"

def load():
    autoencoder, input_shape = make_model('', settings.input_size)
    autoencoder.load_weights(settings.input_size, settings.inpaint_size)
    autoencoder.build((settings.batch_size,) + input_shape)
    autoencoder.summary()
    return autoencoder, input_shape

if len( sys.argv ) < 4:
    print( "Synopsis: (-c|-d|-t|-v) <input file|folder> <output file|folder>")
else:
    autoencoder, input_shape = load()
    mode, data_path, out_path = sys.argv[1:4]
    if not os.path.isdir(data_path):
        print ("Not a folder " + data_path)
        exit(-1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.isdir(out_path):
        print ("Cannot create output folder " + out_path)
        exit(-1)
    if settings.inpaint_size > 0:
        print("Inpaint mode", settings.inpaint_size)
    else:
        print("Latent mode")

    if mode in ("-c", "-C", "--compress"): # compress mode
        print("compress")
        files = tf.data.Dataset.list_files([os.path.join(data_path,"*.jpg"),
                                       os.path.join(data_path,"*.png"),
                                       os.path.join(data_path,"*.gif")], shuffle=False)
        for image_path in files:
            base = os.path.basename(image_path.numpy().decode())
            print(base)
            image, _ = prepare_image(image_path, settings.max_dim, input_shape, settings.inpaint_size)
            compress_image_to_file(autoencoder, image, os.path.join(out_path, base + ".bin"), input_shape,
                                    settings.inpaint_size, settings.batch_size, settings.zero_diff)
            data = tf.io.encode_png(image)
            im_size = image.shape[0] * image.shape[1]
            print("png CR", im_size / tf.strings.length(data).numpy())
            tf.io.write_file(os.path.join(out_path, base + ".png"),data)
            gzip_out = zlib.compress(image)
            assert im_size == len(zlib.decompress(gzip_out)), "Wrong gzip size"
            print("gzip CR", im_size / len(gzip_out))
    elif mode in ("-d", "-D", "--decompress"): # uncompress mode
        print("decompress")
        files = tf.data.Dataset.list_files(os.path.join(data_path,"*.bin"), shuffle=False)
        for compressed_path in files:
            base = os.path.basename(compressed_path.numpy().decode())
            print(base)
            image = restore_image_from_file(autoencoder, compressed_path.numpy(),
              input_shape, settings.inpaint_size, settings.batch_size)
            data = tf.io.encode_png(image)
            base = os.path.basename(compressed_path.numpy().decode())
            tf.io.write_file(os.path.join(out_path, base + "-decompressed.png"), data)
        print("Done")
    elif mode in ("-t", "-T", "-v", "-V"): # test mode
        print("test")
        import cv2 as cv
        files = tf.data.Dataset.list_files([os.path.join(data_path,"*.jpg"),
                                        os.path.join(data_path,"*.png"),
                                        os.path.join(data_path,"*.gif")], shuffle=False)
        for image_path in files:
            image, _ = prepare_image(image_path, settings.max_dim, input_shape, settings.inpaint_size)
            if image is None:
                continue
            base = os.path.basename(image_path.numpy().decode())
            if settings.inpaint_size > 0:
                diff, mask = compress_image(autoencoder, image, input_shape, settings.inpaint_size, settings.batch_size, settings.zero_diff)
                restored = restore_image(autoencoder, diff, mask, input_shape, settings.inpaint_size)
                mask_image_upscaled = cv.resize(mask * 255, (0,0), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
            else:
                latent, diff = latent_compress_image(autoencoder, image, input_shape, settings.batch_size)
                if settings.zero_diff == 1:
                    diff.fill(0)
                restored = latent_restore_image(autoencoder, latent, diff, input_shape, settings.batch_size)
                mask_image_upscaled = None
            psnr = calculate_psnr(image, restored)
            print(image_path.numpy().decode(), "psnr=", psnr)
            cv.imwrite(os.path.join(out_path, base + "-compressed.png"), np.clip(diff + 128, 0, 255).astype(np.int8))
            if not mask_image_upscaled is None:
                cv.imwrite(os.path.join(out_path, base + "-mask.png"), mask_image_upscaled)
            print(image_path.numpy().decode(), "CNN predicted blocks", mask.sum(),
                "of", mask.shape[0] * mask.shape[1],
                "ratio", mask.sum() / (mask.shape[0] * mask.shape[1]))

            visualize = sys.argv[1] == "-v" or sys.argv[1] == "-V"
            if visualize:
                plt.close()
                crop_copy = diff[settings.input_size - settings.inpaint_size:,
                    settings.input_size - settings.inpaint_size:, ...]
                plt.hist(crop_copy.ravel(), bins=60, color='white', edgecolor='black', alpha=0.7)
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.title('Histogram of prediction diff')
                plt.savefig(os.path.join(out_path, base + "_fig_" + str(settings.inpaint_size) + ".png"))
                if not mask_image_upscaled is None:
                    cv.imshow("mask", mask_image_upscaled)
                cv.imshow("diff", np.clip(diff + 128, 0, 255).astype(np.uint8))
                cv.imshow("original", np.clip(image, 0, 255).astype(np.uint8))
                cv.imshow("restored", np.clip(restored, 0, 255).astype(np.uint8))
                plt.show()
    else:
        print("Unknown mode switch " + mode)
