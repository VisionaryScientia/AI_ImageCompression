import numpy as np
import zlib
import bitstring
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

CHUNKSIZE = 1024 * 10

def encode_latent(latent: np.ndarray, diff: np.ndarray):
   shape = np.zeros(6, dtype=np.int16)
   shape[:2] = diff.shape[:2]
   shape[2:] = latent.shape
   compressed_data = shape.tobytes()
   compress = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, method=zlib.DEFLATED, wbits=zlib.MAX_WBITS)
   compressed_data += compress.compress(latent.tobytes())
   diff_bits = bitstring.Bits().join(f'se={i}' for i in diff.ravel())
   print("Diff ", int(shape[0]) * shape[1], len(diff_bits.tobytes()))
   if int(shape[0]) * shape[1] < len(diff_bits.tobytes()):
      # data too curly, no need exp encoding
      compressed_data += compress.compress(b'\0')
      compressed_data += compress.compress(diff.tobytes())
   else:
      compressed_data += compress.compress(b'\1')
      compressed_data += compress.compress(diff_bits.tobytes())
   compressed_data += compress.flush()
   return compressed_data


def decode_latent(buf):
   shape = np.frombuffer(buf[:2 * 6], np.int16)
   decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
   decompressed_data = decomp.decompress(buf[2*6:])
   decompressed_data += decomp.flush()
   size = int(2) * shape[2] * shape[3] * shape[4] * shape[5]
   latent = np.frombuffer(decompressed_data[:size], dtype = np.float16).reshape(shape[2:])
   if decompressed_data[size] == 1:
      a = bitstring.ConstBitStream(decompressed_data[size + 1:])
      diff_data_list = []
      len_a = int(shape[0]) * int(shape[1])
      count = 0
      while count < len_a:
         diff_data_list.append(a.read('se'))
         count += 1
      diff_data = np.array(diff_data_list, dtype = np.int8)
   else:
      diff_data = np.frombuffer(decompressed_data[size + 1:], dtype = np.int8)
   return latent, diff_data.reshape(shape[:2])


def read_encoded_latent(input_stream):
   shape = np.frombuffer(input_stream.read(2 * 6), np.int16)
   decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
   buf = input_stream.read(CHUNKSIZE)
   # Decompress stream chunks
   decompressed_data = b''
   while buf:
      decompressed_data += decomp.decompress(buf)
      buf = input_stream.read(CHUNKSIZE)
   decompressed_data += decomp.flush()
   size = int(2) * shape[2] * shape[3] * shape[4] * shape[5]
   latent = np.frombuffer(decompressed_data[:size], dtype = np.float16).reshape(shape[2:])
   if decompressed_data[size] == 1:
      a = bitstring.ConstBitStream(decompressed_data[size + 1:])
      diff_data_list = []
      len_a = int(shape[0]) * int(shape[1])
      count = 0
      while count < len_a:
         diff_data_list.append(a.read('se'))
         count += 1
      diff_data = np.array(diff_data_list, dtype = np.int8)
   else:
      diff_data = np.frombuffer(decompressed_data[size + 1:], dtype = np.int8)
   return latent, diff_data.reshape(shape[:2])


def encode_datadiff(diff: np.ndarray, mask: np.array):
   shape = np.zeros(4, dtype=np.int16)
   shape[:2] = diff.shape[:2]
   shape[2:] = mask.shape[:2]
   compressed_data = shape.tobytes()
   compress = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, method=zlib.DEFLATED, wbits=zlib.MAX_WBITS)
   mask_bits = bitstring.Bits(mask.ravel().astype(bool))
   compressed_data += compress.compress(mask_bits.tobytes())
   #pdiff = diff.ravel()
   #abs_diff = tf.math.abs(pdiff)
   #sign = bitstring.Bits(pdiff >= 0)
   #diff_bits = bitstring.Bits().join(f'ue={i}' for i in abs_diff)
   #compressed_data += compress.compress(diff_bits.tobytes())
   #compressed_data += compress.compress(sign.tobytes())

   diff_bits = bitstring.Bits().join(f'se={i}' for i in diff.ravel())
   print("Diff ", int(shape[0]) * shape[1], len(diff_bits.tobytes()))
   if int(shape[0]) * shape[1] < len(diff_bits.tobytes()):
     # data too curly, no need exp encoding
      compressed_data += compress.compress(b'\0')
      compressed_data += compress.compress(diff.tobytes())
   else:
      compressed_data += compress.compress(b'\1')
      compressed_data += compress.compress(diff_bits.tobytes())
   compressed_data += compress.flush()
   return compressed_data


def decode_datadiff(buf):
   shape = np.frombuffer(buf[:4 * 2], np.int16)
   decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
   decompressed_data = decomp.decompress(buf[4 * 2:])

   mask_size = int(shape[2]) * shape[3]
   mask_size_in_bytes = (mask_size + 7)//8
   bit_array_mask = decompressed_data[:mask_size_in_bytes]
   bit_array = bitstring.ConstBitStream(bit_array_mask)
   mask = [bit for bit in bit_array]
   mask = mask[:mask_size]
   mask = np.array(mask, np.uint8).reshape(shape[2:])
   if decompressed_data[mask_size_in_bytes] == 1:
      a = bitstring.ConstBitStream(decompressed_data[mask_size_in_bytes + 1:])
      diff_data_list = []
      len_a = int(shape[0]) * int(shape[1])
      count = 0
      while count < len_a:
         diff_data_list.append(a.read('se'))
         count += 1
      diff_data = np.array(diff_data_list, np.int16)
   else:
      diff_data = np.frombuffer(decompressed_data[mask_size_in_bytes + 1:], dtype = np.int16)
   diff = diff_data.reshape(shape[:2])
   return diff, mask


def read_encoded_datadiff(input_stream):
   shape = np.frombuffer(input_stream.read(4 * 2), np.int16)
   decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
   buf = input_stream.read(CHUNKSIZE)
   # Decompress stream chunks
   decompressed_data = b''
   while buf:
      decompressed_data += decomp.decompress(buf)
      buf = input_stream.read(CHUNKSIZE)
   decompressed_data += decomp.flush()

   mask_size = int(shape[2]) * shape[3]
   mask_size_in_bytes = (mask_size + 7)//8
   bit_array_mask = decompressed_data[:mask_size_in_bytes]
   bit_array = bitstring.ConstBitStream(bit_array_mask)
   mask = []
   for bit in bit_array:
      mask.append(bit)
   mask = mask[:mask_size]
   mask = np.array(mask, np.uint8).reshape(shape[2:])
   #sign_size = int(shape[0]) * shape[1]
   #sign_size_in_bytes = (sign_size + 7)//8
   #bit_array_sign = decompressed_data[mask_size_in_bytes : mask_size_in_bytes + sign_size_in_bytes ]
   #bit_array = bitstring.ConstBitStream(bit_array_sign)
   #sign = []
   #for bit in bit_array:
   #   sign.append(bit)
   #sign = sign[:sign_size]
   #diff_data = np.frombuffer(decompressed_data[mask_size_in_bytes + sign_size_in_bytes :], dtype = '<u1')
   #diff = diff_data.astype(np.float32) * np.where(sign, 1., -1.)
   #diff = diff.reshape(shape[:2])
   diff_data = np.frombuffer(decompressed_data[mask_size_in_bytes :], dtype = '<u1')
   diff = diff_data.astype(np.float32)
   diff = diff.reshape(shape[:2])

   return diff, mask


def test_diff():
   test_data = np.array(np.random.randint(-128, 128, 50 * 51), np.int8).reshape((50, 51))
   test_data_mask = np.array([np.mod(i, 2) for i in range(10*9)]).reshape(10,9)

   # test_data = np.array([[20,10,30,30],[20,20,50,30]], np.uint8)
   buf = encode_datadiff(test_data, test_data_mask)
   print(test_data, test_data_mask)

   res_data, res_data_mask = decode_datadiff(buf)
   print(res_data, res_data_mask)
   for a, b in zip(test_data.ravel(), res_data.ravel()):
      if abs(a - b) > 0.005:
         return False
   for a, b in zip(test_data_mask.ravel(), res_data_mask.ravel()):
      if a != b:
         return False
   return True


def test_latent():
   test_latent = np.array(np.random.randint(0, 255, 4 * 8 * 16), np.float16).reshape((4,8,16,1))
   test_data = np.array(np.random.randint(-128, 128, 50 * 51), np.int8).reshape((50, 51))
   buf = encode_latent(test_latent, test_data)
   res_data_latent, res_data = decode_latent(buf)
   print(res_data, res_data_latent)
   for a, b in zip(test_data.ravel(), res_data.ravel()):
      if a != b:
         return False
   for a, b in zip(test_latent.ravel(), res_data_latent.ravel()):
      if a != b:
         return False
   return True


if __name__ == "__main__":
   if test_diff():
      print("Zlib diff test success")
   else:
      print("Zlib diff test failed")

   if test_latent():
      print("Zlib latent test success")
   else:
      print("Zlib latent test failed")

