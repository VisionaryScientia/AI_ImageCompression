import numpy as np
import zlib


def write_encoded_datadiff(output, diff: np.ndarray):
   shape = np.array(diff.shape[:2], dtype=np.int16)
   output.write(shape.tobytes())
   compress = zlib.compressobj(zlib.Z_BEST_COMPRESSION, zlib.DEFLATED, +15) #, strategy=zlib.Z_HUFFMAN_ONLY)
   compressed_data = compress.compress(diff.tobytes())
   compressed_data += compress.flush()
   print(len(compressed_data))
   output.write(compressed_data)


def read_encoded_datadiff(input_stream):
   shape = np.frombuffer(input_stream.read(4), np.int16)
   CHUNKSIZE = 1024 * 10
   decomp = zlib.decompressobj()
   buf = input_stream.read(CHUNKSIZE)
   # Decompress stream chunks
   decompressed_data = b''
   while buf:
      decompressed_data += decomp.decompress(buf)
      buf = input_stream.read(CHUNKSIZE)
   decompressed_data += decomp.flush()
   diff_data = np.frombuffer(decompressed_data, dtype = '<u1').reshape(shape)
   return diff_data


if __name__ == "__main__":
   test_data = np.array(np.random.randint(0,255,500 * 501), np.uint8).reshape((500,501))
   #test_data = np.array([[20,10,30,30],[20,20,50,30]], np.uint8)
   with open("compressed.bin","wb") as f:
      write_encoded_datadiff(f, test_data)
   print(test_data)

   res_data = None
   with open("compressed.bin","rb") as f:
      res_data = read_encoded_datadiff(f)
   print(res_data)
   for a,b in zip(test_data.ravel(), res_data.ravel()):
     if a != b:
        print("Zlib test failed")
        exit(0)
   print("Zlib test success")
