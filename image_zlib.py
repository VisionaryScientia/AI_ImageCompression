import numpy as np
import zlib
import bitstring

CHUNKSIZE = 1024 * 10

def encode_residuals(compress, diff):
    diff_bytes = diff.tobytes()
    exp_diff_bytes = bitstring.Bits().join(f'se={i}' for i in diff.ravel()).tobytes()
    compressed_data = b""
    if len(diff_bytes) < len(exp_diff_bytes):
        # data too curly, no need exp encoding
        compressed_data += compress.compress(b'\0')
        compressed_data += compress.compress(diff_bytes)
    else:
        compressed_data += compress.compress(b'\1')
        compressed_data += compress.compress(exp_diff_bytes)
    return compressed_data


def decode_residuals(decompressed_data, shape, dtype=np.int16):
    use_exp_coding = decompressed_data[0]
    if use_exp_coding:
        a = bitstring.ConstBitStream(decompressed_data[1:])
        diff_data_list = []
        expected_len = int(shape[0]) * shape[1]
        while expected_len > 0:
           diff_data_list.append(a.read('se'))
           expected_len -= 1
        diff_data = np.array(diff_data_list, dtype)
    else:
        diff_data = np.frombuffer(decompressed_data[1:], dtype=dtype)
    return diff_data.reshape(shape)


def encode_datadiff(diff: np.ndarray, mask: np.array):
    shape = np.zeros(4, dtype=np.int16)
    shape[:2] = diff.shape[:2]
    shape[2:] = mask.shape[:2]
    compressed_data = shape.tobytes()
    compress = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, method=zlib.DEFLATED, wbits=zlib.MAX_WBITS)
    mask_bits = bitstring.Bits(mask.ravel().astype(bool))
    compressed_data += compress.compress(mask_bits.tobytes())
    compressed_data += encode_residuals(compress, diff)
    compressed_data += compress.flush()
    return compressed_data


def encode_latent(latent: np.ndarray, diff: np.ndarray):
    shape = np.zeros(6, dtype=np.int16)
    shape[:2] = diff.shape[:2]
    shape[2:] = latent.shape
    compressed_data = shape.tobytes()
    compress = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, method=zlib.DEFLATED, wbits=zlib.MAX_WBITS)
    compressed_data += compress.compress(latent.tobytes())
    compressed_data += encode_residuals(compress, diff)
    compressed_data += compress.flush()
    return compressed_data

INT16_SIZE = int(2)

def decode_latent(buf):
    shape = np.frombuffer(buf[:INT16_SIZE * 6], np.int16)
    decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    decompressed_data = decomp.decompress(buf[INT16_SIZE * 6:])
    decompressed_data += decomp.flush()
    latent_size = INT16_SIZE * shape[2] * shape[3] * shape[4] * shape[5]
    latent = np.frombuffer(decompressed_data[:latent_size], dtype = np.float16).reshape(shape[2:])
    diff = decode_residuals(decompressed_data[latent_size:], shape[:2], np.int8)
    return latent, diff


def read_encoded_latent(input_stream):
    shape = np.frombuffer(input_stream.read(INT16_SIZE * 6), np.int16)
    decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    buf = input_stream.read(CHUNKSIZE)
    # Decompress stream chunks
    decompressed_data = b''
    while buf:
        decompressed_data += decomp.decompress(buf)
        buf = input_stream.read(CHUNKSIZE)
    decompressed_data += decomp.flush()
    latent_size = INT16_SIZE * shape[2] * shape[3] * shape[4] * shape[5]
    latent = np.frombuffer(decompressed_data[:latent_size], dtype = np.float16).reshape(shape[2:])
    diff = decode_residuals(decompressed_data[latent_size:], shape[:2])
    return latent, diff


def decode_datadiff(buf):
    shape = np.frombuffer(buf[:INT16_SIZE * 4], np.int16)
    decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    decompressed_data = decomp.decompress(buf[INT16_SIZE * 4:])
    mask_size = int(shape[2]) * shape[3]
    mask_size_in_bytes = (mask_size + 7)//8
    bit_array_mask = decompressed_data[:mask_size_in_bytes]
    bit_array = bitstring.ConstBitStream(bit_array_mask)
    mask = [bit for bit in bit_array]
    mask = np.array(mask[:mask_size], np.uint8).reshape(shape[2:4])
    diff = decode_residuals(decompressed_data[mask_size_in_bytes:], shape[:2])
    return diff, mask


def read_encoded_datadiff(input_stream):
    shape = np.frombuffer(input_stream.read(INT16_SIZE * 4), np.int16)
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
    mask = [bit for bit in bit_array]
    mask = np.array(mask[:mask_size], np.uint8).reshape(shape[2:4])
    diff = decode_residuals(decompressed_data[mask_size_in_bytes:], shape[:2])
    return diff, mask


if __name__ == "__main__":

    def test_residuals():
        h, w = 50, 51
        test_data = np.array(np.random.randint(-128, 128, h * w), np.int16).reshape(h, w)
        comp = zlib.compressobj(level=zlib.Z_BEST_COMPRESSION, method=zlib.DEFLATED, wbits=zlib.MAX_WBITS)
        compressed_data = encode_residuals(comp, test_data)
        compressed_data += comp.flush()
        decomp = zlib.decompressobj(wbits=zlib.MAX_WBITS)
        decompressed_data = decomp.decompress(compressed_data)
        decompressed_data += decomp.flush()
        res_data = decode_residuals(decompressed_data, (h,w))
        return (test_data == res_data).all()

    print("residuals encode test",  "success" if test_residuals() else "failed")

    def test_inpaint():
        test_data = np.array(np.random.randint(-128, 128, 50 * 51), np.int16).reshape((50, 51))
        test_data_mask = np.array([np.mod(i, 2) for i in range(10*9)]).reshape(10,9)
        buf = encode_datadiff(test_data, test_data_mask)
        print(test_data)
        print(test_data_mask)
        res_data, res_data_mask = decode_datadiff(buf)
        print(res_data)
        print(res_data_mask)
        return (test_data == res_data).all() and (test_data_mask == res_data_mask).all()

    print("inpaint encode test",  "success" if test_inpaint() else "failed")

    def test_latent():
        test_latent = np.array(np.random.randint(0, 255, 4 * 8 * 16), np.float16).reshape(4,8,16,1)
        test_data = np.array(np.random.randint(-128, 128, 50 * 51), np.int16).reshape(50,51)
        buf = encode_latent(test_latent, test_data)
        res_latent, res_data = decode_latent(buf)
        return (test_latent == res_latent).all() and (test_data == res_data).all()

    print("latent encode test",  "success" if test_latent() else "failed")
