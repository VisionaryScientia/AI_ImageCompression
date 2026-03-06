import numpy as np


def diff_image_inplace(image:np.ndarray):
  h,w = image.shape[:2]
  for x in range(w-1,0,-1):
     image[:, x] -= image[:, x - 1]
  for y in range(h-1,0,-1):
     image[y, :] -= image[y - 1, :]


def int_image_inplace(diff: np.ndarray):
    h, w = diff.shape[:2]
    for y in range(1, h):
       diff[y, :] += diff[y - 1, :]
    for x in range(1, w):
       diff[ :, x] += diff[ :, x - 1]


def diff_image(image: np.ndarray):
    diff = image.copy()
    diff_image_inplace(diff)
    return diff


def int_image(diff: np.ndarray):
    image = diff.copy()
    int_image_inplace(image)
    return image


def diff_crop(image:np.ndarray):
    h, w = image.shape[:2]
    diff = image.astype(np.int16)
    for x in range(w - 1, 0, -1):
        diff[:, x] -= diff[:, x - 1]
    for y in range(h - 1, 0, -1):
        diff[y, 1:] -= diff[y - 1, 1:]
    diff[0] = image[0]
    return diff


def diff_crop_inplace(image:np.ndarray):
    h, w = image.shape[:2]
    diff = image.copy()
    for x in range(w - 1, 0, -1):
        diff[:, x] -= diff[:, x - 1]
    for y in range(h - 1, 0, -1):
        diff[y, 1:] -= diff[y - 1, 1:]
    image[1:,1:] = diff[1:,1:]


def int_crop(diff:np.ndarray):
    h, w = diff.shape[:2]
    image = diff.copy()
    for x in range(w - 1, 0, -1):
        image[0, x] -= image[0, x - 1]
    for y in range(h - 1, 0, -1):
        image[y, 0] -= image[y - 1, 0]
    int_image_inplace(image)
    return image

def int_crop_inplace(image:np.ndarray):
    h, w = image.shape[:2]
    for x in range(w - 1, 0, -1):
        image[0, x] -= image[0, x - 1]
    for y in range(h - 1, 0, -1):
        image[y, 0] -= image[y - 1, 0]
    int_image_inplace(image)


def test_diff():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    diff = diff_image(image)
    restored = int_image(diff)
    return (image == restored).all()


def test_diff_inplace():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    transformed = image.copy()
    diff_image_inplace(transformed)
    int_image_inplace(transformed)
    return (image == transformed).all()


def test_crop():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    transformed = image.copy()
    transformed = diff_crop(transformed)
    res = int_crop(transformed)
    return (image == res).all()


def test_crop_inplace():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    transformed = image.copy()
    diff_crop_inplace(transformed)
    int_crop_inplace(transformed)
    return (image == transformed).all()

if __name__ == "__main__":
   if test_diff():
      print("image integration test success")
   else:
      print("image integration test failed")
   if test_diff_inplace():
      print("image inplace integration test success")
   else:
      print("image inplace integration test failed")
   if test_crop():
       print("image crop integration test success")
   else:
       print("image crop integration test failed")
   if test_crop_inplace():
       print("image crop inplace integration test success")
   else:
       print("image crop inplace integration test failed")