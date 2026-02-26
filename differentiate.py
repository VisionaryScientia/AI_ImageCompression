import numpy as np


def diff_image(image:np.ndarray):
  h,w = image.shape[:2]
  diff = image.copy()
  for x in range(w-1, 0, -1):
     diff[1:, x] -= diff[1:, x - 1]
  for y in range(h-1, 1, -1):
     diff[y, 1:] -= diff[y - 1, 1:]
  return diff


def int_diff(diff: np.ndarray):
    h, w = diff.shape[:2]
    image = diff.copy()
    for y in range(1, h):
       image[y, 1:] += image[y - 1, 1:]
    for x in range(1, w):
       image[1:, x] += image[1:, x - 1]
    return image


def diff_image_inplace(image:np.ndarray):
  h,w = image.shape[:2]
  for x in range(w-1,0,-1):
     image[1:, x] -= image[1:, x - 1]
  for y in range(h-1,0,-1):
     image[y, 1:] -= image[y - 1, 1:]


def int_image_inplace(diff: np.ndarray):
    h, w = diff.shape[:2]
    for y in range(2, h):
       diff[y,1:] += diff[y - 1,1:]
    for x in range(1, w):
       diff[1:,x] += diff[1:,x - 1]


def test_diff():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    diff = diff_image(image)
    restored = int_diff(diff)
    return (image == restored).all()


def test_diff_inplace():
    data = [x for x in range(100)]
    image = np.array(data,np.float).reshape(10,10)
    transformed = image.copy()
    diff_image_inplace(transformed)
    int_image_inplace(transformed)
    return (image == transformed).all()


if __name__ == "__main__":
   if test_diff():
      print("image integration test success")
   else:
      print("image integration  test failed")
   if test_diff_inplace():
      print("image inplace integration test success")
   else:
      print("image inplace integration  test failed")