import cv2
import numpy as np

image = cv2.imread("image/1.jpg")
resized_image = cv2.resize(image, (300, 300))

if image is None:
    print("Image could not be loaded.")
    exit()

print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Size:", image.size)
print("Dimensions:", image.ndim)
cv2.imshow("Original Image", image)
cv2.imshow("Resized Image", resized_image)
saved = cv2.imwrite("output/copy.jpg", image)
print("Image saved:", saved)

cv2.waitKey(0)
cv2.destroyAllWindows()

# array = [1, 2, 3]
# result = []

# for row in range(3):
#     new_row = []
#     for value in array:
#         new_row.append((value + row * 3) * 10)
#     result.append(new_row)
#     arr = np.array(result)
# arr[2, 2] = 255
# print(arr)
# print(arr.shape)
# print(arr.dtype)
# print(arr.size)
# print(arr.ndim)
# print(arr[0,:])
# print(arr[:,0])
# print(arr[2,:])
# print(arr[:,2])
# print(arr[1,1])
# print(arr[2, 2])
# print("first 2 rows",arr[:2, :])
# print(" first 2 columns",arr[:,:2])
# print("last 2 rows",arr[-2:, :])
# print("last 2 columns",arr[:,-2:])
# print("middle",arr[1:,1:])
# print("row reverse",arr[:, ::-1])
# print("arr.nbytes",arr.nbytes)
# print("arr.itemsize",arr.itemsize)
# print("arr.strides",arr.strides)
# print("arr.flags",arr.flags)
# print("arr.real",arr.real)
# print("arr.imag",arr.imag)
# print("arr.flat",arr.flat)
# print("arr.flat[0]",arr.flat[0])
# print("arr.flat[1]",arr.flat[1])
# print("arr.flat[2]",arr.flat[2])