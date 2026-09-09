import cv2
import numpy as np

image = cv2.imread("image/1.jpg")

if image is None:
    print("Image could not be loaded.")
    exit()

resized_image = cv2.resize(image, (300, 300))
cropped_image = resized_image[50:250, 50:250]
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

#3 * 3 blur
mean_blur = cv2.blur(gray_image, (3, 3))

# 5 * 5 Gaussian blur
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Median blur
median = cv2.medianBlur(gray_image, 3)

##Canny threshold( gradient strength brightness 
#
# 50 ≤ gradient < 150 → weak edgechagessbetween pixels)
low_edges_without_blur = cv2.Canny(gray_image, 50, 150)
Low_edges_with_blur = cv2.Canny(gaussian_blur, 50, 150)

# Gradient ≥ 150 → strong edge
high_edges_50_150 = cv2.Canny(gray_image, 100, 200)
high_edges_100_200 = cv2.Canny(gaussian_blur, 100, 200)




print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Size:", image.size)
print("Dimensions:", image.ndim)
print("Grayscale shape:", gray_image.shape)
print("Grayscale udim:", gray_image.ndim)

# cv2.imshow("Original Image", image)
# cv2.imshow("Resized Image", resized_image)
# cv2.imshow("Croped Image", cropped_image)
# cv2.imshow("Grayscale Image",gray_image)
# cv2.imshow("mean_blur Image",mean_blur)
# cv2.imshow("gaussian Image",gaussian_blur)
# cv2.imshow("median Image",median)
cv2.imshow("edges_without_blur",low_edges_without_blur)
cv2.imshow("edges_with_blur",Low_edges_with_blur)
cv2.imshow("high_edges_50_150",high_edges_50_150)
cv2.imshow("high_edges_100_200",high_edges_100_200)

saved = cv2.imwrite("output/copy.jpg", resized_image)
# cropped_saved = cv2.imwrite("output/cropped.jpg", cropped_image)
# grayed_saved = cv2.imwrite("output/greyed.jpg", gray_image)
mean_saved = cv2.imwrite("output/mean_blur.jpg", mean_blur)
gaussian_saved = cv2.imwrite("output/gaussian_blur.jpg", gaussian_blur)
median_saved = cv2.imwrite("output/median_blur.jpg", median)

# without_blur_saved = cv2.imwrite("output/edges_without_blur.jpg",low_edges_without_blur)
with_blur_saved = cv2.imwrite("output/edges_with_blur.jpg",Low_edges_with_blur)

without_blur_saved = cv2.imwrite("output/high_edges_50_150.jpg",high_edges_50_150)
with_blur_saved = cv2.imwrite("output/high_edges_100_200.jpg",high_edges_100_200)


# cv2.imwrite("output/mean_blur.jpg", mean_blur)
# cv2.imwrite("output/gaussian_blur.jpg", gaussian_blur)
# cv2.imwrite("output/gaussian_blur.jpg", median)

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