"""The original still-image OpenCV exercises, kept in their learning order."""

import cv2
import numpy as np
from heart_detection import is_heart_contour


image = cv2.imread("image/1.jpg")

if image is None:
    print("Image could not be loaded.")
    exit()

resized_image = cv2.resize(image, (300, 300))
cropped_image = resized_image[50:250, 50:250]
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
 
lower_bound = np.array([0, 70, 50])
upper_bound = np.array([10, 255, 255])
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
masked_result = cv2.bitwise_and(
    resized_image,
    resized_image,
    mask=mask,
)


####? Coordinates for rectangle ######
#?
#1. Create a copy of the resized image
copy_image = resized_image.copy()

height, width = resized_image.shape[:2]

#2. Top-left coordinate — (x1, y1)
top_left = (width // 4, height // 4)

#3. Bottom-right coordinate — (x2, y2)
bottom_right = (3 * width // 4, 3 * height // 4)

#4. Red rectangle
# cv2.rectangle(
#     copy_image,
#     top_left,
#     bottom_right,
#     (0, 0, 255),  # BGR color (red)
#     1            # 1 pixels
# )

#5 
# x1, y1 = top_left


########?

####? Coordinates for rectangle

#3 * 3 blur
mean_blur = cv2.blur(gray_image, (3, 3))

# 5 * 5 Gaussian blur
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Median blur
median = cv2.medianBlur(gray_image, 3)

###?
###
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated_mask = cv2.dilate(mask, kernel, iterations=1)
morphology_result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
Erosion = cv2.erode(mask, kernel, iterations=1)
Closing  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(Closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_areas = [cv2.contourArea(contour) for contour in contours]
minimum_area = 100  

for contour in contours:
    area = cv2.contourArea(contour)
    if area < minimum_area: #21780 < 100
        continue

    cv2.drawContours(copy_image, [contour], -1, (0, 255, 0), 2)

    perimeter = cv2.arcLength(contour, True)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        continue  # A zero-area contour has no usable centroid.
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    x, y, width, height = cv2.boundingRect(contour)

    print(
        "Area:", area,
        "x:", x,
        "y:", y,
        "width:", width,
        "height:", height,
        "Perimeter:", perimeter,
        "Center:", (center_x, center_y)
    )

    cv2.rectangle(
        copy_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    cv2.circle(
        copy_image,
        (center_x, center_y),
        5,
        (255, 0, 0),
        -1
    )

    aspect_ratio = width / height
    print("Aspect ratio:", aspect_ratio)

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx)
    # Check heart geometry first: a heart cannot be identified by corner count alone.
    if is_heart_contour(contour):
        shape_name = "Heart"
    elif corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        shape_name = "Rectangle"
    elif corners > 8:
        shape_name = "Circle-like"
    else:
        shape_name = "Unknown"

    # Keep labeling inside the loop so every accepted contour gets its own name.
    cv2.putText(
        copy_image,                 # Image
        shape_name,                 # TEXT wanted to display
        (x, max(20, y - 10)),        # PLACE text above the rectangle
        cv2.FONT_HERSHEY_SIMPLEX,    # Font
        0.6,                        # Text size
        (0, 255, 0),                # Text color (BGR)
        2,                          # Text thickness
        cv2.LINE_AA
    )

#####? Canny threshold( gradient strength brightness#######
#
# 50 ≤ gradient < 150 → weak edgechagessbetween pixels)
low_edges_without_blur = cv2.Canny(gray_image, 50, 150)
Low_edges_with_blur = cv2.Canny(gaussian_blur, 50, 150)

# Gradient ≥ 150 → strong edge
high_edges_50_150 = cv2.Canny(gray_image, 100, 200)
high_edges_100_200 = cv2.Canny(gaussian_blur, 100, 200)

#@@ image processing on webcam frames can be added here, similar to the static image processing above.
##
#
def process_frame(frame):
    resized_frame = cv2.resize(frame, (300, 300))
    result = resized_frame.copy()

    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closing,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        if not is_heart_contour(contour):
            continue

        x, y, width, height = cv2.boundingRect(contour)

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )
        cv2.putText(
            result,
            "Heart",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return result

###? Webcam/video processing
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera could not be opened.")
    exit()
success, frame = camera.read()
if not success:
    print("Failed to capture frame from camera.")
    camera.release()
    exit()

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame from camera.")
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Webcam Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()

##?

print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Size:", image.size)
print("Dimensions:", image.ndim)
print("Grayscale shape:", gray_image.shape)
print("Grayscale udim:", gray_image.ndim)
print("Contour areas:", contour_areas)
print("mask:", mask.shape, mask.dtype)
print("closing:", Closing.shape, Closing.dtype)

###* Important: Display the images in separate windows########
#
# cv2.imshow("Original Image", image)
# cv2.imshow("Resized Image", resized_image)
# cv2.imshow("Croped Image", cropped_image)
# cv2.imshow("Grayscale Image",gray_image)
# cv2.imshow("mean_blur Image",mean_blur)
# cv2.imshow("gaussian Image",gaussian_blur)
# cv2.imshow("median Image",median)
# cv2.imshow("edges_without_blur",low_edges_without_blur)
# cv2.imshow("edges_with_blur",Low_edges_with_blur)
# cv2.imshow("high_edges_50_150",high_edges_50_150)
# cv2.imshow("high_edges_100_200",high_edges_100_200)
# cv2.imshow("Rectangle", copy_image)
# cv2.imshow("HSV Image", hsv_image)
# cv2.imshow("Mask Image", mask)
# cv2.imshow("Dilated Mask", dilated_mask)
# cv2.imshow("Morphology Result", morphology_result)
# cv2.imshow("Erosion", Erosion)
# cv2.imshow("Closing", Closing)
# cv2.imshow("Opening", Opening)
# cv2.imshow("Masked Result", masked_result)
# cv2.imshow("Contour Areas", contour_areas)
# cv2.imshow("Contours", copy_image)
######* save processed images to the output folder#######
#
saved = cv2.imwrite("output/copy.jpg", resized_image)
cv2.imwrite("output/rectangle.jpg", copy_image)
# cv2.imwrite("output/cropped.jpg", cropped_image)
# cv2.imwrite("output/greyed.jpg", gray_image)
# cv2.imwrite("output/mean_blur.jpg", mean_blur)
# cv2.imwrite("output/gaussian_blur.jpg", gaussian_blur)
# cv2.imwrite("output/median_blur.jpg", median)
# cv2.imwrite("output/edges_without_blur.jpg",low_edges_without_blur)
# cv2.imwrite("output/edges_with_blur.jpg",Low_edges_with_blur)
# cv2.imwrite("output/high_edges_50_150.jpg",high_edges_50_150)
# cv2.imwrite("output/high_edges_100_200.jpg",high_edges_100_200)
# cv2.imwrite("output/mean_blur.jpg", mean_blur)
# cv2.imwrite("output/gaussian_blur.jpg", gaussian_blur)
# cv2.imwrite("output/gaussian_blur.jpg", median)

print("Image saved:", saved)

cv2.waitKey(0)
cv2.destroyAllWindows()


#! testing numpy array
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
