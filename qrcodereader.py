from pyzbar.pyzbar import decode
import cv2
import math
import numpy as np

is_rotated = 0  #global variable to determine if the image is correctly orientated

def decode_qr(image):    #pyzbars qr decoder used after rotating
    decoded = decode(image)
    for obj in decoded:
        print('Type:', obj.type)
        print('Data:', obj.data.decode('utf-8'))

def rotate_image(image, angle):  #function to rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def template_matching(image, template_path, threshold=1):   #template matching using sliding window method, difference of squares used at each iteration, dynamically adjusted threshold
    global is_rotated
    print(f"Current threshold: {threshold}")  #debugging line
    
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:    #error handling
        print("Error loading template")
        return image, is_rotated

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_height, template_width = template_gray.shape

    pyramid_levels = 8  #amount of scales of the template used for the template matching
    scale_factor = 1.5
    best_matches = []  #best matches will be stored here

    for level in range(pyramid_levels):   #for each scale we apply template matching
        resized_image = cv2.resize(image_gray, None, fx=1/scale_factor**level, fy=1/scale_factor**level, interpolation=cv2.INTER_LINEAR)
        if resized_image.shape[0] < template_height or resized_image.shape[1] < template_width:
            break

        result = cv2.matchTemplate(resized_image, template_gray, cv2.TM_CCOEFF_NORMED) #template matching using opencv
        loc = np.where(result >= threshold)  #saving all the matches that are above the threshold

        for pt in zip(*loc[::-1]): #drawing bounding boxes
            top_left = (int(pt[0] * scale_factor**level), int(pt[1] * scale_factor**level))
            bottom_right = (int((pt[0] + template_width) * scale_factor**level), int((pt[1] + template_height) * scale_factor**level))
            best_matches.append((top_left, bottom_right))

    if len(best_matches) == 0:  #no matches found
        print("No matches found.")
        if threshold > 0.1:
            return template_matching(image, template_path, threshold - 0.1) #recursively adjusting the threshold level until matches are found
        return image, is_rotated

    boxes = np.array([[x1, y1, x2, y2] for ((x1, y1), (x2, y2)) in best_matches])
    boxes = non_max_suppression_fast(boxes, 0.3)

    if len(boxes) < 2:  #must be 3 matches in order to perform image alignment
        print("Not enough matches found for angle calculation")
        return image, is_rotated

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  #drawing bounding boxes
        print(f"Bounding Box: Top-left: ({x1}, {y1}), Bottom-right: ({x2}, {y2})") 

    delta_x = boxes[0][0] - boxes[1][0]
    delta_y = boxes[0][1] - boxes[1][1]
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    print(f"Angle: {angle_degrees} degrees")

    if len(boxes) >= 3 and (   #  checking for image alignment
        abs(boxes[1][1] - boxes[2][1]) <= 10 and
        abs(boxes[0][0] - boxes[1][0]) <= 10 and
        boxes[0][0] < boxes[2][0]
    ):
        is_rotated = 1
    else:  
        image = rotate_image(image, 90) #if image is not aligned, rotate and start again

    return image, is_rotated

def non_max_suppression_fast(boxes, overlap_thresh): #function for non maximal suppression of largely overlapping bounding boxes
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")

image_path = "cropped_qr.png"    #qr code image path (use a clear, black and white and non tilted qr code)
template_path = "corner.png"

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    print("Error loading image")
    exit()

while is_rotated == 0:
    image, is_rotated = template_matching(image, template_path)
    if is_rotated == 0:
        print("Rotating image...")
    else:
        print("QR code found and aligned.")

cv2.imshow('Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

decode_qr(image)
print("Image processing complete.")
