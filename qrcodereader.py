from pyzbar.pyzbar import decode
import cv2
import math
import numpy as np

is_rotated = 0
def decode_qr(image):
    decoded = decode(image)
    for obj in decoded:
        print('Type:', obj.type)
        print('Data:', obj.data.decode('utf-8'))
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def template_matching(image, template_path, threshold=0.3):
    global is_rotated

    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    
    if template is None:
        print("Error loading template")
        return image, is_rotated

    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_height, template_width = template_gray.shape

    # Create image pyramid
    pyramid_levels = 3
    scale_factor = 1.5
    best_matches = []

    for level in range(pyramid_levels):
        # Resize the image for the current pyramid level
        resized_image = cv2.resize(image_gray, None, fx=1/scale_factor**level, fy=1/scale_factor**level, interpolation=cv2.INTER_LINEAR)
        if resized_image.shape[0] < template_height or resized_image.shape[1] < template_width:
            break

        # Perform template matching
        result = cv2.matchTemplate(resized_image, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        # Store bounding box corners with the correct scale
        for pt in zip(*loc[::-1]):
            top_left = (int(pt[0] * scale_factor**level), int(pt[1] * scale_factor**level))
            bottom_right = (int((pt[0] + template_width) * scale_factor**level), int((pt[1] + template_height) * scale_factor**level))
            best_matches.append((top_left, bottom_right))

    # Apply non-maximum suppression
    if len(best_matches) == 0:
        print("No matches found.")
        return image, is_rotated

    boxes = np.array([[x1, y1, x2, y2] for ((x1, y1), (x2, y2)) in best_matches])
    boxes = non_max_suppression_fast(boxes, 0.3)

    if len(boxes) < 2:
        print("Not enough matches found for angle calculation")
        return image, is_rotated

    # Draw rectangles around matched regions and print coordinates
    for (x1, y1, x2, y2) in boxes:
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        print(f"Bounding Box: Top-left: {top_left}, Bottom-right: {bottom_right}")
    print(f"box 1 {boxes[0][0] ,boxes[0][1] }")
    print(f"box 2 {boxes[1][0] ,boxes[1][1] }")
    print(f"box 3 {boxes[2][0] ,boxes[2][1] }")
    # Calculate angle between the first two boxes
    delta_x = boxes[0][0] - boxes[1][0]
    delta_y = boxes[0][1] - boxes[1][1]
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    print(f"Angle: {angle_degrees} degrees")

    # Check for perpendicular alignment
    if len(boxes) >= 3 and (
        abs(boxes[1][1] - boxes[2][1]) <= 10 and  # Vertical alignment check
        abs(boxes[0][0] - boxes[1][0]) <= 10 and
        boxes[0][0] < boxes[2][0]):  # Angle check
        is_rotated = 1
    else:
        image = rotate_image(image, 90)

    return image, is_rotated

def non_max_suppression_fast(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to float as we need to do calculations
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")

# Provide the paths to your images here
image_path = "your_qr_code_path"
template_path = "corner.png"

# Load the initial image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    print("Error loading image")
    exit()

while is_rotated == 0:
    image, is_rotated = template_matching(image, template_path)

    # Display the result
    cv2.imshow('Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

decode_qr(image)
print("Image processing complete.")
