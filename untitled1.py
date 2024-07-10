import cv2
import numpy as np

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def rotate_right(arr, k):
    k = k % len(arr)
    return arr[-k:] + arr[:-k]

def template_matching(image_path, template_path, threshold=0.3):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    
    if image is None or template is None:
        print("Error loading image or template")
        return

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_height, template_width = template_gray.shape

    pyramid_levels = 3
    scale_factor = 1.5
    best_matches = []

    for level in range(pyramid_levels):
        resized_image = cv2.resize(image_gray, None, fx=1/scale_factor**level, fy=1/scale_factor**level, interpolation=cv2.INTER_LINEAR)
        if resized_image.shape[0] < template_height or resized_image.shape[1] < template_width:
            break

        result = cv2.matchTemplate(resized_image, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            top_left = (int(pt[0] * scale_factor**level), int(pt[1] * scale_factor**level))
            bottom_right = (int((pt[0] + template_width) * scale_factor**level), int((pt[1] + template_height) * scale_factor**level))
            best_matches.append((top_left, bottom_right))

    boxes = np.array([[x1, y1, x2, y2] for ((x1, y1), (x2, y2)) in best_matches])
    boxes = non_max_suppression_fast(boxes, 0.3)

    i = 0
    while True:
        # Check if condition met to stop rotating
        if i > 0 and abs(boxes[1][3] - boxes[2][3]) <= 20:
            break
        
        # Rotate bounding boxes
        rotate_right(boxes, 1)
        i += 1

    # Rotate image
    rotated_image = rotate_image(image, i * 90)

    for (x1, y1, x2, y2) in boxes:
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        cv2.rectangle(rotated_image, top_left, bottom_right, (0, 255, 0), 2)
        print(f"Bounding Box: Top-left: {top_left}, Bottom-right: {bottom_right}")

    cv2.imshow('Detected', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression_fast(boxes, overlap_thresh):
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

# Example usage
image_path = "C:/Users/farid/qrcodereader/cropped_qr.png"
template_path = "C:/Users/farid/qrcodereader/corner.png"

template_matching(image_path, template_path)

