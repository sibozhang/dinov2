import cv2
import numpy as np

# depth_image_dir = 'depth/north-040_depth.jpeg'
# depth_image_dir = 'depth/south-040_depth.jpeg'
# depth_image_dir = 'depth/shendong_20230815133500-20230815134500_1_022_depth.jpeg'
depth_image_dir = 'depth/shendong_20230815133500-20230815134500_1_022_zoedepth.png'

# depth_image_dir = 'depth/shendong4_地面工业广场-布尔台井口_20230815133500-20230815134500_1_102_depth.jpeg'
# depth_image_dir = 'depth/shendong4_地面工业广场-布尔台井口_20230815133500-20230815134500_1_102_zoedepth.png'

# label_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp86/labels/north-040.txt'
# label_dir = '/mnt/scratch/sibo/yolov5/runs/detect/exp87/labels/south-040.txt'
label_dir = '/mnt/data/sibo/china_crane/shendong/dino_4class/train/labels/shendong_20230815133500-20230815134500_1_022.txt'
# label_dir = '/mnt/data/sibo/china_crane/shendong/dino_4class/train/labels/shendong4_地面工业广场-布尔台井口_20230815133500-20230815134500_1_102.txt'
# depth_img_output_dir = 'depth/north-040_depth_bbox.png'
# depth_img_output_dir = 'depth/south-040_depth_bbox.png'
# depth_img_output_dir = 'depth/shendong_20230815133500-20230815134500_1_022_depth_bbox_color.png'
depth_img_output_dir = 'depth/shendong_20230815133500-20230815134500_1_022_zoedepth_bbox_color.png'
# depth_img_output_dir = 'depth/shendong4_地面工业广场-布尔台井口_20230815133500-20230815134500_1_102_zoedepth_bbox_color.png'

# Class names
class_names = {
    0: 'person', 1: 'car', 2: 'truck', 3: 'machinery vehicle'
}

# class_names = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
#     5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
#     10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
#     14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
#     19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
#     24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
#     28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
#     32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
#     36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
#     44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 
#     48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
#     52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
#     56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#     60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
#     64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
#     68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
#     72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
#     76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
#     79: 'toothbrush', 80: 'machinery vehicle'
# }

# Load depth image
depth_img = cv2.imread(depth_image_dir, cv2.IMREAD_ANYDEPTH)
img_height, img_width = depth_img.shape
print(depth_img.shape)

# Copy original depth image for depth value access
depth_img_copy = depth_img.copy()

# Print the minimum and maximum depth values in the image
min_depth = np.amin(depth_img)
max_depth = np.amax(depth_img)

print("Minimum depth in image: ", min_depth)
print("Maximum depth in image: ", max_depth)

# Apply colormap
depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

# Load labels
with open(label_dir, 'r') as f:
    labels = f.readlines()

for label in labels:
    data = [float(x) for x in label.strip().split()]
    class_id, x_center, y_center, width, height = data

    # Convert normalized coordinates to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate bounding box coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    bbox = [x1, y1, x2, y2]

    # Calculate bottom center of the bounding box
    center_x = int((bbox[0] + bbox[2]) / 2)
    bottom_y = int(bbox[3])
    
    # Access depth value at the bottom center of the bounding box
    # Invert the depth values
    depth_img_copy = max_depth - depth_img_copy
    depth_value = depth_img_copy[bottom_y, center_x]

    class_name = class_names.get(int(class_id), 'Unknown')
    print(f"class, depth, position: {class_name}, {depth_value}, ({center_x}, {bottom_y})")

    # Draw bounding box and labels on depth image
    cv2.rectangle(depth_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    cv2.circle(depth_img, (center_x, bottom_y), 2, (0, 0, 0), 1)
    cv2.putText(depth_img, f"{depth_value}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save the depth image in a specified directory
cv2.imwrite(depth_img_output_dir, depth_img)

print("image wrote to: ", depth_img_output_dir)

