import cv2
import numpy as np

depth_image_dir = 'depth/shendong4_地面工业广场-布尔台井口_20230815133500-20230815134500_1_103_depth.jpeg'
ply_file_dir = 'depth/pointcloud.ply'

# Load depth image
depth_img = cv2.imread(depth_image_dir, cv2.IMREAD_ANYDEPTH)
img_height, img_width = depth_img.shape
print(depth_img.shape)

# Copy original depth image for depth value access
depth_img_copy = depth_img.copy()

# Assuming camera intrinsics
fx = 1125.0  # Focal length
fy = 1125.0
cx = img_width / 2  # Optical center x
cy = img_height / 2  # Optical center y

# Function to convert depth image to point cloud
def depth_to_pointcloud(depth_img, fx, fy, cx, cy):
    # Initialize pointcloud array
    pointcloud = np.empty((img_height, img_width, 3))

    # Get coordinates
    x = np.linspace(0, img_width - 1, img_width)
    y = np.linspace(0, img_height - 1, img_height)
    x, y = np.meshgrid(x, y)

    # Backproject to 3D (divide by 1000 to convert depth from mm to meters if necessary)
    pointcloud[..., 0] = (x - cx) * depth_img / fx
    pointcloud[..., 1] = (y - cy) * depth_img / fy
    pointcloud[..., 2] = depth_img

    return pointcloud

# Generate pointcloud
pointcloud = depth_to_pointcloud(depth_img_copy, fx, fy, cx, cy)

#save pointcloud
from plyfile import PlyData, PlyElement

def save_pointcloud_as_ply(pointcloud, filename):
    # Reshape the point cloud to N x 3 shape
    reshaped_pointcloud = pointcloud.reshape(-1, 3)

    # Create vertex list
    vertex = np.array([tuple(x) for x in reshaped_pointcloud], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    # Create PlyElement
    el = PlyElement.describe(vertex, 'vertex')

    # Write PlyData
    PlyData([el], text=True).write(filename)

# Save the point cloud as a PLY file
save_pointcloud_as_ply(pointcloud, ply_file_dir)

print("Point cloud saved as a PLY file: ", ply_file_dir)