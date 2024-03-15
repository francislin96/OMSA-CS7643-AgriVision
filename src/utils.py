import cv2
import json
import numpy as np
from shapely import Polygon

def convert_boundary_to_mask(boundary_path: str, image: np.ndarray): 
    """
    Convert a Polygon at boundary_path to a mask of the image

    Args:
        boundary_path: 
            path where boundary.json is located, contains Polygon coordinates
        image:
            image to be masked
    Returns:
        A mask with values of either 0 (exclude) or 255 (include)
    """
    # read the boundary path to extract the points inside the Polygon
    with open(boundary_path) as f:
        features = json.load(f)["coordinates"][0]
        boundary = Polygon([f for f in features])
        points = np.array([[[x, y] for x, y in zip(*boundary.boundary.coords.xy)]])

    # use cv2.fillPoly to extract mask from 
    mask = cv2.fillPoly(np.zeros(image.shape[0:2]), points.astype(image.dtype), color=255).astype(np.uint8)

    return mask

def apply_boundary_to_image(boundary_path, image):
    """
    Apply the boundaries at boundary_path to image

    Args:
        boundary_path: 
            path where boundary.json is located, contains Polygon coordinates
        image:
            image to be masked
    Returns:
        An image with 0 values for areas outside of boundary Polygon
    """
    
    # convert boundary to mask
    mask = convert_boundary_to_mask(boundary_path, image)
    
    # apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask = mask)

    return mask, masked_image