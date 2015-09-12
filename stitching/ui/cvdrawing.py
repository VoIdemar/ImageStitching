from math import cos, sin
import cv
import cv2

from stitching.surf.constants import SIZES, SCALES
from stitching.surf.errors import NoFeaturesError
    
DEFAULT_RADIUS = 3

def draw_feature(image, descriptor, color=(0, 255, 255)):
    x, y, s, _ = descriptor.feature
    theta = descriptor.orientation
    size = SIZES[s]
    x *= size
    y *= size
    r = SCALES[s]/3
    cv2.circle(image, (x, y), r, color, thickness=1)
    X, Y = cv.Round(x + r*cos(theta)), cv.Round(y + r*sin(theta))
    cv2.line(image, (x, y), (X, Y), color)

def draw_features(image, descriptors, color=(0, 255, 255)):        
    if descriptors is not None:
        for d in descriptors:
            draw_feature(image, d, color)
    else:
        raise NoFeaturesError('No features to draw')

def draw_match(feature1, feature2, combined_image, shift, color=(0, 255, 255)):
    if len(feature1) == 4 and len(feature2) == 4:
        x1, y1, s1, _ = feature1
        x2, y2, s2, _ = feature2
        c1, c2 = SIZES[s1], SIZES[s2]
        x1, y1 = c1*x1, c1*y1
        x2, y2 = c2*x2, c2*y2
    else:
        x1, y1 = feature1
        x2, y2 = feature2
    cv2.circle(combined_image, (x1, y1), DEFAULT_RADIUS, color, thickness=1)
    cv2.circle(combined_image, (x2 + shift, y2), DEFAULT_RADIUS, color, thickness=1)
    cv2.line(combined_image, (x1, y1), (x2 + shift, y2), color)

def draw_matches(matches, combined_image, shift, color=(0, 255, 255)):
    for (d1, d2) in matches:
        draw_match(d1, d2, combined_image, shift, color)