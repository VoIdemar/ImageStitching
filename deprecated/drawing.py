import numpy as np
import cv
from stitching.surf.errors import NoFeaturesError, IncompatibleImagesError

class Drawing(object):
    
    DEFAULT_RADIUS = 3
    
    @staticmethod
    def draw_features(image, features, color):
        if features is not None:
            for feature in features:
                cv.Circle(image, (feature[0], feature[1]), Drawing.DEFAULT_RADIUS, color, thickness=1)
        else:
            raise NoFeaturesError('No features to draw')
        
    @staticmethod
    def combine_images(img1, img2):
        if img1.nChannels == img2.nChannels and img1.depth == img2.depth:
            depth = img1.depth
            nChannels = img1.nChannels
            height = np.max([img1.height, img2.height])
            width = img1.width + img2.width        
            result = cv.CreateImage((width, height), depth, nChannels)
            cv.SetImageROI(result, (0, 0, img1.width, img1.height))
            cv.Copy(img1, result)
            cv.ResetImageROI(result)
            cv.SetImageROI(result, (img1.width, 0, img2.width, img2.height))
            cv.Copy(img2, result)
            cv.ResetImageROI(result)
            return result
        else:
            raise IncompatibleImagesError('Incompatible color depth')
    
    @staticmethod
    def draw_correspondence(combined_image, shift, point1, point2, color):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        cv.Circle(combined_image, (x1, y1), Drawing.DEFAULT_RADIUS, color, thickness=1)
        cv.Circle(combined_image, (x2 + shift, y2), Drawing.DEFAULT_RADIUS, color, thickness=1)
        cv.Line(combined_image, (x1, y1), (x2 + shift, y2), color)