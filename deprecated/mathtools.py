import itertools
import numpy as np
import cv

class MathTools(object):
    
    SIGMA = [1.2, 2.0, 2.8, 3.6, 5.2, 6.8, 10, 13.2, 19.6, 26, 38.8, 51.6]
    NEIGHBOURS = np.array([[0, 0, 0],
                           [0, 1, 2], 
                           [1, 2, 3], 
                           [1, 3, 4], 
                           [3, 4, 5], 
                           [3, 5, 6], 
                           [5, 6, 7], 
                           [5, 7, 8], 
                           [7, 8, 9], 
                           [7, 9, 10], 
                           [9, 10, 11]])
    TO_RADS_RATIO = np.pi / 180
    COLLINEARITY_PRECISION = 0.01
    
    @staticmethod
    def integral_image(src):
        width, height = src.width + 1, src.height + 1
        integral = cv.CreateImage((width, height), cv.IPL_DEPTH_64F, 1)
        cv.Integral(src, integral)
        return integral[1:height, 1:width]
    
    #Normalizing the image in case of low contrast
    @staticmethod
    def normalize_image(image):
        result = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_64F, 1)
        minimum, maximum, _, _ = cv.MinMaxLoc(image)
        for y in range(0, image.height):
            for x in range(0, image.width):
                result[y, x] = int(255*(image[y, x] - minimum) / (maximum - minimum))
        return result
    
    @staticmethod
    def depth_to_matrix_type(depth):
        if depth == cv.IPL_DEPTH_8U:
            return cv.CV_8U
        elif depth == cv.IPL_DEPTH_8S:
            return cv.CV_8S
        elif depth == cv.IPL_DEPTH_16U:
            return cv.CV_16U
        elif depth == cv.IPL_DEPTH_16S:
            return cv.CV_16S
        elif depth == cv.IPL_DEPTH_32S:
            return cv.CV_32S
        elif depth == cv.IPL_DEPTH_32F:
            return cv.CV_32F
        elif depth == cv.IPL_DEPTH_64F:
            return cv.CV_64F
        return -1    
                
    @staticmethod
    def list_intersection(list1, list2):
        s2 = set(list2)
        return [elem for elem in list1 if elem in s2]
    
    @staticmethod
    def list_subtraction(list1, list2):
        s2 = set(list2)
        return [elem for elem in list1 if not (elem in s2)]
    
    #2D non-maximum suppression
    @staticmethod
    def nms_2d(array, W, H, n):
        result = []
        pointSet = itertools.product(range(0, H-n), range(0, W-n))
        maxLength = np.max([W-n, H-n])
        nSet = np.arange(n, maxLength, n+1)
        n2Set = itertools.product(nSet, nSet)
        pixelSet = MathTools.list_intersection(n2Set, pointSet)
        for (i, j) in pixelSet:
            failed = False
            (mi, mj) = (i, j)
            set1 = set(itertools.product(range(i, i+n+1), range(j, j+n+1)))
            for (i2, j2) in set1:
                if array[i2, j2] > array[mi, mj]:
                    (mi, mj) = (i2, j2)
            set2 = set(itertools.product(range(mi-n, mi+n+1), range(mj-n, mj+n+1)))
            for (i2, j2) in (set2 - set1):
                if (i2 >= H) or (j2 >= W) or (array[i2, j2] >= array[mi, mj]):
                    failed = True
                    break 
            if not failed:
                result.append((mi, mj))
        return result
    
    # Returns the value of the Gaussian function:
    @staticmethod 
    def gaussian(x, y, sigma):
        temp = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        return temp / (2.0 * np.pi * sigma**2)
    
    @staticmethod
    def du_dx(responses, x, y, s):
        h = responses[s]
        return 0.5*(h[y, x+1] - h[y, x-1])
    
    @staticmethod
    def du_dy(responses, x, y, s):
        h = responses[s]
        return 0.5*(h[y+1, x] - h[y-1, x])
    
    @staticmethod
    def du_ds(responses, x, y, s):
        bottom = MathTools.NEIGHBOURS[s, 0]
        middle = MathTools.NEIGHBOURS[s, 1]
        top = MathTools.NEIGHBOURS[s, 2]
        return (responses[top][y, x] - responses[bottom][y, x]) / (2.0*(MathTools.SIGMA[top] - MathTools.SIGMA[middle]))
    
    @staticmethod
    def du2_dx2(responses, x, y, s):
        h = responses[s]
        return h[y, x+1] - 2.0*h[y, x] + h[y, x-1]
    
    @staticmethod
    def du2_dy2(responses, x, y, s):
        h = responses[s]
        return h[y-1, x] - 2.0*h[y, x] + h[y+1, x]
    
    @staticmethod
    def du2_ds2(responses, x, y, s):
        bottom = MathTools.NEIGHBOURS[s, 0]
        middle = MathTools.NEIGHBOURS[s, 1]
        top = MathTools.NEIGHBOURS[s, 2]
        h = responses
        return (h[top][y, x] - 2.0*h[middle][y, x] + h[bottom][y, x]) / ((MathTools.SIGMA[top] - MathTools.SIGMA[middle])**2)
    
    @staticmethod
    def du2_dxdy(responses, x, y, s):
        h = responses[s]
        return 0.25*(h[y+1, x+1] + h[y-1, x-1] - h[y+1, x-1] - h[y-1, x+1])
    
    @staticmethod
    def du2_dxds(responses, x, y, s):
        bottom = MathTools.NEIGHBOURS[s, 0]
        middle = MathTools.NEIGHBOURS[s, 1]
        top = MathTools.NEIGHBOURS[s, 2]
        h = responses
        return 0.25*(h[top][y,x+1] - h[top][y, x-1] - 
                     h[bottom][y, x+1] + h[bottom][y, x-1]) / (MathTools.SIGMA[top] - MathTools.SIGMA[middle])
    
    @staticmethod
    def du2_dyds(responses, x, y, s):
        bottom = MathTools.NEIGHBOURS[s, 0]
        middle = MathTools.NEIGHBOURS[s, 1]
        top = MathTools.NEIGHBOURS[s, 2]
        h = responses
        return 0.25*(h[top][y+1, x] - h[top][y-1, x] - 
                     h[bottom][y+1, x] + h[bottom][y-1, x]) / (MathTools.SIGMA[top] - MathTools.SIGMA[middle])
    
    @staticmethod
    def are_noncollinear(points):
        for i in range(0, len(points) - 1):
            for j in range(i + 1, len(points)):
                (x1, y1), (x2, y2) = points[i], points[j]
                for k in range(j + 1, len(points)):
                    (x, y) = points[k]
                    if abs((y1 - y2)*x + (x2 - x1)*y + (x1*y2 - x2*y1)) < MathTools.COLLINEARITY_PRECISION:
                        return False
        return True
    
    @staticmethod
    def to_homogeneous(point):
        return [point[0], point[1], 1]
    
    @staticmethod
    def to_cartesian(hg_point):
        return (hg_point[0] / hg_point[2], hg_point[1] / hg_point[2])