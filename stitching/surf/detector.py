import logging
from logging import DEBUG, INFO

import cv2
import numpy as np

from stitching.cvtools.integral import IntegralImage
from stitching.cvtools import imgtools
from stitching.surf.constants import SCALE_FACTORS, SCALES, OCTAVE_MAP, OUTLIER_DIST_THRESHOLD
from stitching.surf.refinement import ScaleSpaceRefiner
from stitching.surf.descriptor import SURFDescriptor
from stitching.surf import cudatools
from stitching.mathtools import points

logger = logging.getLogger('detectorLogger')

class SURFDetector(object):
    
    FILTER_MAP = [[0, 1, 2, 3], 
                  [1, 3, 4, 5], 
                  [3, 5, 6, 7], 
                  [5, 7, 8, 9], 
                  [7, 9, 10, 11]]
    N = 1
    HESSIAN_RELATIVE_WEIGHT = 0.912
    SCALE_INDICES = range(1, 11)
    
    def __init__(self, image, threshold=1):        
        self.__threshold = threshold
        self.__image = image 
        grayscale = imgtools.ndarray_to_grayscale(image)
        self.__grayscale = imgtools.ndarray_normalize_image(grayscale)
        self.__features = []
        self.__traces = []
        self.__responses = []
        self.__descriptors = []
        self.__integral_image = None        
    
    @property
    def features(self):
        return self.__features
    
    @property
    def responses(self):
        return self.__responses
    
    @property
    def descriptors(self):
        return self.__descriptors
    
    @property
    def threshold(self):
        return self.__threshold
    
    def extract(self):
        self.__integral_image = IntegralImage(self.__grayscale)
        
        if logger.isEnabledFor(INFO):
            logger.info('==========================================================')
            logger.info('SURF started.')
            logger.info('Computing hessians...')
            
        self.__build_resp_layers()
        
        if logger.isEnabledFor(INFO):
            logger.info('Detecting features...')
            
        self.__detect_features()
        
        if logger.isEnabledFor(INFO):
            logger.info('%s interest points found', len(self.features))
            
        self.__scale_space_refinement()
        
        if logger.isEnabledFor(INFO):
            logger.info('%s interest points found after refinement', len(self.features))
            logger.info('Extraction success! Building descriptors...')
            
        self.__build_descriptors()
        
        if logger.isEnabledFor(INFO):
            logger.info('Descriptors obtaned. SURF finished successfully')
            
        self.__save_responses_to_files(u'D:\\test\\testing')
        self.__clear_unnecessary_data()
    
    @staticmethod
    def extract_features(image, threshold=1):
        detector = SURFDetector(image, threshold)        
        detector.extract()
        return detector.descriptors
    
    def __build_resp_layers(self):       
        grayscale = self.__grayscale
        responses = []
        traces = []    
        for s in SCALES:
            if logger.isEnabledFor(DEBUG):
                logger.debug('Computing hessian in scale %s...', s)
                
            octave = OCTAVE_MAP[s]
            scale = 2**(octave - 1)
            h, w = grayscale.shape
            h, w = h/scale, w/scale
            
            if logger.isEnabledFor(DEBUG):
                logger.debug('Response size: (%s, %s)' % (w, h))
                
            hessian, trace = self.__build_response(w, h, s, scale)
            responses.append(hessian)
            traces.append(trace)            
        self.__responses = responses
        self.__traces = traces
    
    def __build_response(self, w, h, s, scale):
        integral = self.__integral_image
        L = s/3
        scale_factor_dxx_dyy = 6*L*(2*L - 1)
        scale_factor_dxy = 4*L*L        
        
        dxx = integral.nd_dxx(L)[::scale, ::scale]
        dyy = integral.nd_dyy(L)[::scale, ::scale]
        dxy = integral.nd_dxy(L)[::scale, ::scale] 
                   
        dxx = dxx/scale_factor_dxx_dyy
        dyy = dyy/scale_factor_dxx_dyy
        dxy = SURFDetector.HESSIAN_RELATIVE_WEIGHT*dxy/scale_factor_dxy
        hessian = dxx*dyy - dxy**2
        trace = np.sign(dxx + dyy)
        return (hessian, trace)
    
    def __is_maximum(self, x, y, b, m, t, indM):
        curr_value = m[y, x]
        if abs(curr_value) <= self.threshold:
            return False
        scale_b, _, scale_t = SCALE_FACTORS[indM]
        bx, by = int(scale_b*x), int(scale_b*y)
        tx, ty = int(scale_t*x), int(scale_t*y)
        for (dx, dy) in ((dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)):
            if curr_value < max((b[by+dy, bx+dx], m[y+dy, x+dx], t[ty+dy, tx+dx])):
                return False
        return True
    
    def __is_not_outlier(self, x, y, s):
        """ 
        Checks whether (x, y, s) is far enough from the borders.
        """
        scale_indices = SURFDetector.SCALE_INDICES
        if s in scale_indices:
            dist = OUTLIER_DIST_THRESHOLD[s]
            h, w = self.__responses[s].shape
            return (dist < x < (w-dist) and 
                    dist < y < (h-dist))
        else:
            return False  
    
    def __detect_features(self):
        resp_layers = self.__responses
        is_maximum = self.__is_maximum
        is_not_outlier = self.__is_not_outlier
        features = self.__features
        traces = self.__traces
        
        for i in range(len(SURFDetector.FILTER_MAP)):
            for j in range(1, 3):
                b = resp_layers[SURFDetector.FILTER_MAP[i][j-1]]
                indM = SURFDetector.FILTER_MAP[i][j]
                m = resp_layers[indM]
                t = resp_layers[SURFDetector.FILTER_MAP[i][j+1]]
                h, w = m.shape
#                 scale_b, _, scale_t = SCALE_FACTORS[indM]
#                 dist_threshold = OUTLIER_DIST_THRESHOLD[indM]
#                 rowIdx, colIdx = cudatools.detect_features(b, m, t, self.threshold, scale_b, scale_t, dist_threshold)
#                 scales = np.empty_like(rowIdx)
#                 scales.fill(indM)
#                 lapl_signs = traces[indM][rowIdx, colIdx]              
#                 l = map(tuple, np.dstack((colIdx, rowIdx, scales, lapl_signs)).tolist()[0])
#                 features = features + l
                #print result
                for x, y in ((x, y) for x in range(1, w-1) for y in range(1, h-1)):
                    if is_not_outlier(x, y, indM) and is_maximum(x, y, b, m, t, indM):
                        features.append((x, y, indM, traces[indM][y, x]))
                    else:
                        continue
                    
        self.__features = features
    
    def __scale_space_refinement(self):
        refiner = ScaleSpaceRefiner(self.__features, self.__responses)
        refiner.refine()
        self.__features = refiner.refined_features
        refiner.release_data()
        
    def __build_descriptors(self):
        features = self.__features        
        descriptors = []
        for feature in features:
            descriptors.append(SURFDescriptor.build_by_integral(feature, self.__integral_image))
        self.__descriptors = descriptors
    
    def __clear_unnecessary_data(self):
        self.__integral_image = None
        self.__responses = None
        self.__grayscale = None
    
    def __save_responses_to_files(self, dirname):
        import os
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for s in range(len(SCALES)):
            scale = SCALES[s]
            cv2.imwrite(u'{0}\\response {1}.jpg'.format(dirname, scale), self.__responses[s])