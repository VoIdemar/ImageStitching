import logging
from logging import DEBUG

import numpy as np
import numpy.ma as ma
import cv

from stitching.surf.constants import SIGMA, SCALES, SIZES
from stitching.mathtools.functions import gaussian
from stitching.mathtools.constants import TO_RADS_RATIO
import stitching.surf.tools as tools

logger = logging.getLogger('descriptorLogger')

class SURFDescriptor(object):
    
    STATISTICS_INDICES_X, STATISTICS_INDICES_Y = tools.get_twodim_indices(5)
    STATISTICS_SAMPLES_COUNT = 25
    DESCRIPTOR_COMP_INDICES_X, DESCRIPTOR_COMP_INDICES_Y = tools.get_twodim_indices(4)
    ORIENTATION_INDICES_X, ORIENTATION_INDICES_Y = tools.get_orientation_indices()
    ORIENTATION_INDICES_LEN = ORIENTATION_INDICES_X.shape[0] 
    
    STARTING_ANGLE = 0
    LAST_ANGLE = 351
    TEST_ANGLES = np.arange(STARTING_ANGLE, LAST_ANGLE, 9)
    TEST_ANGLES_COUNT = TEST_ANGLES.shape[0]
    TEST_VECTORIZED_ANGLES = tools.get_vectorized_angles(ORIENTATION_INDICES_LEN, TEST_ANGLES)
    LEFT_EDGE_ANGLES = TEST_VECTORIZED_ANGLES - 30
    RIGHT_EDGE_ANGLES = TEST_VECTORIZED_ANGLES + 30
    
    def __init__(self, feature):
        self.__feature = feature
        self.__orientation = None
        self.__descriptor = None        
    
    @property
    def feature(self):
        return self.__feature
    
    @property
    def orientation(self):
        return self.__orientation
    
    @property
    def laplacian_sign(self):
        """ 
        Returns laplacian sign of the feature.
        """
        return self.feature[-1]
    
    @property
    def data(self):
        return self.__descriptor
    
    def __getitem__(self, index):
        return self.__descriptor[index]
    
    @staticmethod
    def build_by_integral(feature, integral_image):
        if logger.isEnabledFor(DEBUG):
            logger.debug('Obtaining descriptor of the feature {0}'.format(feature))
            
        descriptor = SURFDescriptor(feature)
        
        if logger.isEnabledFor(DEBUG):
            logger.debug('Finding orientation...')
            
        descriptor.__find_orientation(integral_image)
        
        if logger.isEnabledFor(DEBUG):
            logger.debug('Orientation angle: %s' % descriptor.orientation)
            
        descriptor.__build_descriptor(integral_image)
        
        if logger.isEnabledFor(DEBUG):
            logger.debug('Descriptor obtained successfully:\n{0}'.format(descriptor))
            
        return descriptor
    
    def __find_orientation(self, integral_image):
        test_angles_count = SURFDescriptor.TEST_ANGLES_COUNT
        left_edge_angles = SURFDescriptor.LEFT_EDGE_ANGLES
        right_edge_angles = SURFDescriptor.RIGHT_EDGE_ANGLES
        x_indices = SURFDescriptor.ORIENTATION_INDICES_X
        y_indices = SURFDescriptor.ORIENTATION_INDICES_Y
        idx_count = SURFDescriptor.ORIENTATION_INDICES_LEN
        
        x, y, scale, _ = self.feature
        sigma = SIGMA[scale]
        L = int(sigma)
        size = SIZES[scale]
        x *= size
        y *= size
        
        # Calculate orientations using Haar responses
        X = tools.nd_round(x + x_indices*sigma)
        Y = tools.nd_round(y + y_indices*sigma)
        gauss = gaussian(X-x, Y-y, 2*sigma)
        haar_resp_x = gauss*integral_image.nd_haar_x(L)[Y, X]
        haar_resp_y = gauss*integral_image.nd_haar_y(L)[Y, X]
        angles = np.degrees(np.arctan2(haar_resp_y, haar_resp_x)) + 180
        vectorized_angles = (np.zeros((test_angles_count, idx_count)) + 
                             angles.reshape((1, idx_count)))
        
        # Filter unsuitable directions
        angles_mask = (left_edge_angles <= vectorized_angles)*(vectorized_angles <= right_edge_angles)
        vectorized_haar_x = tools.extend_dim(haar_resp_x, test_angles_count)
        vectorized_haar_y = tools.extend_dim(haar_resp_y, test_angles_count)
        filtered_haar_x = ma.MaskedArray(vectorized_haar_x, mask=angles_mask).filled(0)
        filtered_haar_y = ma.MaskedArray(vectorized_haar_y, mask=angles_mask).filled(0)
        
        # Find the longest orientation vector
        resp_x = np.sum(filtered_haar_x, axis=1)
        resp_y = np.sum(filtered_haar_y, axis=1)
        resp_len = np.sqrt(resp_x**2 + resp_y**2)
        max_resp_len_idx = np.argsort(resp_len)[-1]
        max_resp_x = resp_x[max_resp_len_idx]
        max_resp_y = resp_y[max_resp_len_idx]
        
        self.__orientation = cv.FastArctan(max_resp_y, max_resp_x)
    
    def __build_descriptor(self, integral_image):
        x, y, scale, _ = self.feature
        sigma = SIGMA[scale]
        g_sigma = 3.3*sigma
        size = SIZES[scale]
        x *= size
        y *= size
        L = int(0.5*sigma)
        theta = self.orientation*TO_RADS_RATIO
        cosT, sinT = np.cos(theta), np.sin(theta)
        transform_matrix = np.array([[ cosT, sinT],
                                     [-sinT, cosT]])
        samples_count = SURFDescriptor.STATISTICS_SAMPLES_COUNT       
        descriptor = np.zeros((64))
        k = 0
        x_indices = SURFDescriptor.STATISTICS_INDICES_X
        y_indices = SURFDescriptor.STATISTICS_INDICES_Y
#         x_shifts = 5*(SURFDescriptor.DESCRIPTOR_COMP_INDICES_X - 2)
#         y_shifts = 5*(SURFDescriptor.DESCRIPTOR_COMP_INDICES_Y - 2)        
        for (i, j) in ((i, j) for i in range(0, 4) for j in range(0, 4)):            
            x_shift, y_shift = 5*(i-2), 5*(j-2)
            
            shifted_x_indices = x_indices + x_shift
            shifted_y_indices = y_indices + y_shift
            X = tools.nd_round(x + sigma*(cosT*shifted_x_indices - sinT*shifted_y_indices))
            Y = tools.nd_round(y + sigma*(sinT*shifted_x_indices + cosT*shifted_y_indices))
            gauss = gaussian(X-x, Y-y, g_sigma)
            resp_x = integral_image.nd_haar_x(L)[Y, X].reshape((1, samples_count))
            resp_y = integral_image.nd_haar_y(L)[Y, X].reshape((1, samples_count))
            rotated_responses = gauss*np.dot(transform_matrix, np.append(resp_x, resp_y, axis=0))
            sum_resp_x, sum_resp_y = np.sum(rotated_responses, axis=1)
            sum_abs_resp_x, sum_abs_resp_y = np.sum(np.abs(rotated_responses), axis=1)
            
            descriptor[k] = sum_resp_x
            descriptor[k+1] = sum_resp_y
            descriptor[k+2] = sum_abs_resp_x
            descriptor[k+3] = sum_abs_resp_y
            k += 4
        norm = np.linalg.norm(descriptor)
        self.__descriptor = descriptor/norm    
        
    def __str__(self):
        x, y, s, l_sign = self.feature
        return ('SURF descriptor:\n  feature coordinates: x={0}, y={1}\n  ' +
                'scale: {2}\n  laplacian sign: {3}').format(
            x, y, SCALES[s], '>0' if l_sign == 1 else '<0' if l_sign == -1 else '0'
        )