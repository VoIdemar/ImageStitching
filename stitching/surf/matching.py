import logging
import numpy as np

from stitching.surf.constants import SIZES

logger = logging.getLogger('matcherLogger')

class DescriptorMatcher(object):
    
    MATCHER_RATIO = 0.8
    DIST_THRESHOLD = 10E-3
        
    def __init__(self, descriptor_set1, descriptor_set2):
        self.__descriptor_set1 = descriptor_set1
        self.__descriptor_set2 = descriptor_set2
        self.__matches = []
        self.__feature_matches = None
        
        self.__preparative_sorting()           

    def match(self):
        """
        Matches SURF descriptors of the specified images features.
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(('\nMatching started. Number of descriptors1: {0}, ' +
                         'number of descriptors2: {1}').format(
                 len(self.descriptor_set1), len(self.descriptor_set2)))
        
        matches = []
        matcher_ratio = DescriptorMatcher.MATCHER_RATIO
        dist_threshold = DescriptorMatcher.DIST_THRESHOLD
        
        descriptors1 = self.descriptor_set1
        sorted_descr2 = self.__sorted_descr2
        sorted_descr2_data = self.__sorted_descr2_data        
        for i in range(len(descriptors1)):
            target_descr = descriptors1[i]
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('\nMatching descriptors for descriptor ' + str(target_descr))
            
            lapl_sign = target_descr.laplacian_sign
            td_data = target_descr.data
            # Get descriptors of the second image features with the same laplacian sign value
            cd_data = sorted_descr2_data[lapl_sign]
            # Calculate euclidean distances between target descriptor and descriptors 
            # of the second image features
            distances = np.sqrt(np.sum((cd_data-td_data)**2, axis=1))
            # Get indices of two nearest descriptors
            sorted_dist_indices = np.argsort(distances)
            min_idx, prev_min_idx = sorted_dist_indices[:2]
            # Get two smallest distances
            min_dist, previous_dist = distances[[min_idx, prev_min_idx]]
            if (previous_dist <= dist_threshold) or ((min_dist/previous_dist) < matcher_ratio):
                matches.append( (i, sorted_descr2[lapl_sign][min_idx]) )
#             elif to_check_count == 1:
#                 distance = np.sqrt(np.sum((cd_data[0]-td_data)**2))
#                 if distance <= dist_threshold:
#                     matches.append( (i, sorted_descr2[lapl_sign][0]) )
        self.__matches = matches
        self.__feature_matches = None
    
    @staticmethod
    def sort_descriptors(descriptor_set):
        """
        Returns descriptors of the image features sorted by the values or their laplacian signs.
        """
        lapl_sign_to_descr_idx = { 1:[], 
                                  -1:[], 
                                   0:[] }
        lapl_sign_to_descr = { 1:[],
                              -1:[],
                               0:[] }
        
        for i in range(len(descriptor_set)):
            descriptor = descriptor_set[i]
            lapl_sign = descriptor.laplacian_sign
            lapl_sign_to_descr_idx[lapl_sign].append(i)
            lapl_sign_to_descr[lapl_sign].append(descriptor.data)
            
        ignored_lapl_signs = []
        for key in lapl_sign_to_descr.keys():
            descriptor_count = len(lapl_sign_to_descr[key])
            if descriptor_count == 0:
                lapl_sign_to_descr.pop(key)
                lapl_sign_to_descr_idx.pop(key)
                ignored_lapl_signs.append(key)
            else:
                lapl_sign_to_descr[key] = np.array(lapl_sign_to_descr[key])
        
        return (lapl_sign_to_descr, lapl_sign_to_descr_idx, ignored_lapl_signs)
    
    @property
    def descriptor_set1(self):
        return self.__descriptor_set1
    
    @property
    def descriptor_set2(self):
        return self.__descriptor_set2

    @property
    def matches(self):
        return self.__matches
    
    def get_feature_match(self, idx):
        if self.__feature_matches is None:
            idx1, idx2 = self.matches[idx]
            return self.__get_feature_match(idx1, idx2)
        else:
            return self.__feature_matches[idx]
    
    def get_feature_matches(self):
        if self.__feature_matches is None:
            self.__feature_matches = [self.__get_feature_match(i1, i2) for (i1, i2) in self.matches]          
        return self.__feature_matches
    
    def get_descriptor_match(self, idx):
        idx1, idx2 = self.matches[idx]
        return [self.descriptor_set1[idx1], self.descriptor_set2[idx2]]
    
    def get_descriptor_matches(self):
        return [(self.descriptor_set1[i1], self.descriptor_set2[i2]) for (i1, i2) in self.matches]
    
    def __get_feature_match(self, i1, i2):
        d1, d2 = self.descriptor_set1[i1], self.descriptor_set2[i2]
        x1, y1 = DescriptorMatcher.__scale_feature(d1.feature)
        x2, y2 = DescriptorMatcher.__scale_feature(d2.feature)
        return [(x1, y1), (x2, y2)]
    
    @staticmethod
    def __scale_feature(feature):
        x, y, s, _ = feature
        scale = SIZES[s]
        return (x*scale, y*scale)
    
    def __preparative_sorting(self):
        sorted_info = DescriptorMatcher.sort_descriptors(self.descriptor_set2)
        sorted_descr2_data, sorted_descr2, ignored_lapl_signs = sorted_info
        self.__sorted_descr2 = sorted_descr2
        self.__sorted_descr2_data = sorted_descr2_data
        self.__descriptor_set1 = [descr for descr in self.descriptor_set1 
                                  if not (descr.laplacian_sign in ignored_lapl_signs)]