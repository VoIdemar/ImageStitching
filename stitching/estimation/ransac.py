import logging
from logging import DEBUG, INFO

import random
import math
import numpy as np

from stitching.transformations.homography import normalized_dlt
import stitching.mathtools.points as pt

logger = logging.getLogger('ransacLogger')

class RansacHomographyEstimator(object):
    
    MIN_STD = 10**5
    P = 0.99
    CORR_COUNT = 4
    INITIAL_NUMBER_OF_SAMPLES = 500
    DIST_THRESHOLD = 2
    ADAPTIVE_THRESH_ENABLED = False
    THRESH_CHANGE_STEP = 0.25
    THRESH_CHANGE_FREQ = 100
    
    def __init__(self, feature_matches, dist_threshold=DIST_THRESHOLD, 
                 adaptive_thresh_enabled=ADAPTIVE_THRESH_ENABLED,
                 thresh_change_step=THRESH_CHANGE_STEP,
                 thresh_change_freq=THRESH_CHANGE_FREQ):
        self.__initial_matches = feature_matches
        self.__dist_threshold = dist_threshold
        self.__matches_count = len(feature_matches)
        self.__indices = range(self.__matches_count)
        self.__adaptive_thresh_enabled = adaptive_thresh_enabled
        self.__thresh_change_step = thresh_change_step
        self.__thresh_change_freq = thresh_change_freq
        self.__H = None
        self.__inliers = []
        
    @staticmethod
    def create_for_matches(feature_matches):
        return RansacHomographyEstimator(feature_matches)
    
    @staticmethod
    def create_for_matcher(descriptor_matcher):
        return RansacHomographyEstimator(descriptor_matcher.get_feature_matches())
    
    def estimate(self):
        if logger.isEnabledFor(INFO):
            logger.info('=================================================================')
            logger.info("RANSAC estimation started, matches: %s, threshold: %s", 
                        self.__initial_matches, self.dist_threshold)
            
        matches = self.__initial_matches
        get_sample_points = self.__get_noncollinear_sample_points

        max_inlier_count = curr_std = 0
        threshold = self.dist_threshold
        matches_count = self.matches_count
        float_matches_count = float(matches_count)
        inliers = None
        best_h = None
        
        nd_matches = np.array(matches)
        points1, points2 = np.split(nd_matches, 2, axis=1)
        points1 = points1.reshape((matches_count, 2)).T
        points2 = points2.reshape((matches_count, 2)).T
        hg_points1 = pt.nd_to_homogeneous(points1)
        hg_points2 = pt.nd_to_homogeneous(points2)
        
        adaptive_thresh_enabled = self.adaptive_threshold_enabled
        thresh_change_freq = self.threshold_change_frequency
        thresh_change_step = self.threshold_change_step
        
        N = RansacHomographyEstimator.INITIAL_NUMBER_OF_SAMPLES
        p = RansacHomographyEstimator.P
        i = 1
        
        while i <= N:
            i = i + 1
            
            if adaptive_thresh_enabled and (i % thresh_change_freq == 0):
                threshold += thresh_change_step
             
            match_list = get_sample_points()         
            curr_h = normalized_dlt(match_list)
            inv_h = np.linalg.inv(curr_h)
            
            projected_points1 = pt.nd_to_cartesian(np.dot(curr_h, hg_points1))
            projected_points2 = pt.nd_to_cartesian(np.dot(inv_h, hg_points2))
            
            distances1 = pt.get_distances(projected_points1, points2)
            distances2 = pt.get_distances(projected_points2, points1)
            distances = distances1 + distances2
            
            inlier_indices = np.where(distances < threshold)[0]
            inlier_distances = distances[inlier_indices]
            curr_std = np.std(inlier_distances)            
            inlier_count = inlier_indices.shape[0]
            
            if (logger.isEnabledFor(DEBUG)):
                logger.debug('inlier count: {0}, max inlier count: {1}, matches_count = {2}'.format(
                            inlier_count, max_inlier_count, matches_count))
                
            if (inlier_count > max_inlier_count or 
                    (inlier_count == max_inlier_count and 
                     curr_std < RansacHomographyEstimator.MIN_STD)):
                max_inlier_count = inlier_count
                best_h = curr_h                
                inliers = [matches[i] for i in list(inlier_indices)]
            
            eps = 1.0 - inlier_count/float_matches_count
            if eps == 1.0:
                N = N + 1
                continue
            N = int(math.log(1.0-p)/math.log(1.0 - (1.0-eps)**4))                
            
            if (logger.isEnabledFor(DEBUG)):
                logger.debug('i = {0}, N = {1}'.format(i, N))
        # End while    
        self.__H = best_h
        self.__inliers = inliers
        
        if logger.isEnabledFor(INFO):
            logger.info('RANSAC estimation finished successfully, number of inliers: %s, final threshold: %s', 
                         len(self.inliers), threshold)
            logger.info('=================================================================')
    
    @property
    def homography(self):
        return self.__H
    
    @property
    def inliers(self):
        return self.__inliers
    
    @property
    def matches_count(self):
        return self.__matches_count
    
    @property
    def dist_threshold(self):
        return self.__dist_threshold
        
    @dist_threshold.setter
    def dist_threshold(self, value):
        self.__dist_threshold = value
    
    @property
    def adaptive_threshold_enabled(self):
        return self.__adaptive_thresh_enabled
        
    @adaptive_threshold_enabled.setter
    def adaptive_threshold_enabled(self, value):
        self.__adaptive_thresh_enabled = value
        
    @property
    def threshold_change_step(self):
        return self.__thresh_change_step
        
    @threshold_change_step.setter
    def threshold_change_step(self, value):
        self.__thresh_change_step = value
    
    @property
    def threshold_change_frequency(self):
        return self.__thresh_change_freq
        
    @threshold_change_frequency.setter
    def threshold_change_frequency(self, value):
        self.__thresh_change_freq = value    
    
    def __get_noncollinear_sample_points(self):
        matches = self.__initial_matches
        img1_points = [(0, 0), (0, 0), (0, 0)]
        img2_points = [(0, 0), (0, 0), (0, 0)]
        match_list_idx = []
        indices = self.__indices
        while (not pt.are_noncollinear(img1_points) or
               not pt.are_noncollinear(img2_points)):
            match_list_idx = random.sample(indices, RansacHomographyEstimator.CORR_COUNT)
            img1_points, img2_points = [], []
            for idx in match_list_idx:                
                ((x1, y1), (x2, y2)) = matches[idx]
                img1_points.append( (x1, y1) )
                img2_points.append( (x2, y2) )
        return [matches[i] for i in match_list_idx]