import logging
from logging import DEBUG

import numpy as np

import stitching.mathtools.points as point_transforms
import stitching.utils.listutils as listutils

DIM_X = DIM_Y = 3
MIN_NUMBER_OF_MATCHES = 4

logger = logging.getLogger('homographyLogger')

def dlt(match_list):
    """
    Calculates a homography using Direct Linear Transformation algorithm.
    
    match_list: a list of point matches
        [[(x1, y1), (x1', y1')], [(x2, y2), (x2', y2')], ...]
    """
    A = _build_matrix_A(match_list)
    _, _, vt = np.linalg.svd(A, full_matrices=True)
    H = vt[-1].reshape(DIM_X, DIM_Y)
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nDLT, homography:\n%s', H)
        
    return H

def normalized_dlt(match_list):
    """
    Calculates a homography using Direct Linear Transformation algorithm, normalizing
    image point sets independently.
    
    match_list: a list of point matches
        [[(x1, y1), (x1', y1')], [(x2, y2), (x2', y2')], ...]
    """
    if logger.isEnabledFor(DEBUG):
        logger.debug('Normalized DLT started: match list = %s', match_list)
    
    matches_count = len(match_list)
        
    points1, points2 = np.split(np.array(match_list), 2, axis=1)
    points1 = points1.reshape((matches_count, 2)).T
    points2 = points2.reshape((matches_count, 2)).T
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nPoints1 =\n%s\nPoints2 =\n%s', points1, points2)
        
    T1 = similarity_transform_matr(points1)
    T2 = similarity_transform_matr(points2)
    points1 = np.dot(T1, point_transforms.nd_to_homogeneous(points1))
    points2 = np.dot(T2, point_transforms.nd_to_homogeneous(points2))
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nPoints1 =\n%s\nPoints2 =\n%s', points1, points2)
    
    points1 = point_transforms.nd_to_cartesian(points1).T.reshape((matches_count, 1, 2))
    points2 = point_transforms.nd_to_cartesian(points2).T.reshape((matches_count, 1, 2))
    
    normalized_points = np.hstack((points1, points2))
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nNormalized points: %s', normalized_points)
        
    H = dlt(normalized_points)
    return np.dot(np.linalg.inv(T2), np.dot(H, T1))

def similarity_transform_matr(points):
    n, _ = points.shape   
    mean = np.sum(points, axis=1).reshape((2, 1))/n
    avg_dist = np.sum(np.sqrt(np.sum((points-mean)**2, axis=0)))/n
    s = n*np.sqrt(2.0)/avg_dist
    tx, ty = -s*mean
    return np.array([[  s, 0.0,  tx],
                     [0.0,   s,  ty],
                     [0.0, 0.0, 1.0]])

def _build_matrix_Ai(point1, point2):
    x1, y1 = point1
    x2, y2 = point2 
    return np.array([[0.0, 0.0, 0.0, -x1, -y1, -1.0,  y2*x1,  y2*y1,  y2],
                     [ x1,  y1, 1.0, 0.0, 0.0,  0.0, -x2*x1, -x2*y1, -x2]])

def _build_matrix_A(matches):
    matches_count = len(matches)
    if matches_count >= MIN_NUMBER_OF_MATCHES:
        result = np.zeros((2*matches_count, 9))
        i = 0
        for point1, point2 in matches:
            Ai = _build_matrix_Ai(point1, point2)
            result[2*i:2*i+2,:] = Ai
            i += 1
        return result
    else:
        return None