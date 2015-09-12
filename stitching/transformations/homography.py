import logging
import math
from logging import DEBUG

import numpy as np

import stitching.mathtools.points as pt
import stitching.utils.listutils as lu

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
        
    points1, points2 = lu.unzip(match_list)
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nPoints1 =\n%s\nPoints2 =\n%s', points1, points2)
        
    T1 = similarity_transform_matr(points1)
    T2 = similarity_transform_matr(points2)
    hg_points1 = pt.list_to_homogeneous(points1)
    hg_points2 = pt.list_to_homogeneous(points2)
    hg_points1 = np.dot(T1, np.array(hg_points1).T).T
    hg_points2 = np.dot(T2, np.array(hg_points2).T).T
    
    points1 = pt.ndarray2d_to_list(hg_points1)
    points2 = pt.ndarray2d_to_list(hg_points2)
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nPoints1 =\n%s\nPoints2 =\n%s', points1, points2)
    
    points1 = pt.list_to_cartesian(points1)
    points2 = pt.list_to_cartesian(points2)
    
    normalized_points = zip(points1, points2)
    
    if logger.isEnabledFor(DEBUG):
        logger.debug('\nNormalized points: %s', normalized_points)
        
    H = dlt(normalized_points)
    return np.dot(np.linalg.inv(T2), np.dot(H, T1))

def similarity_transform_matr(points):
    f_points_count = float(len(points))
    xi, yi = lu.unzip(points)
    mean_x = lu.mean(xi)
    mean_y = lu.mean(yi)
    distances = map(lambda (x, y): math.sqrt((x-mean_x)**2 + (y-mean_y)**2), 
                    zip(xi, yi))
    avg_dist = sum(distances)/f_points_count
    s = f_points_count*math.sqrt(2.0)/avg_dist
    tx, ty = -s*mean_x, -s*mean_y
    return np.array([[  s, 0.0,  tx],
                     [0.0,   s,  ty],
                     [0.0, 0.0, 1.0]])

def _build_matrix_A(matches):
    matches_count = len(matches)
    if matches_count >= MIN_NUMBER_OF_MATCHES:
        result = []
        for point1, point2 in matches:
            Ai = _build_matrix_Ai(point1, point2)
            result = result + Ai            
        return np.array(result)
    else:
        return None

def _build_matrix_Ai(point1, point2):
    x1, y1 = point1
    x2, y2 = point2 
    return [[0.0, 0.0, 0.0, -x1, -y1, -1.0,  y2*x1,  y2*y1,  y2],
            [ x1,  y1, 1.0, 0.0, 0.0,  0.0, -x2*x1, -x2*y1, -x2]]