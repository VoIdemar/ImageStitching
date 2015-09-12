import numpy as np

COLLINEARITY_PRECISION = 0.01

def to_homogeneous(point):    
    x, y = point
    return [x, y, 1.0]

def to_cartesian(homogeneous_point):
    wx, wy, w = homogeneous_point
    return [wx/float(w), wy/float(w)]

def list_to_homogeneous(points):
    return map(to_homogeneous, points)

def list_to_cartesian(points):
    return map(to_cartesian, points)

def nd_to_homogeneous(points):
    """
    Transforms an array of cartesian points to an array of points with
    homogeneous coordinates.
    
    Parameters
    ----------
    points : ndarray[[x1, x2, x3, ...],
                     [y1, y2, y3, ...]]
    """
    _, points_count = points.shape
    hg_ratios = np.ones((1, points_count))
    return np.append(points, hg_ratios, axis=0)

def nd_to_cartesian(hg_points):
    """
    Transforms an array of points with homogeneous coordinates to 
    an array of cartesian points.
    
    Parameters
    ----------
    hg_points : ndarray[[w1*x1, w2*x2, w3*x3, ...],
                        [w1*y1, w2*y2, w3*y3, ...],
                        [   w1,    w2,    w3, ...]]
    """
    return hg_points[0:2,:]/hg_points[2,:]

def get_distances(points1, points2):
    """
    Calculates euclidean distances between specified point arrays point-wise.
    """
    return np.sqrt(np.sum((points1-points2)**2, axis=0))

def ndarray2d_to_list(ndarray2d):
    return map(list, ndarray2d)
    
def are_noncollinear(points):
    n = len(points)
    for i in range(n - 1):
        for j in range(i + 1, n):
            (x1, y1), (x2, y2) = points[i], points[j]
            for k in range(j + 1, n):
                (x, y) = points[k]
                if abs((y1 - y2)*x + (x2 - x1)*y + (x1*y2 - x2*y1)) < COLLINEARITY_PRECISION:
                    return False
    return True