import itertools
import numpy as np

from stitching.utils.listutils import intersection

def nms_2d(array, W, H, n):
    """
    2D non-maximum suppression.
    """
    result = []
    pointSet = itertools.product(range(0, H-n), range(0, W-n))
    maxLength = np.max([W-n, H-n])
    nSet = np.arange(n, maxLength, n+1)
    n2Set = itertools.product(nSet, nSet)
    pixelSet = intersection(n2Set, pointSet)
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

def save_descriptors_to_file(descriptors, filename):
    fout = None
    try:
        fout = open(filename, 'w')
        for d in descriptors:        
            fout.write(str(d) + '\n\n')
        fout.flush()
    finally:
        try:
            if not fout is None:
                fout.close()
        finally:
            pass
    
def get_twodim_indices(*dims):
    """
    Calculates two arrays containing evenly spaced integer values within specified
    dimensions.
    
    Expects input in the following format:
        [start, ] stop
    """
    dim = 0
    start = 0    
    if len(dims) == 2:
        start, stop = dims
        dim = stop - start
    else:
        stop = dims[0]
        dim = stop
    zeros = np.zeros((dim, dim), dtype=np.int)
    indices = np.arange(start, stop)
    twodim_indices = zeros + indices
    shape = dim*dim
    return (twodim_indices.reshape(shape), 
            twodim_indices.T.reshape(shape))
    
def get_orientation_indices():
    x_indices, y_indices = get_twodim_indices(-6, 7)
    idx_mask = (x_indices**2 + y_indices**2) <= 36
    return (x_indices[idx_mask], y_indices[idx_mask])

def get_vectorized_angles(idx_count, test_angles):
    angles_count = test_angles.shape[0]
    vectorized_angles = np.zeros((angles_count, idx_count)) + test_angles.reshape((angles_count, 1))
    return vectorized_angles

def extend_dim(arr, extension_size):
    size = arr.shape[0]
    return np.zeros((extension_size, size)) + arr
    
def nd_round(arr):
    return np.round(arr).astype(np.int32, copy=False)