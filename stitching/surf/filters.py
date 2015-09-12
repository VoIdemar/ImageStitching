import numpy as np
import cv

class FastHessianType:
    DXX = 1
    DYY = 2
    DXY = 3

class HaarWvType:
    RESP_X = 1
    RESP_Y = 2

class Filters(object):
    # Fast-Hessian filters (3 types)
    @staticmethod   
    def fast_hessian(dim, filterType):
        #kernel = cv.CreateMat(dim, dim, cv.CV_32SC1)
        kernel = np.zeros((dim, dim))
        l0 = dim / 3
        l1 = (dim-l0) / 2
        if filterType == FastHessianType.DXX:
            # Grey areas (0)
            c = range(0, l1) + range(dim-l1, dim)
            for i in c:
                for j in xrange(0, dim):
                    kernel[i][j] = 0
            # White and black areas (1 & -2)
            for i in xrange(l1, dim-l1):    
                c = range(0, l0) + range(dim-l0, dim)
                for j in c:
                    kernel[i][j] = 1
                for j in xrange(l0, dim-l0):
                    kernel[i][j] = -2  
        elif filterType == FastHessianType.DYY:
            # Grey areas (0)     
            c = range(0, l1) + range(dim-l1, dim)
            for i in xrange(0, dim):
                for j in c:
                    kernel[i][j] = 0                 
            # White and black areas (1 & -2)
            for j in xrange(l1, dim-l1):
                c = range(0, l0) + range(dim-l0, dim)
                for i in c:
                    kernel[i][j] = 1
                for i in xrange(l0, dim-l0):
                    kernel[i][j] = -2
        elif filterType == FastHessianType.DXY:
            # Grey areas (0)
            l2 = (l1-1) / 2
            c = range(0, l2) + range(l0+l2, l0+l2+1) + range(dim-l2, dim)            
            for i in c:
                for j in xrange(0, dim):
                    kernel[i][j] = 0     
            a1 = range(l2, l0+l2)
            a2 = range(dim-l0-l2, dim-l2)      
            c1 = a1 + a2    
            for j in c:
                for i in c1:
                    kernel[i][j] = 0
            # White areas (1)
            for i in a1:
                for j in a1:
                    kernel[i][j] = 1    
            for i in a2:
                for j in a2:
                    kernel[i][j] = 1
            # Black areas (-1)    
            for i in a1:
                for j in a2:
                    kernel[i][j] = -1
            for i in a2:
                for j in a1:
                    kernel[i][j] = -1
        return cv.fromarray(kernel)
    
    # Haar filters (2 types)
    @staticmethod
    def haar_wavelet(dim, wvType):
        kernel = cv.CreateMat(dim, dim, cv.CV_32FC1)
        if wvType == HaarWvType.RESP_X:
            for i in xrange(0, dim):    
                for j in xrange(0, dim/2):
                    kernel[i, j] = -1
                for j in xrange(dim/2, dim):
                    kernel[i, j] = 1
        elif wvType == HaarWvType.RESP_Y:
            for j in xrange(0, dim):
                for i in xrange(0, dim/2):
                    kernel[i, j] = -1
                for i in xrange(dim/2, dim):
                    kernel[i, j] = 1
        return kernel