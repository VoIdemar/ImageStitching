from stitching.surf.constants import SIGMA, SIZES, OUTLIER_DIST_THRESHOLD
from stitching.utils.listutils import subtraction
from stitching.surf.constants import SCALE_FACTORS, OCTAVES

class ScaleSpaceRefiner(object):
    
    NEIGHBOURS = [[0,  0,  0],
                  [0,  1,  2], 
                  [1,  2,  3], 
                  [1,  3,  4], 
                  [3,  4,  5], 
                  [3,  5,  6], 
                  [5,  6,  7], 
                  [5,  7,  8], 
                  [7,  8,  9], 
                  [7,  9, 10], 
                  [9, 10, 11]]
    
    REFINEMENT_ITERATION_COUNT = 5
    
    def __init__(self, features, responses):
        self.__features = features
        self.__responses = responses
        self.__refined_features = []
        
    @property
    def features(self):
        return self.__features
    
    @property
    def responses(self):
        return self.__responses
    
    @property
    def refined_features(self):
        return self.__refined_features
    
    def du_dx(self, x, y, s):
        h = self.responses[s]        
        return 0.5*(h[y, x+1] - h[y, x-1])
        
    def du_dy(self, x, y, s):
        h = self.responses[s]
        return 0.5*(h[y+1, x] - h[y-1, x])
    
    def du_ds(self, x, y, s):
        bottom, middle, top, by, bx, tx, ty, h = self.__get_variables(x, y, s)
        return (h[top][ty, tx] - h[bottom][by, bx])/(2.0*(SIGMA[top] - SIGMA[middle]))
    
    def du2_dx2(self, x, y, s):
        h = self.responses[s]
        return h[y, x+1] - 2.0*h[y, x] + h[y, x-1]
    
    def du2_dy2(self, x, y, s):
        h = self.responses[s]
        return h[y-1, x] - 2.0*h[y, x] + h[y+1, x]
    
    def du2_ds2(self, x, y, s):
        bottom, middle, top, by, bx, tx, ty, h = self.__get_variables(x, y, s)
        return (h[top][ty, tx] - 2.0*h[middle][y, x] + h[bottom][by, bx])/((SIGMA[top] - SIGMA[middle])**2)
    
    def du2_dxdy(self, x, y, s):
        h = self.responses[s]
        return 0.25*(h[y+1, x+1] + h[y-1, x-1] - h[y+1, x-1] - h[y-1, x+1])
    
    def du2_dxds(self, x, y, s):
        bottom, middle, top, by, bx, tx, ty, h = self.__get_variables(x, y, s)
        return 0.25*(h[top][ty, tx + 1] - h[top][ty, tx - 1] - 
                     h[bottom][by, bx + 1] + h[bottom][by, bx - 1])/(SIGMA[top] - SIGMA[middle])
    
    def du2_dyds(self, x, y, s):
        bottom, middle, top, by, bx, tx, ty, h = self.__get_variables(x, y, s)
        return 0.25*(h[top][ty + 1, tx] - h[top][ty - 1, tx] - 
                     h[bottom][by + 1, bx] + h[bottom][by - 1, bx])/(SIGMA[top] - SIGMA[middle])
                     
    def refine(self):    
        features = self.__features
        is_not_outlier = self.__is_not_outlier
    
        fake_maxima = []
    
        for i in range(len(features)):
            (x, y, s, l_sign) = features[i]
            k = ScaleSpaceRefiner.REFINEMENT_ITERATION_COUNT
            failed = False
            while k >= 0 and is_not_outlier(x, y, s) and not failed:
                k = k - 1 
                dx = self.du_dx(x, y, s)
                dy = self.du_dy(x, y, s)
                ds = self.du_ds(x, y, s)
                dxx = self.du2_dx2(x, y, s)
                dyy = self.du2_dy2(x, y, s)
                dxy = self.du2_dxdy(x, y, s)
                dss = self.du2_ds2(x, y, s)
                dxds = self.du2_dxds(x, y, s)
                dyds = self.du2_dyds(x, y, s)
                detH = float(2*dxds*dxy*dyds - (dxds**2)*dyy - (dxy**2)*dss + 
                             dxx*dyy*dss - dxx*(dyds**2))
                if detH <> 0:
                    delta_x = -(dx*(dyy*dss - dyds**2) + 
                                dy*(dxds*dyds - dss*dxy) + 
                                ds*(dxy*dyds - dxds*dyy)) / detH
                    delta_y = -(dx*(dxy*dss - dxds*dyds) + 
                                dy*(dxds**2 - dxx*dss) + 
                                ds*(dxx*dyds - dxds*dxy)) / detH
                    delta_s = -(dx*(dxy*dyds - dxds*dyy) + 
                                dy*(dxds*dxy - dxx*dyds) + 
                                ds*(dxx*dyy - dxy**2)) / detH
                    if (abs(delta_x) < 1 and abs(delta_y) < 1 and 
                        abs(delta_s) < 0.4*(2**OCTAVES[s])):
                        break
                    else:
                        newX = x + delta_x + 0.5
                        newY = y + delta_y + 0.5
                        newS = int(s + delta_s/(0.4*(2**OCTAVES[s])) + 0.5)
                        if not (newS in range(0, 11)):
                            failed = True
                            break
                        oldSize, newSize = SIZES[s], SIZES[newS]
                        scale = newSize/float(oldSize)
                        if newSize > oldSize:
                            newX, newY = newX/scale, newY/scale
                        elif newSize < oldSize:
                            newX, newY = newX*scale, newY*scale 
                        (x, y, s) = (int(newX), int(newY), newS)
                else:
                    break
            if (failed or k == -1 or not is_not_outlier(x, y, s) or 
                abs(x - features[i][0]) > ScaleSpaceRefiner.REFINEMENT_ITERATION_COUNT or 
                abs(y - features[i][1]) > ScaleSpaceRefiner.REFINEMENT_ITERATION_COUNT):
                fake_maxima.append(features[i])
            else:
                features[i] = (x, y, s, l_sign)
        self.__features = features
        self.__refined_features = subtraction(features, fake_maxima)
        
    def release_data(self):
        self.__features = None
        self.__responses = None
        self.__refined_features = None
        
    def __is_not_outlier(self, x, y, s):
        """ 
        Checks whether (x, y, s) is far enough from the borders.
        """
        if s in range(1, 11):
            dist = OUTLIER_DIST_THRESHOLD[s]
            h, w = self.__responses[s].shape
            return (dist < x < (w-dist) and 
                    dist < y < (h-dist))
        else:
            return False  
    
    def __get_variables(self, x, y, s):
        bottom, middle, top = ScaleSpaceRefiner.NEIGHBOURS[s]
        sf_b, _, sf_t = SCALE_FACTORS[s]
        ty, tx = int(y*sf_t), int(x*sf_t)
        by, bx = int(y*sf_b), int(x*sf_b)
        h = self.responses
        return (bottom, middle, top, by, bx, tx, ty, h)