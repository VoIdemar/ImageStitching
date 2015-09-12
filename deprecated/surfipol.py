import itertools
import numpy as np
import cv

from stitching.surf.constants import *
from stitching.cvtools.integral import IntegralImage
from stitching.surf import tools as mt
from stitching.cvtools import imgtools
from stitching.surf.errors import NoFeaturesError

class Surf(object):
    
    OCTAVE_MAP = {9: 1, 15: 1, 21: 1, 27: 2, 39: 2, 51: 3, 75: 3, 99: 4, 147: 4, 195: 5, 291: 5, 387: 5}    
    SCALES = sorted(OCTAVE_MAP.keys())
    FILTER_MAP = np.array([[0, 1, 2, 3], 
                           [1, 3, 4, 5], 
                           [3, 5, 6, 7], 
                           [5, 7, 8, 9], 
                           [7, 9, 10, 11]])
    OCTAVES = sorted(OCTAVE_MAP.values())
    N = 1
    REFINEMENT_ITERATION_COUNT = 5
    HESSIAN_RELATIVE_WEIGHT = 0.912
    STARTING_ANGLE = 30
    LAST_ANGLE = 330
    
    def __init__(self, filename, thresh=10):
        self.__fullname = filename
        self.__filename = filename[filename.rfind('\\') + 1:]
        self.__thresh = thresh
        self.__image = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_UNCHANGED)
        self.__grayscale = imgtools.normalize_image(cv.LoadImage(filename,
                                                    cv.CV_LOAD_IMAGE_GRAYSCALE))
        #ext_grayscale = imgtools.image_symmetric_extension(self.__grayscale, max(Surf.SCALES), max(Surf.SCALES))
        self.__integralImage = IntegralImage(self.__grayscale)
#         max_scale = max(Surf.SCALES)
#         self.__integralImage.extend_symmetrically(max_scale, max_scale)
        self.__responses = []
        self.__features = []
        self.__traces = []
        self.__orientation = []
        self.__descriptors = []
    
    def draw_orientation(self):
        image = cv.CloneImage(self.__image)
        if self.__features is not None:
            for i in range(0, len(self.__features)):
                x, y, s, _ = self.__features[i]
                size = SIZES[s]
                r = size*4
                theta = self.__orientation[i]
                x, y = size*x, size*y
                X, Y = cv.Round(x + r*np.cos(theta)), cv.Round(y + r*np.sin(theta))
                cv.Circle(image, (x, y), r, cv.RGB(255, 255, 0), thickness=1) 
                cv.Line(image, (x, y), (X, Y), cv.RGB(255, 255, 0))
            return image
        else:
            raise NoFeaturesError('No features to draw')

    def save_image(self, filename):
        cv.SaveImage(filename, self.__image)
    
    def get_features(self):
        return self.__features[:]
    
    def get_orientation(self):
        return self.__orientation[:]
    
    def get_descriptors(self):
        return self.__descriptors[:]

    def get_image(self):
        return self.__image
    
    def extract_features(self):
        print 'Analyzing image', self.__fullname
        print '  Computing hessians...'
        self.__build_resp_layers()
        self.__detect_features()
        print '  Finding features...'
        self.__find_features()
        print '   ', len(self.__features), 'interest points found'
        self.__scale_space_refinement()    
        print '   ', len(self.__features), 'interest points after refinement'
        print '  Finding features\' orientation...'
        self.__find_orientation()
        print '  Building descriptors...'
        self.__build_descriptors()
        print 'Extraction success!' 
        self.__save_to_file(self.__fullname + '_points.txt')
        self.__grayscale = None
        self.__integralImage = None
        self.__responses = []        
        self.__traces = []
            
    def __build_resp_layers(self):    
        for s in Surf.SCALES:
            print '   Computing hessian in scale ' + str(s) + '...'
            octave = Surf.OCTAVE_MAP[s]
            scale = 2**(octave - 1)
            w, h = self.__grayscale.width / scale, self.__grayscale.height / scale
            hessian, trace = self.__build_response(w, h, s, scale)
            self.__responses.append(hessian)
            self.__traces.append(trace)
            cv.SaveImage('D:\\test\\hess\\hessian_' + str(s) + '_' + self.__filename + '.jpg', hessian)
            
    def __build_response(self, w, h, s, scale):
        L = s / 3
        scale_factor_dxx_dyy = 6*L*(2*L - 1)
        scale_factor_dxy = 4*L*L
        hessian = cv.CreateImage((w, h), cv.IPL_DEPTH_64F, 1)
        trace = cv.CreateImage((w, h), cv.IPL_DEPTH_8U, 1)
        for (x, y) in [(x, y) for x in range(0, w) for y in range(0, h)]:
            X = x * scale
            Y = y * scale
            dxx = self.__integralImage.dXX(X, Y, L) / scale_factor_dxx_dyy
            dyy = self.__integralImage.dYY(X, Y, L) / scale_factor_dxx_dyy
            dxy = self.__integralImage.dXY(X, Y, L) / scale_factor_dxy
            hessian[y, x] = dxx*dyy - (Surf.HESSIAN_RELATIVE_WEIGHT*dxy)**2
            trace[y, x] = 1 if (dxx + dyy) > 0 else 0
        return (hessian, trace)
        
    #Checks whether (x, y, s) is far enough from the borders        
    def __is_not_outlier(self, x, y, s):
        if s in range(1, 11):
            sigma = SIGMA[s]
            dist = cv.Round(10*cv.Sqrt(2)*sigma + 10)
            w, h = self.__responses[s].width, self.__responses[s].height
            return (x > dist and x < (w-dist) and 
                    y > dist and y < (h-dist))
        else:
            return False    
    
    def __detect_features(self):
        resp_layers = self.__responses
        for i in range(0, len(Surf.FILTER_MAP)):
            for j in range(1, 3):
                b = resp_layers[Surf.FILTER_MAP[i, j-1]]
                indM = Surf.FILTER_MAP[i, j]
                print '   Finding features of scale ' + str(Surf.SCALES[indM]) + '...'
                m = resp_layers[indM]
                t = resp_layers[Surf.FILTER_MAP[i, j+1]]
                for (x, y) in [(x, y) for x in range(1, m.width - 1) for y in range(1, m.height - 1)]:
                    if np.abs(m[y, x]) <= self.__thresh:
                        continue
                    is_max = True
                    for (dx, dy) in [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]:
                        if m[y, x] < max([b[y+dy, x+dx], m[y+dy, x+dx], t[y+dy, x+dx]]):
                            is_max = False
                            break
                    if is_max and self.__is_not_outlier(x, y, indM):
                        self.__features.append((x, y, indM, self.__traces[indM][y, x]))
        img = cv.CloneImage(self.__grayscale)
        from stitching.ui import cvdrawing
        cvdrawing.draw_features(img, self.__features, (1, 0, 0))
        cv.SaveImage('D:\\10.jpg', img)
        
    
    def __is_maximum(self, x, y, b, m, t):
        if np.abs(m[y, x]) <= self.__thresh:
            return False
        for (dx, dy) in [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]:
            if m[y, x] < max([b[y+dy, x+dx], m[y+dy, x+dx], t[y+dy, x+dx]]):
                return False
        return True
    
    #Getting features
    def __find_features(self):
        resp_layers = self.__responses
        for i in range(0, len(Surf.FILTER_MAP)):
            for j in range(1, 3):
                b = resp_layers[Surf.FILTER_MAP[i, j-1]]
                indM = Surf.FILTER_MAP[i, j]
                print '   Finding features of scale ' + str(Surf.SCALES[indM]) + '...'
                m = resp_layers[indM]
                t = resp_layers[Surf.FILTER_MAP[i, j+1]]
                octave_b = Surf.OCTAVES[Surf.FILTER_MAP[i, j-1]]
                scale_b = 2**(octave_b - 1)
                octave_m = Surf.OCTAVES[Surf.FILTER_MAP[i, j-1]]
                scale_m = 2**(octave_m - 1)
                octave_t = Surf.OCTAVES[Surf.FILTER_MAP[i, j+1]]
                scale_t = 2**(octave_t - 1)
                sf_b = scale_b / float(scale_m)
                sf_t = scale_t / float(scale_m)                 
                candidates = mt.nms_2d(m, m.width, m.height, Surf.N)
                for (k, n) in candidates:
                    if np.abs(m[k, n]) <= self.__thresh:
                        continue
                    failed = False
                    for (x, y) in itertools.product(range(k-1, k+2), range(n-1, n+2)):
                        if failed:
                            break
                        failed = b[int(x/sf_b), int(y/sf_b)] >= m[k, n] or t[int(x/sf_t), int(y/sf_t)] >= m[k, n]
                    if not failed and self.__is_not_outlier(n, k, indM):
                        self.__features.append((n, k, indM, self.__traces[indM][k, n]))
        
    def __scale_space_refinement(self):
        responses = self.__responses
        fake_maxima = []
        for i in range(0, len(self.__features)):
            (x, y, s, l_sign) = self.__features[i]
            k = Surf.REFINEMENT_ITERATION_COUNT
            failed = False
            while k >= 0 and self.__is_not_outlier(x, y, s) and not failed:
                k = k - 1 
                dx = mt.du_dx(responses, x, y, s)
                dy = mt.du_dy(responses, x, y, s)
                ds = mt.du_ds(responses, x, y, s)
                dxx = mt.du2_dx2(responses, x, y, s)
                dyy = mt.du2_dy2(responses, x, y, s)
                dxy = mt.du2_dxdy(responses, x, y, s)
                dss = mt.du2_ds2(responses, x, y, s)
                dxds = mt.du2_dxds(responses, x, y, s)
                dyds = mt.du2_dyds(responses, x, y, s)
                detH = (2*dxds*dxy*dyds - (dxds**2)*dyy - (dxy**2)*dss + 
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
                    if (np.abs(delta_x) < 1 and np.abs(delta_y) < 1 and 
                        np.abs(delta_s) < 0.4*(2**Surf.OCTAVES[s])):
                        break
                    else:
                        newX = x + delta_x + 0.5
                        newY = y + delta_y + 0.5
                        newS = int(s + delta_s/(0.4*(2**Surf.OCTAVES[s])) + 0.5)
                        if not (newS in range(0, 11)):
                            failed = True
                            break
                        oldSize, newSize = SIZES[s], SIZES[newS]
                        scale = newSize / float(oldSize)
                        if newSize > oldSize:
                            newX, newY = newX / scale, newY / scale
                        elif newSize < oldSize:
                            newX, newY = newX * scale, newY * scale 
                        (x, y, s) = (int(newX), int(newY), newS)
                else:
                    break
            if (failed or k == -1 or not self.__is_not_outlier(x, y, s) or 
                np.abs(x - self.__features[i][0]) > Surf.REFINEMENT_ITERATION_COUNT or 
                np.abs(y - self.__features[i][1]) > Surf.REFINEMENT_ITERATION_COUNT):
                fake_maxima.append(self.__features[i])
            else:
                self.__features[i] = (x, y, s, l_sign)
        self.__features = mt.list_subtraction(self.__features, fake_maxima)
        
    def __find_orientation(self):
        for (x, y, s, _) in self.__features:
            sigma = SIGMA[s]
            l = int(sigma)
            haar_respX = {}
            haar_respY = {}
            angle = {}
            indexes = set(itertools.product(range(-6, 7), range(-6, 7)))
            for (i, j) in [elem for elem in indexes if (elem[0]**2 + elem[1]**2) <= 36]:
                X, Y = cv.Round(x + i * sigma), cv.Round(y + j * sigma)
                gauss = mt.gaussian(X-x, Y-y, 2*sigma)
                haar_respX[(i, j)] = self.__integralImage.haarX(X, Y, l) * gauss
                haar_respY[(i, j)] = self.__integralImage.haarY(X, Y, l) * gauss
                angle[(i, j)] = cv.FastArctan(haar_respY[(i, j)], haar_respX[(i, j)])
            max_length = 0
            max_resp_x = max_resp_y = 0
            for theta in range(Surf.STARTING_ANGLE, Surf.LAST_ANGLE):
                resp_x = resp_y = 0
                for p in [index for index in angle.keys() if (theta - 30 <= angle[index] and 
                                                              angle[index] <= theta + 30)]:
                    resp_x += haar_respX[p]
                    resp_y += haar_respY[p]
                resp_len = resp_x**2 + resp_y**2
                if resp_len > max_length:
                    max_resp_x, max_resp_y = resp_x, resp_y
                    max_length = resp_len
            self.__orientation.append(cv.FastArctan(max_resp_y, max_resp_x) * mt.TO_RADS_RATIO)        
    
    def __build_descriptors(self):
        for k in range(0, len(self.__features)):
            (x, y, s, _) = self.__features[k]
            sigma = SIGMA[s]
            l = int(0.5*sigma)
            theta = self.__orientation[k]
            cosT, sinT = np.cos(theta), np.sin(theta)
            descriptor = np.zeros((64))
            k = 0
            for (i, j) in itertools.product(range(0, 4), range(0, 4)):
                sum_resp_x = sum_resp_y = sum_abs_resp_x = sum_abs_resp_y = 0
                for (m, n) in itertools.product(range(0, 5), range(0, 5)):
                    X = cv.Round(x + sigma*(cosT*(5*(i-2) + m) - sinT*(5*(j-2) + n)))
                    Y = cv.Round(y + sigma*(sinT*(5*(i-2) + m) + cosT*(5*(j-2) + n)))
                    gauss = mt.gaussian(X-x, Y-y, 3.3*sigma)
                    resp_x = self.__integralImage.haarX(X, Y, l)
                    resp_y = self.__integralImage.haarY(X, Y, l)
                    respX = gauss*(-sinT*resp_x + cosT*resp_y)
                    respY = gauss*(cosT*resp_x + sinT*resp_y)
                    sum_resp_x += respX
                    sum_resp_y += respY
                    sum_abs_resp_x += np.abs(respX)
                    sum_abs_resp_y += np.abs(respY)
                descriptor[k] = sum_resp_x
                descriptor[k+1] = sum_resp_y
                descriptor[k+2] = sum_abs_resp_x
                descriptor[k+3] = sum_abs_resp_y
                k = k + 4
            norm = np.linalg.norm(descriptor)
            self.__descriptors.append(descriptor / norm)   
            
    def __save_to_file(self, filename):
        output = open(filename, 'w')
        for point in self.__features:
            output.write(str(point) + '\n')
        output.flush()
        output.close()  