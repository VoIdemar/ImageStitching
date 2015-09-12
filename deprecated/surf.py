import itertools
import numpy as np
import cv

from filters import Filters, FastHessianType
from mathtools import MathTools as mt
from errors import NoFeaturesError

class Surf:
	
	DEFAULT_THRESHOLD = 0.04
	SCALES = [9, 15, 21, 27, 39, 51, 75, 99, 147, 195, 291, 387]
	FILTER_MAP = np.array([[0, 1, 2, 3], 
						   [1, 3, 4, 5], 
						   [3, 5, 6, 7], 
						   [5, 7, 8, 9], 
						   [7, 9, 10, 11]])
	OCTAVES = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
	N = 1
	REFINEMENT_ITERATION_COUNT = 5
	HESSIAN_RELATIVE_WEIGHT = 0.912
	STARTING_ANGLE = 30
	LAST_ANGLE = 330
	
	def __init__(self, filename, thresh):
		self.__filename = filename
		self.__thresh = thresh
		self.__image = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_UNCHANGED)
		gray = mt.normalize_image(cv.LoadImage(filename,
											   cv.CV_LOAD_IMAGE_GRAYSCALE))
		self.__grayscale = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_8U, 1)
		cv.Convert(gray, self.__grayscale)
		self.__integralImage = mt.integral_image(self.__grayscale)
		self.__responses = []
		self.__features = []
		self.__traces = []
		self.__orientation = []
		self.__descriptors = []
	
	def draw_orientation(self):
		if self.__features is not None:
			for i in range(0, len(self.__features)):
				(x, y, r) = (self.__features[i][0], 
					         self.__features[i][1], 
					         4*self.__features[i][2])
				theta = self.__orientation[i]
				X, Y = cv.Round(x + r*np.cos(theta)), cv.Round(y + r*np.sin(theta))
				cv.Circle(self.__image, (x, y), r, cv.RGB(255, 255, 0), thickness=1) 
				cv.Line(self.__image, (x, y), (X, Y), cv.RGB(255, 255, 0))
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
		print 'Analyzing image', self.__filename
		print '  Finding features...'
		self.__find_features()
		print '   ', len(self.__features), 'interest points found'
		self.__scale_space_refinement()	
		print '   ', len(self.__features), 'interest points after refinement'
		print '  Finding features\' orientation...'
		self.__find_orientation()
		print '  Building descriptors...'
		self.__build_descriptors()
		print '  Extraction success!' 
		self.__grayscale = None
		self.__integralImage = None
		self.__responses = []
		self.__traces = []
	
	def __box_integral(self, dx1, dx2, dy1, dy2, x, y):
		U = self.__integralImage
		if ((y-dy1-1) >= 0 and (x-dx1-1) >= 0 and 
		    (y+dy2) < U.height and (x+dx2) < U.width):
			return U[y-dy1-1, x-dx1-1] + U[y+dy2, x+dx2] - U[y+dy2, x-dx1-1] - U[y-dy1-1, x+dx2]
		else:
			return 0
	
	# Building response layers (filtering image with Fast-Hessian filters)
	def __build_resp_layers(self): 
		target_image = self.__grayscale
		image_size = (target_image.width, target_image.height)
		for s in Surf.SCALES:
			Dxx = cv.CloneImage(target_image)
			Dyy = cv.CloneImage(target_image)
			Dxy = cv.CloneImage(target_image)
			# Calculating convolutions
			cv.Filter2D(target_image, Dxx, 
					    Filters.fast_hessian(s, FastHessianType.DXX))
			cv.Filter2D(target_image, Dyy, 
					    Filters.fast_hessian(s, FastHessianType.DYY))		
			cv.Filter2D(target_image, Dxy, 
					    Filters.fast_hessian(s, FastHessianType.DXY))
			l = s / 3.0
			scale_factor1 = 1
			scale_factor2 = 1
			temp = cv.CloneImage(Dyy)
			cv.ConvertScale(Dyy, temp, scale=(1.0 / scale_factor1))
			#cv.SaveImage('D:\\hess\\dyy'+str(s)+'.jpg', temp)
			#cv.SaveImage('D:\\hess\\dyy'+str(s)+'.jpg', Dyy)
			#cv.SaveImage('D:\\hess\\dxy'+str(s)+'.jpg', Dxy)
			#Calculating hessian
			HessMatrS = cv.CreateImage(image_size, cv.IPL_DEPTH_64F, 1)
			DxxMultDyy = cv.CloneImage(target_image)
			cv.Mul(Dxx, Dyy, DxxMultDyy, 1.0 / scale_factor1**2)
			DxySquared = cv.CloneImage(target_image)
			cv.Mul(Dxy, Dxy, DxySquared, Surf.HESSIAN_RELATIVE_WEIGHT**2 / scale_factor2**2)
			cv.Sub(DxxMultDyy, DxySquared, HessMatrS)
			cv.SaveImage('D:\\hess\\hessian'+str(s)+'.jpg', HessMatrS)
			self.__responses.append(HessMatrS)
			#Calculating laplacian (trace of the hessian matrix)
			traceS = cv.CreateImage(image_size, cv.IPL_DEPTH_64F, 1)
			cv.Add(Dxx, Dyy, traceS)
			self.__traces.append(traceS)
		
	#Checks whether (x, y, s) is far enough from the borders		
	def __is_not_outlier(self, x, y, s):
		if s in range(1, 11):
			sigma = mt.SIGMA[s]
			dist = cv.Round(10*cv.Sqrt(2)*sigma + 10)
			w = self.__image.width
			h = self.__image.height
			return (x > dist and x < (w-dist) and 
				    y > dist and y < (h-dist))
		else:
			return False
	
	#Returns 1 if sign of the laplacian in (x, y, s) is '+', 0 otherwise
	def __laplacian_sign(self, x, y, s):
		if self.__traces[s][y, x] > 0:
			return 1
		else:
			return 0
	
	#Getting features
	def __find_features(self):
		self.__build_resp_layers()
		resp_layers = self.__responses
		for i in range(0, len(Surf.FILTER_MAP)):
			for j in range(1, 3):
				b = resp_layers[Surf.FILTER_MAP[i, j-1]]
				indM = Surf.FILTER_MAP[i, j]
				m = resp_layers[indM]
				t = resp_layers[Surf.FILTER_MAP[i, j+1]]
				candidates = mt.nms_2d(m, m.width, m.height, Surf.N)
				for (k, n) in candidates:
					if np.abs(m[k, n]) <= self.__thresh:
						continue
					failed = False
					for (x, y) in itertools.product(range(k-1, k+2), range(n-1, n+2)):
						if failed:
							break
						# > or >=
						if b[x, y] >= m[k, n] or t[x, y] >= m[k, n]:
							failed = True
					if not failed and self.__is_not_outlier(n, k, indM):
						self.__features.append((n, k, indM, 
											    self.__laplacian_sign(n, k, indM)))
		
	#Scale-space interpolation after non-maximum suppression
	def __scale_space_refinement(self):
		responses = self.__responses
		fake_maxima = []
		for i in range(0, len(self.__features)):
			(x, y, s, l_sign) = self.__features[i]
			k = Surf.REFINEMENT_ITERATION_COUNT
			while k >= 0 and self.__is_not_outlier(x, y, s):
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
						(x, y, s) = (int(x + delta_x + 0.5), 
									 int(y + delta_y + 0.5), 
									 int(s + delta_s/(0.4*(2**Surf.OCTAVES[s])) + 0.5))
				else:
					break
			if (k == -1 or not self.__is_not_outlier(x, y, s) or 
			    np.abs(x - self.__features[i][0]) > Surf.REFINEMENT_ITERATION_COUNT or 
			    np.abs(y - self.__features[i][1]) > Surf.REFINEMENT_ITERATION_COUNT):
				fake_maxima.append(self.__features[i])
			else:
				self.__features[i] = (x, y, s, l_sign)
		self.__features = mt.list_subtraction(self.__features, fake_maxima)
		
	def __find_orientation(self):
		for (x, y, s, l_sign) in self.__features:
			sigma = mt.SIGMA[s]
			l = cv.Round(2*sigma)
			haar_respX = {}
			haar_respY = {}
			angle = {}
			indexes = set(itertools.product(range(-6, 7), range(-6, 7)))
			for (i, j) in [elem for elem in indexes if (elem[0]**2 + elem[1]**2) <= 36]:
				X, Y = cv.Round(x + i * sigma), cv.Round(y + j * sigma)
				gauss = mt.gaussian(X-x, Y-y, 2*sigma)
				haar_respX[(i, j)] = (self.__box_integral(0, l, l, l, X, Y) - 
									  self.__box_integral(l, 0, l, l, X, Y)) * gauss
				haar_respY[(i, j)] = (self.__box_integral(l, l, l, 0, X, Y) - 
									  self.__box_integral(l, l, 0, l, X, Y)) * gauss
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
			(x, y, s, l_sign) = self.__features[k]
			sigma = mt.SIGMA[s]
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
					resp_x = (self.__box_integral(0, l, l, l, X, Y) - 
							  self.__box_integral(l, 0, l, l, X, Y))
					resp_y = (self.__box_integral(l, l, l, 0, X, Y) - 
							  self.__box_integral(l, l, 0, l, X, Y))
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