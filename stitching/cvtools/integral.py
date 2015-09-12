import cv2
  
from stitching.cvtools import imgtools
from stitching.surf.constants import SCALES

class IntegralImage(object):
    
    EXT_SIZE = SCALES[-1]/3
    OFFSET = EXT_SIZE + (EXT_SIZE-1)/2 + 10
    
    def __init__(self, original):        
        oh, ow = original.shape
        self.__original_width = ow
        self.__original_height = oh
        offset = IntegralImage.OFFSET
        
        extended = imgtools.ndarray_symmetric_ext(original, offset, offset)
        h, w = extended.shape
        self.__integral = cv2.integral(extended)[1:h, 1:w]
        
        self.__haar_x_resp = {}
        self.__haar_y_resp = {}       
    
    @property
    def shape(self):
        return self.__integral.shape
    
    @property
    def original_width(self):
        return self.__original_width
    
    @property
    def original_height(self):
        return self.__original_height
    
    @property
    def haar_x_resp(self):
        return self.__haar_x_resp
    
    @property
    def haar_y_resp(self):
        return self.__haar_y_resp
    
    def box_integral(self, dx1, dx2, dy1, dy2, X, Y):        
        get_item = self.__integral.item
        offset = IntegralImage.OFFSET
        x = X + offset
        y = Y + offset
        y1 = y-dy1-1
        x1 = x-dx1-1
        y2 = y+dy2
        x2 = x+dx2
        return get_item(y1, x1) + get_item(y2, x2) - get_item(y2, x1) - get_item(y1, x2)
    
    def nd_box_integral(self, dx1, dx2, dy1, dy2):
        integral = self.__integral
        offset = IntegralImage.OFFSET
        w = self.original_width
        h = self.original_height
        
        x_shift1 = dx1
        x_shift2 = -dx2 - 1
        y_shift1 = dy1
        y_shift2 = -dy2 - 1
        
        u1 = integral[offset + y_shift1 : offset + y_shift1 + h, 
                      offset + x_shift1 : offset + x_shift1 + w]
        u2 = integral[offset + y_shift2 : offset + y_shift2 + h, 
                      offset + x_shift2 : offset + x_shift2 + w]
        u3 = integral[offset + y_shift2 : offset + y_shift2 + h, 
                      offset + x_shift1 : offset + x_shift1 + w]
        u4 = integral[offset + y_shift1 : offset + y_shift1 + h, 
                      offset + x_shift2 : offset + x_shift2 + w]
        
        return u1 + u2 - u3 - u4
    
    def box_integral_wh(self, w, h, x, y):
        return self.box_integral(w, w, h, h, x, y)
    
    def haar_x(self, x, y, L):
        bi = self.box_integral
        return bi(L, -1, L, L, x, y) - bi(-1, L, L, L, x, y)
    
    def nd_haar_x(self, L):        
        if not self.haar_x_resp.has_key(L):
            bi = self.nd_box_integral
            self.haar_x_resp[L] = bi(L, -1, L, L) - bi(-1, L, L, L)
        return self.haar_x_resp[L]
    
    def haar_y(self, x, y, L):
        bi = self.box_integral
        return bi(L, L, L, -1, x, y) - bi(L, L, -1, L, x, y)
    
    def nd_haar_y(self, L):
        if not self.haar_y_resp.has_key(L):
            bi = self.nd_box_integral
            self.haar_y_resp[L] = bi(L, L, L, -1) - bi(L, L, -1, L)
        return self.haar_y_resp[L]

    def dxx(self, x, y, L):
        bi = self.box_integral
        return (bi(L + (L-1)/2, L + (L-1)/2, L-1, L-1, x, y) - 
                3*bi((L-1)/2, (L-1)/2, L-1, L-1, x, y))
    
    def nd_dxx(self, L):
        bi = self.nd_box_integral
        return (bi(L + (L-1)/2, L + (L-1)/2, L-1, L-1) - 3*bi((L-1)/2, (L-1)/2, L-1, L-1))
        
    def dxy(self, x, y, L):
        bi = self.box_integral
        return (bi(-1, L, -1, L, x, y) + bi(L, -1, L, -1, x, y) -
                (bi(L, -1, -1, L, x, y) + bi(-1, L, L, -1, x, y)))
        
    def nd_dxy(self, L):
        bi = self.nd_box_integral
        return (bi(-1, L, -1, L) + bi(L, -1, L, -1) - (bi(L, -1, -1, L) + bi(-1, L, L, -1)))
    
    def dyy(self, x, y, L):
        bi = self.box_integral
        return (bi(L-1, L-1, L + (L-1)/2, L + (L-1)/2, x, y) - 
                3*bi(L-1, L-1, (L-1)/2, (L-1)/2, x, y))
        
    def nd_dyy(self, L):
        bi = self.nd_box_integral
        return (bi(L-1, L-1, L + (L-1)/2, L + (L-1)/2) - 3*bi(L-1, L-1, (L-1)/2, (L-1)/2))
        
    def __getitem__(self, (y, x)):
        return self.__integral.item(y, x)
    
    def __str__(self):
        return str(self.__integral)