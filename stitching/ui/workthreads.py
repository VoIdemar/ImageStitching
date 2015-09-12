import logging 
from PyQt4 import QtCore

from stitching.surf.detector import SURFDetector
from stitching.surf.matching import DescriptorMatcher
from stitching.estimation.ransac import RansacHomographyEstimator
from stitching.facade import PanoramaBuilder

logger = logging.getLogger('threadsLogger') 

class ImageListThread(QtCore.QThread, object):
    
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.__selected_indices = []
        
    @property
    def selected_indices(self):
        return self.__selected_indices
    
    @selected_indices.setter
    def selected_indices(self, indices):
        self.__selected_indices = indices

# ===============================================================        
class SURFThread(ImageListThread, object):
    
    def __init__(self, parent=None):
        super(SURFThread, self).__init__(parent)
        self.__descriptors = []
    
    @property
    def descriptors(self):
        return self.__descriptors
    
    @property
    def image(self):
        return self.__image
    
    @image.setter
    def image(self, img):
        self.__image = img
    
    def run(self):
        try:
            self.__descriptors = SURFDetector.extract_features(self.__image)
        except Exception as error:
            logger.exception(error)

# ===============================================================        
class MatchingThread(ImageListThread, object):
    
    def __init__(self, parent=None):
        super(MatchingThread, self).__init__(parent)
        self.__descriptor_set1 = None
        self.__descriptor_set2 = None
        self.__feature_matches = []
        
    @property
    def descriptor_set1(self):
        return self.__descriptor_set1
    
    @property
    def descriptor_set2(self):
        return self.__descriptor_set2
    
    @descriptor_set1.setter
    def descriptor_set1(self, descriptor_set1):
        self.__descriptor_set1 = descriptor_set1
    
    @descriptor_set2.setter
    def descriptor_set2(self, descriptor_set2):
        self.__descriptor_set2 = descriptor_set2
    
    @property
    def feature_matches(self):
        return self.__feature_matches
    
    def run(self):
        try:
            matcher = DescriptorMatcher(self.__descriptor_set1, self.__descriptor_set2)
            matcher.match()
            self.__feature_matches = matcher.get_feature_matches()
        except Exception as error:
            logger.exception(error)
 
# ===============================================================       
class RANSACThread(ImageListThread, object):
    
    def __init__(self, parent=None):
        super(RANSACThread, self).__init__(parent)
        self.__descriptor_set1 = None
        self.__descriptor_set2 = None        
        self.__homography = None
        self.__inliers = []
        
    @property
    def descriptor_set1(self):
        return self.__descriptor_set1
    
    @property
    def descriptor_set2(self):
        return self.__descriptor_set2
    
    @descriptor_set1.setter
    def descriptor_set1(self, descriptor_set1):
        self.__descriptor_set1 = descriptor_set1
    
    @descriptor_set2.setter
    def descriptor_set2(self, descriptor_set2):
        self.__descriptor_set2 = descriptor_set2
        
    @property
    def homography(self):
        return self.__homography
    
    @property
    def inliers(self):
        return self.__inliers
    
    def run(self):
        try:
            matcher = DescriptorMatcher(self.__descriptor_set1, self.__descriptor_set2)
            matcher.match()
            estimator = RansacHomographyEstimator.create_for_matcher(matcher)
            estimator.estimate()
            self.__homography = estimator.homography
            self.__inliers = estimator.inliers
        except Exception as error:
            logger.exception(error)

# ===============================================================          
class StitchingThread(ImageListThread, object):
    
    def __init__(self, parent=None):
        super(StitchingThread, self).__init__(parent)
        self.__image1 = None
        self.__image2 = None        
        self.__panorama = None        
        
    @property
    def image1(self):
        return self.__image1
    
    @property
    def image2(self):
        return self.__image2
    
    @image1.setter
    def image1(self, img):
        self.__image1 = img
    
    @image2.setter
    def image2(self, img):
        self.__image2 = img
        
    @property
    def panorama(self):
        return self.__panorama
    
    def run(self):
        try:
            builder = PanoramaBuilder(self.image1, self.image2)
            self.__panorama = builder.build()
        except Exception as error:
            logger.exception(error)