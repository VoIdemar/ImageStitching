import cv2
import numpy as np

from stitching.surf.detector import SURFDetector
from stitching.surf.matching import DescriptorMatcher
from stitching.estimation.ransac import RansacHomographyEstimator
from stitching.utils.profiling import namedProfile
import stitching.cvtools.blending as blending
import cvtools.imgtools as imgtools

class PanoramaBuilder(object):
    
    def __init__(self, *images):
        self.__images = images
        
    @property
    def initial_images(self):
        return self.__images
    
    @namedProfile('./profiling/panorama_builder_opt_matching_ransac_surf.prof')
    def build(self):
        images_count = len(self.initial_images)
        if images_count == 0:
            return None
        elif images_count == 1:
            return self.initial_images[0]
        else:
            first_images = self.initial_images[0:2]
            panorama = PanoramaBuilder.stitch(*first_images)
            for img in self.initial_images[2:]:
                panorama = PanoramaBuilder.stitch(panorama, img)
            return panorama
            
    @staticmethod
    def stitch(img1, img2):
        descriptors1 = SURFDetector.extract_features(img1)
        descriptors2 = SURFDetector.extract_features(img2)
        matcher = DescriptorMatcher(descriptors1, descriptors2)
        matcher.match()
        
        estimator = RansacHomographyEstimator.create_for_matches(matcher.get_feature_matches())
        estimator.estimate()
        homography = estimator.homography
        
        h1, w1 = img1.shape[0], img1.shape[1]
        w2 = img2.shape[1]
        dsize = (w1 + w2, h1)
        inv_homography = np.linalg.inv(homography)
        warped_img2 = cv2.warpPerspective(img2, inv_homography, dsize)
        
        panorama = blending.pyramid_blending(img1, warped_img2, 5)
        
        return imgtools.crop_panorama(panorama)

@namedProfile('./profiling/stitching2.prof')
def align_images(img1, img2, threshold):
    descriptors1 = SURFDetector.extract_features(img1, threshold)
    descriptors2 = SURFDetector.extract_features(img2, threshold)
    matcher = DescriptorMatcher(descriptors1, descriptors2)
    matcher.match()
    estimator = RansacHomographyEstimator.create_for_matches(matcher.get_feature_matches())
    estimator.estimate()
    homography = estimator.homography
    print homography
    np.save('D:\\test\\test2\\ololo.txt', homography)

if __name__ == "__main__":
    
    fname1 = 'D:\\test\\test2\\35.jpg'
    fname2 = 'D:\\test\\test2\\36.jpg'

    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
#     surf = cv2.SURF(hessianThreshold=2)
#     keypoints = surf.detect(img1)
#     z = len(keypoints)
#     descriptors = SURFDetector.extract_features(img1, 2)
#     k = len(descriptors)
#     c = 0
#     for kp in keypoints:
#         x, y = kp.pt
#         
#         for d in descriptors:
#             xd, yd, _, _ = d.feature
#             if (abs(xd - x) <= 1 and abs(yd- y) <=1 ):
#                 c = c + 1
#                 print '({0}, {1}), ({2}, {3})'.format(x, y, xd, yd)
#         x = int(x)
#         y = int(y)
#         cv2.circle(img1, (x, y), 3, (0, 255, 255), 1)
#     print c, z, k
#     
    #cv2.imwrite('D:\\test\\test2\\b1_ololo.jpg', img1)
    builder = PanoramaBuilder(img1, img2)
    panorama = builder.build()
    cv2.imwrite('D:\\test\\qwert123.jpg', panorama)
    cv2.imshow("Panorama", panorama)
    cv2.waitKey()