import cv2
from stitching.cvtools import imgtools

def test_lapl_pyramid():
    img = cv2.imread('C:\\Users\\Voldemar\\Desktop\\IMG_2.jpg')
    pyr = imgtools.lapl_pyramid(img, 5, True)
    j = 0
    for i in pyr:
        cv2.imwrite('C:\\Users\\Voldemar\\Desktop\\' + str(j) + '.jpg', i)
        j = j + 1
        
def test_gauss_pyramid():
    img = cv2.imread('C:\\Users\\Voldemar\\Desktop\\IMG_2.jpg')
    pyr = imgtools.gauss_pyramid(img, 5)
    j = 0
    for i in pyr:
        cv2.imwrite('C:\\Users\\Voldemar\\Desktop\\' + str(j) + '.jpg', i)
        j = j + 1
        
def test_gauss_interpolation():
    img = cv2.imread('C:\\Users\\Voldemar\\Desktop\\IMG_2.jpg') 
    img2 = cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(img))))
    cv2.imwrite('C:\\Users\\Voldemar\\Desktop\\' + str(9) + '.jpg', img2)
    
if __name__ == "__main__":
    test_gauss_pyramid()
    test_gauss_interpolation()