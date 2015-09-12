import operator

import cv
import cv2
import numpy as np

from stitching.surf.errors import  IncompatibleImagesError

def symmetric_ext(image, offset_width, offset_height):
    """
    Returns the extension of the original image reflecting parts of it beyond original image borders.
    """
    width, height = image.width, image.height
    ext_width, ext_height = width + 2*offset_width, height + 2*offset_height
    extended = cv.CreateImage((ext_width, ext_height), image.depth, image.nChannels)
    #Copying image
    cv.SetImageROI(extended, (offset_width, offset_height, width, height))
    cv.Copy(image, extended)
    cv.ResetImageROI(extended)
    #Copying upper-left corner
    for (x, y) in ((x, y) for x in range(0, offset_width) for y in range(0, offset_height)):
        extended[y, x] = image[offset_height-y, offset_width-x]
    #Copying upper-right corner
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(0, offset_height)):
        extended[y, x] = image[offset_height-y, 2*width-1+offset_width-x]
    #Copying bottom-left corner
    for (x, y) in ((x, y) for x in range(0, offset_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, offset_width-x]
    #Copying bottom-right corner
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, 2*width-1+offset_width-x]
    #Copying upper stripe 
    for (x, y) in ((x, y) for x in range(offset_width, ext_width-offset_width) for y in range(0, offset_height)):
        extended[y, x] = image[offset_height-y, x-offset_width]
    #Copying lower stripe
    for (x, y) in ((x, y) for x in range(offset_width, ext_width-offset_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, x-offset_width]
    #Copying left stripe
    for (x, y) in ((x, y) for x in range(0, offset_width) for y in range(offset_height, ext_height-offset_height)):
        extended[y, x] = image[y-offset_height, offset_width-x]
    #Copying right stripe
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(offset_height, ext_height-offset_height)):
        extended[y, x] = image[y-offset_height, 2*width-1+offset_width-x]
    return extended

def ndarray_symmetric_ext(image, offset_width, offset_height):
    """
    Returns the extension of the original image reflecting parts of it beyond original image borders.
    """
    height = width = 0     
    nchannels = 1
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, nchannels = image.shape
    ext_width, ext_height = width + 2*offset_width, height + 2*offset_height
    ext_size = (ext_height, ext_width) if nchannels == 1 else (ext_height, ext_width, nchannels)
    extended = np.ndarray(ext_size, dtype=image.dtype)
    #Copying image
    extended[offset_height:offset_height+height, offset_width:offset_width+width] = image
    #Copying upper-left corner
    for (x, y) in ((x, y) for x in range(offset_width) for y in range(offset_height)):
        extended[y, x] = image[offset_height-y, offset_width-x]
    #Copying upper-right corner
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(offset_height)):
        extended[y, x] = image[offset_height-y, 2*width-1+offset_width-x]
    #Copying bottom-left corner
    for (x, y) in ((x, y) for x in range(offset_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, offset_width-x]
    #Copying bottom-right corner
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, 2*width-1+offset_width-x]
    #Copying upper stripe 
    for (x, y) in ((x, y) for x in range(offset_width, ext_width-offset_width) for y in range(offset_height)):
        extended[y, x] = image[offset_height-y, x-offset_width]
    #Copying lower stripe
    for (x, y) in ((x, y) for x in range(offset_width, ext_width-offset_width) for y in range(ext_height-offset_height, ext_height)):
        extended[y, x] = image[2*height-1+offset_height-y, x-offset_width]
    #Copying left stripe
    for (x, y) in ((x, y) for x in range(offset_width) for y in range(offset_height, ext_height-offset_height)):
        extended[y, x] = image[y-offset_height, offset_width-x]
    #Copying right stripe
    for (x, y) in ((x, y) for x in range(ext_width-offset_width, ext_width) for y in range(offset_height, ext_height-offset_height)):
        extended[y, x] = image[y-offset_height, 2*width-1+offset_width-x]
    return extended

def normalize_image(image):
    """
    Returns normalized copy of the grayscale image in case of low contrast.
    """
    result = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 1)
    minimum, maximum, _, _ = cv.MinMaxLoc(image)
    r = 255/float(maximum - minimum)
    for (x, y) in ((x, y) for x in range (0, image.width) for y in range(0, image.height)):        
        result[y, x] = int(r*(image[y, x] - minimum))
    return result

def ndarray_normalize_image(image):
    """
    Returns normalized copy of the grayscale image in case of low contrast.
    """
    minimum, maximum, _, _ = cv2.minMaxLoc(image)
    r = 255/float(maximum - minimum)
    return (r*(image - minimum)).astype(np.uint8)

def to_grayscale(image):
    grayscale = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 1)
    cv.ConvertImage(image, grayscale)
    return grayscale

def ndarray_to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def depth_to_matrix_type(depth):
    if depth == cv.IPL_DEPTH_8U:
        return cv.CV_8U
    elif depth == cv.IPL_DEPTH_8S:
        return cv.CV_8S
    elif depth == cv.IPL_DEPTH_16U:
        return cv.CV_16U
    elif depth == cv.IPL_DEPTH_16S:
        return cv.CV_16S
    elif depth == cv.IPL_DEPTH_32S:
        return cv.CV_32S
    elif depth == cv.IPL_DEPTH_32F:
        return cv.CV_32F
    elif depth == cv.IPL_DEPTH_64F:
        return cv.CV_64F
    return -1

def combine_images(img1, img2):
    if img1.nChannels == img2.nChannels and img1.depth == img2.depth:
        depth = img1.depth
        nChannels = img1.nChannels
        height = np.max([img1.height, img2.height])
        width = img1.width + img2.width        
        result = cv.CreateImage((width, height), depth, nChannels)
        cv.SetImageROI(result, (0, 0, img1.width, img1.height))
        cv.Copy(img1, result)
        cv.ResetImageROI(result)
        cv.SetImageROI(result, (img1.width, 0, img2.width, img2.height))
        cv.Copy(img2, result)
        cv.ResetImageROI(result)
        shift = img1.width
        return (result, shift)
    else:
        raise IncompatibleImagesError('Incompatible color depth')

def nd_combine_images(img1, img2):
    if len(img1.shape) <> len(img2.shape) and img1.dtype <> img2.dtype:
        raise IncompatibleImagesError('Incompatible color depth')
    are_grayscale = (len(img1.shape) == 2)
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    h = max([h1, h2])
    w = w1 + w2
    result = None
    if are_grayscale:
        result = np.empty((h, w), img1.dtype)
    else:
        d1 = img1.shape[2]
        d2 = img2.shape[2]
        if d1 <> d2:
            raise IncompatibleImagesError('Incompatible color depth')
        result = np.empty((h, w, d1), img1.dtype)        
    result[:h1, :w1] = img1
    result[:h2, w1:] = img2
    return (result, w1)
    
def gauss_pyramid(img, octave_count):
    downsampled = img[:,:]
    octaves = [downsampled]
    for _ in range(octave_count):
        downsampled = cv2.pyrDown(downsampled)
        octaves.append(downsampled)
    return octaves

def lapl_pyramid(img, octave_count, saturated=True):
    gauss_pyr = gauss_pyramid(img, octave_count)
    return lapl_pyramid_by_pyr(gauss_pyr, saturated)

def lapl_pyramid_by_pyr(gauss_pyramid, saturated=True):
    subtract = cv2.subtract if saturated else operator.sub
    lapl_pyramid = [gauss_pyramid[-1]]
    for i in range(len(gauss_pyramid)-1, 0, -1):
        upsampled = cv2.pyrUp(gauss_pyramid[i])
        aligned_gp_img, aligned_upsampled = aligned_images(gauss_pyramid[i-1], upsampled)
        lapl_octave = subtract(aligned_gp_img, aligned_upsampled)
        lapl_pyramid.append(lapl_octave)
    return lapl_pyramid

def reconstruct_by_lapl_pyr(lapl_pyr, saturated=True):
    add = cv2.add if saturated else operator.add
    result_img = lapl_pyr[0]
    for octave in lapl_pyr[1:]:
        aligned_lapl_img, aligned_octave = aligned_images(cv2.pyrUp(result_img), octave)
        result_img = add(aligned_lapl_img, aligned_octave)
    return result_img

def crop_panorama(img):
    """
    Removes black edges which appeared during image alignment stage.
    """
    grayscale = ndarray_to_grayscale(img)
    blurred = cv2.medianBlur(grayscale, 3)
    _, binary = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = get_biggest_contour(contours)
    approxContour = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    r, _, c = approxContour.shape
    approxContour = approxContour.reshape(r, c)
    max_x, max_y = np.max(approxContour, axis=0)
    bottom_right_corner_idx = (approxContour[:,0]*approxContour[:,1]).argmax()
    far_x, far_y = approxContour[bottom_right_corner_idx]
    x = min([max_x, far_x])
    y = min([max_y, far_y])
    return img[:y, :x]
    
def get_biggest_contour(contours):
    areas = map(lambda contour: cv2.contourArea(contour), contours)
    contour_idx = np.argmax(areas)
    return contours[contour_idx]

def aligned_images(*images):
    min_shape = min([img.shape for img in images])
    h, w = min_shape[0], min_shape[1]
    return [img[:h, :w] for img in images]