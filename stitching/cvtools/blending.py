import imgtools

def pyramid_blending(img1, img2, level_count=5):
    """
    Performs stitching of the specified images using Laplacian pyramids.
    
    Parameters
    ----------
    img1, img2 : ndarray
                 images to blend 
    homography : ndarray
                 projective transformation between images
    level_count : int
                  size of the Laplacian pyramids to use
                  
    Notes
    ----------
    img2 is supposed to be projected onto the plane of the img1 beforehand.
    """
    def merge_lapl_pyrs(lapl_pyramid1, lapl_pyramid2):
        merged_pyr = []
        for lapl1, lapl2 in zip(lapl_pyramid1, lapl_pyramid2):
            h, w = lapl1.shape[0], lapl1.shape[1]
            lapl2[0:h, 0:w] = lapl1
            merged_pyr.append(lapl2)
        return merged_pyr
    
    saturated = False
    lapl_pyramid1 = imgtools.lapl_pyramid(img1, level_count, saturated)
    lapl_pyramid2 = imgtools.lapl_pyramid(img2, level_count, saturated)
    
    merged_pyr = merge_lapl_pyrs(lapl_pyramid1, lapl_pyramid2)
    return imgtools.reconstruct_by_lapl_pyr(merged_pyr, saturated)  

def feathering(img1, img2, inliers):
    pass