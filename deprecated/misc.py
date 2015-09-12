def on_exec_clicked(self):
        img1_filename = str(self.input1Edit.text())
        img2_filename = str(self.input2Edit.text())
        threshold = float(self.threshEdit.text())
        img1 = cv2.imread(img1_filename)
        img2 = cv2.imread(img2_filename)
        ipl_img1 = cv2.cv.fromarray(img1)
        ipl_img2 = cv2.cv.fromarray(img2)
        self.pixmapLabel = QtGui.QLabel(u'', self)
        self.pixmapLabel.setPixmap(convert_to_qt_pixmap(img1).scaledToHeight(400))
        self.pixmapLabel.setGeometry(self.pixmapLabel.x(), self.pixmapLabel.y(), 400,400)
        self.grid.addWidget(self.pixmapLabel, 7, 1, QtCore.Qt.AlignRight)
#         descriptors1 = SURFDetector.extract_features(ipl_img1, threshold)
#         descriptors2 = SURFDetector.extract_features(ipl_img2, threshold)
         
def on_draw_clicked(self):
    print 'Visualizing found correspondences...'
    outputFolder = str(self.outputEdit.text())
    img1, img2 = self.__surf1.get_image(), self.__surf2.get_image()
    kp1, kp2 = self.__surf1.get_features(), self.__surf2.get_features()
    img = combine_images(img1, img2)
    all_points_image = cv.CloneImage(img)
    shift = img1.width
    for match in self.__matches:
        (i, j) = match
        point1, point2 = kp1[i], kp2[j]            
        image = cv.CloneImage(img)
        draw_correspondence(image, shift, point1, point2, cv.RGB(255, 255, 0))
        draw_correspondence(all_points_image, shift, point1, point2, cv.RGB(255, 255, 0))
        cv.SaveImage(outputFolder + '\\' + str(i + j) + '.jpg', image)
    cv.SaveImage(outputFolder + '\\all_points.jpg', all_points_image)
    print 'Done!'

def on_ransac_clicked(self):
    pass
#         print 'RANSAC estimation'
#         t_dist = int(self.tDistEdit.text())
#         outputFolder = str(self.outputEdit.text())
#         img1, img2 = self.__surf1.get_image(), self.__surf2.get_image()
#         kp1, kp2 = self.__surf1.get_features(), self.__surf2.get_features()
#         matches = self.__matches
#         img = combine_images(img1, img2)
#         shift = img1.width
#         H = Ransac.find_homography_cv(kp1, kp2, matches)
#         myH = Ransac.estimate_homography(kp1, kp2, matches)
#         print 'H:'
#         print np.array(H)
#         print 'My H:'
#         print myH[0]
#         transformed1 = cv.CloneImage(img1)
#         transformed11 = cv.CloneImage(img2)
#         transformed2 = cv.CloneImage(img2)
#         transformed22 = cv.CloneImage(img2)
#         Hinv = cv.CreateMat(3, 3, cv.CV_64FC1)
#         cv.Invert(H, Hinv)
#         cv.WarpPerspective(img2, transformed2, H)
#         cv.WarpPerspective(img2, transformed22, Hinv)
#         cv.WarpPerspective(img1, transformed1, H)
#         cv.WarpPerspective(img1, transformed11, Hinv)
#         cv.SaveImage(outputFolder + '\\result1.jpg', transformed1)
#         cv.SaveImage(outputFolder + '\\result2.jpg', transformed11)
#         cv.SaveImage(outputFolder + '\\result3.jpg', transformed2)
#         cv.SaveImage(outputFolder + '\\result4.jpg', transformed22)
#         result = cv.CreateImage((img1.width + img2.width, img1.height), cv.IPL_DEPTH_64F, 3)
#         myHcv = cv.fromarray(myH[0])
#         myHcv_inv = cv.CreateMat(3, 3, cv.CV_64FC1)
#         cv.Invert(myHcv, myHcv_inv)
#         cv.WarpPerspective(img2, transformed2, myHcv)
#         cv.WarpPerspective(img2, transformed22, myHcv_inv)
#         cv.WarpPerspective(img1, transformed1, myHcv)
#         cv.WarpPerspective(img1, transformed11, myHcv_inv)
#         cv.AddWeighted(img1, 0.5, img2, 0.5, 0, result)        
#         cv.SaveImage(outputFolder + '\\result5.jpg', transformed1)
#         cv.SaveImage(outputFolder + '\\result6.jpg', transformed11)
#         cv.SaveImage(outputFolder + '\\result7.jpg', transformed2)
#         cv.SaveImage(outputFolder + '\\result8.jpg', transformed22)

def on_draw_orient_clicked(self):
    cv.SaveImage(str(self.outputEdit.text()) + '\\orient.jpg', self.__surf1.draw_orientation())