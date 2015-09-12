import logging

from PyQt4 import QtGui, Qt
import cv2

logger = logging.getLogger('globalLogger')

__gray_color_table = [Qt.qRgb(i, i, i) for i in range(256)]

def convert_to_qt_pixmap(cv_image):
    is_grayscale = (len(cv_image.shape) == 2)
    height, width = cv_image.shape[0], cv_image.shape[1]
    img = cv_image if is_grayscale else cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img_format =  QtGui.QImage.Format_Indexed8 if is_grayscale else QtGui.QImage.Format_RGB888
    
    is_contiguous = img.flags['C_CONTIGUOUS']
    if is_contiguous:
        qt_image = QtGui.QImage(img.data, width, height, img.strides[0], img_format)
    else:
        qt_image = QtGui.QImage(img.tostring(), width, height, img_format)
        
    if is_grayscale:
        qt_image.setColorTable(__gray_color_table)
        
    pixmap = QtGui.QPixmap.fromImage(qt_image)    
    return pixmap

def read_stylesheet(fname):
    css = Qt.QString()
    css_file = Qt.QFile(fname)
    if css_file.open(Qt.QIODevice.ReadOnly):
        css = Qt.QString(css_file.readAll())
        css_file.close()
    return css