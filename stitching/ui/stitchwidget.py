#-*- coding: utf-8 -*-
import sys

import numpy as np
import cv2
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import QSizePolicy, QSize

from stitching.ui.cvdrawing import draw_matches, draw_features
from stitching.ui.tools import convert_to_qt_pixmap, read_stylesheet
from stitching.surf.tools import save_descriptors_to_file
from stitching.cvtools.imgtools import nd_combine_images
from stitching.ui.workthreads import SURFThread, MatchingThread, RANSACThread, StitchingThread

import properties
from stitching.cvtools import imgtools

class StitchingWidget(QtGui.QWidget):
    
    ICON_SIZE = 50
    BUTTON_ICON_SIZE = QSize(30, 30)
    PREVIEW_SIZE = 300
    SAVE_FEATURES_DIALOG_FILTER = u'JPEG (*.jpg *.jpeg);;BMP (*.bmp);;PNG (*.png)'
    
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        # Images
        self.images = []
        self.images_with_features = {}
        self.descriptors = {}
        
        self.stylesheet = read_stylesheet('common/ui-styles.qss')
        
        # Buttons
        self.list_controls_group_box = QtGui.QGroupBox(self.trUtf8(u'List controls'), self)
        
        self.add_image_btn = QtGui.QPushButton(u'', self.list_controls_group_box)
        self.remove_image_btn = QtGui.QPushButton(u'', self.list_controls_group_box)
        self.save_features_btn = QtGui.QPushButton(u'', self.list_controls_group_box)
        self.show_features_btn = QtGui.QPushButton(u'', self.list_controls_group_box)

        self.control_buttons = [
             self.add_image_btn,
             self.remove_image_btn,
             self.save_features_btn,
             self.show_features_btn                   
        ]
        
        self.stitching_controls_group_box = QtGui.QGroupBox('&' + self.trUtf8(u'Algorithms'), self)
        self.stitching_controls_group_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.surf_btn = QtGui.QPushButton(u'SURF', self.stitching_controls_group_box)
        self.match_btn = QtGui.QPushButton(self.trUtf8(u'Match features'), self.stitching_controls_group_box)
        self.ransac_btn = QtGui.QPushButton(u'RANSAC', self.stitching_controls_group_box)
        self.homography_btn = QtGui.QPushButton(self.trUtf8(u'Homography'), self.stitching_controls_group_box)
        self.stitch_btn = QtGui.QPushButton(self.trUtf8(u'Stitch'), self.stitching_controls_group_box)
        self.to_grayscale_btn = QtGui.QPushButton(self.trUtf8(u'To grayscale'), self.stitching_controls_group_box)
        
        self.algo_buttons = [
            self.surf_btn,
            self.match_btn,
            self.ransac_btn,
            self.homography_btn,
            self.stitch_btn,
            self.to_grayscale_btn
        ]
      
        # Image list
        self.img_list = QtGui.QListWidget(self)
        self.img_list.setIconSize(QtCore.QSize(StitchingWidget.ICON_SIZE, StitchingWidget.ICON_SIZE))
        self.img_list.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        
        # Preview label
        self.preview_label = QtGui.QLabel(self)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #c0c0c0;")
        self.set_expanding_policy(self.preview_label)
        
        # Results widget
        self.results_widget = QtGui.QWidget(self)
        
        # Preview widget
        self.preview_widget = QtGui.QWidget(self)
        
        # Stitch widget
        self.stitch_widget = QtGui.QWidget(self)
        
        # Console
        self.console = QtGui.QPlainTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(120)        
        
        # Labels
        self.img_list_label = QtGui.QLabel(self.trUtf8(u'Images'), self)
        self.console_label = QtGui.QLabel(self.trUtf8(u'Console'), self)
        self.preview_text_label = QtGui.QLabel(self.trUtf8(u'Preview'), self)
        self.results_label = QtGui.QLabel(self.trUtf8(u'Results'), self.results_widget)
        
        self.result_img_label = QtGui.QLabel(u'', self.results_widget)
        self.result_img_label.setStyleSheet("background-color: #c0c0c0;")
        self.result_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.set_expanding_policy(self.result_img_label)

        # Splitters
        self.img_ctrl_splitter = QtGui.QSplitter(QtCore.Qt.Vertical, self)
        self.img_ctrl_splitter.setOpaqueResize(False)
        self.ctrl_central_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal, self)
        self.ctrl_central_splitter.setOpaqueResize(False)

        self.set_event_handlers()
        self.set_layouts()        
        self.set_geometry()
        self.set_stylesheet_to_widgets()
        self.set_icons()
        self.set_tooltips()
        self.load_stitching_settings()
        
        self.show_status(self.trUtf8('Application started'))
    
    def add_image(self):
        filename = QtGui.QFileDialog.getOpenFileName(
             parent=self,
             directory=u'D:\\test\\',
             caption=self.trUtf8(u'Open image'), 
             filter=(self.trUtf8(u'Images') + ' (*.png *.xpm *.jpg *.bmp)')
        )
        self.show_status(u'')               
        if not (filename is None) and not filename.isEmpty():
            img = cv2.imread(str(filename))
            self.images.append(img)
            img_height, img_width, _ = img.shape
            list_item = QtGui.QListWidgetItem(self.img_list)
            list_item.setText(self.trUtf8(u'Filename: %1\nImage size: %2 x %3')
                .arg(filename).arg(img_width).arg(img_height)
            )
            list_item.setIcon(QtGui.QIcon(
              convert_to_qt_pixmap(img).scaled(StitchingWidget.ICON_SIZE, StitchingWidget.ICON_SIZE, 
              QtCore.Qt.KeepAspectRatio))
            )
            self.img_list.addItem(list_item)
            msg = self.trUtf8(u'Image added')
            self.show_status(msg, msg + ': ' + filename)
            
    def remove_image(self):
        self.show_status(u'')
        selected_indices = sorted(self.get_selected_img_indices(), reverse=True)
        if self.img_list.currentRow() in selected_indices:
            self.preview_label.clear()
        for i in selected_indices:
            item = self.img_list.takeItem(i)
            self.images.pop(i)
            if self.images_with_features.has_key(i):
                self.images_with_features.pop(i)
            if self.descriptors.has_key(i):
                self.descriptors.pop(i)
            msg = self.trUtf8(u'Image removed')
            self.show_status(msg, msg + ':\n' + item.text())

    def preview_image_handler(self):
        selected = self.img_list.currentRow()
        if selected <> (-1):
            w, h = self.preview_label.width(), self.preview_label.height()
            self.preview_label.setPixmap(
              convert_to_qt_pixmap(self.images[selected]).scaled(w, h, QtCore.Qt.KeepAspectRatio)
            )
    
    def execute_surf(self):
        current_row = self.img_list.currentRow()
        if current_row <> (-1):
            img = np.copy(self.images[current_row])
            
            self.show_status(self.trUtf8(u'SURF started'), 
                    self.trUtf8(u'SURF started for image:\n') + self.img_list.item(current_row).text())
            self.start_surf_thread(img, current_row)
        else:
            self.show_msg_box(self.trUtf8(u'SURF Execution not started'), 
                              self.trUtf8(u'Please specify an image to be analyzed'))
    
    def show_features(self):
        current_row = self.img_list.currentRow()
        if current_row == -1:
            self.show_msg_box(self.trUtf8(u'Image not selected'), 
                              self.trUtf8(u'Please specify an image to show it\'s features'))
            return
        if self.images_with_features.has_key(current_row):
            self.set_result(self.images_with_features[current_row])
        else:
            self.show_msg_box(self.trUtf8(u'No features found'), 
                              self.trUtf8(u'No features has been found for this picture'))
            
    def save_features(self):
        current_row = self.img_list.currentRow()
        if self.images_with_features.has_key(current_row):
            ipl_img = self.images_with_features[current_row]
            self.save_image_to_file_by_dialog(ipl_img)
            descriptors_filename = QtGui.QFileDialog.getSaveFileName(
              parent=self,
              directory=u'D:\\test\\',
              caption=self.trUtf8(u'Save features'), 
              filter=self.trUtf8(u'Text') + ' (*.txt)'
            )
            if not descriptors_filename.isEmpty():
                save_descriptors_to_file(self.descriptors[current_row], str(descriptors_filename))
            
    def match_features(self):
        selected_indices = self.get_selected_img_indices()
        if len(selected_indices) == 2:
            i1, i2 = selected_indices
            if self.descriptors.has_key(i1) and self.descriptors.has_key(i2):
                descriptor_set1 = self.descriptors[i1]
                descriptor_set2 = self.descriptors[i2]
                self.show_status(self.trUtf8(u'Matching started'), 
                     self.trUtf8(u'Matching started for images:\n') + self.get_selected_img_descriptions())
                self.start_matching_thread(descriptor_set1, descriptor_set2, selected_indices)
            else:
                self.show_msg_box(self.trUtf8(u'No features found'), 
                    self.trUtf8(u'Features of at least one of the specified images have not been found.\n'))
        else:
            self.show_msg_box(self.trUtf8(u'Wrong number of images'), 
                    self.trUtf8(u'2 images have to be specified for features matching'))

    def set_result(self, result_img):
        w, h = self.result_img_label.width(), self.result_img_label.height()         
        self.result_img_label.setPixmap(
            convert_to_qt_pixmap(result_img).scaled(w, h, 
             QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )
    
    def save_image_to_file_by_dialog(self, img):
        img_filename = QtGui.QFileDialog.getSaveFileName(
          parent=self,
          directory=u'D:\\test\\',
          caption=self.trUtf8(u'Save image'), 
          filter=StitchingWidget.SAVE_FEATURES_DIALOG_FILTER
        )
        if not img_filename.isEmpty():
            cv2.imwrite(str(img_filename), img)
    
    def ransac_homography(self):
        selected_indices = self.get_selected_img_indices()
        if len(selected_indices) == 2:
            i1, i2 = selected_indices
            if self.descriptors.has_key(i1) and self.descriptors.has_key(i2):
                descriptor_set1 = self.descriptors[i1]
                descriptor_set2 = self.descriptors[i2]
                self.show_status(self.trUtf8(u'RANSAC started'), 
                            self.trUtf8(u'RANSAC started for images'))
                self.start_ransac_thread(descriptor_set1, descriptor_set2, selected_indices)
            else:
                self.show_msg_box(self.trUtf8(u'No features found'), 
                    self.trUtf8(u'Features of at least one of the specified images have not been found.\n'))
        else:
            self.show_msg_box(self.trUtf8(u'Wrong number of images'),
                        self.trUtf8(u'2 images have to be specified for features matching'))
    
    def stitch_images(self):
        selected_indices = self.get_selected_img_indices()
        if len(selected_indices) == 2:
            i1, i2 = selected_indices
            img1 = self.images[i1]
            img2 = self.images[i2]
            self.show_status(self.trUtf8(u'Stitching started'), 
                self.trUtf8(u'Stitching started for images:\n') + self.get_selected_img_descriptions())
            self.start_stitching_thread(img1, img2)
        else:
            self.show_msg_box(self.trUtf8(u'Wrong number of images'),
                    self.trUtf8(u'2 images have to be specified for stitching'))
    
    def to_grayscale(self):
        selected_indices = self.get_selected_img_indices()
        for idx in selected_indices:
            img = imgtools.ndarray_to_grayscale(self.images[idx])
            self.images[idx] = img
            list_item = self.img_list.item(idx)
            list_item.setIcon(QtGui.QIcon(
              convert_to_qt_pixmap(img).scaled(StitchingWidget.ICON_SIZE, StitchingWidget.ICON_SIZE, 
              QtCore.Qt.KeepAspectRatio))
            )
        self.preview_label.clear()
    
    def resize_handler(self, event):
        super(StitchingWidget, self).resizeEvent(event)
        self.preview_image_handler()
    
    def show_status(self, status, log_msg=u''):
        self.parent().setStatusTip(status)
        self.console_log(log_msg)
        
    def show_msg_box(self, title, msg):
        QtGui.QMessageBox.about(self, title, msg)
        
    def set_expanding_policy(self, widget):
        policy = QtGui.QSizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Expanding)
        policy.setHorizontalPolicy(QSizePolicy.Expanding)   
        widget.setSizePolicy(policy)
        
    def set_stylesheet_to_widgets(self):
        for button in self.control_buttons:
            button.setStyleSheet(self.stylesheet)
        self.img_ctrl_splitter.setStyleSheet(self.stylesheet)        
    
    def set_icons(self):
        self.add_image_btn.setIcon(QtGui.QIcon('common/icons/plus-icon.png'))        
        self.remove_image_btn.setIcon(QtGui.QIcon('common/icons/minus-icon.png'))
        self.show_features_btn.setIcon(QtGui.QIcon('common/icons/magnifier-icon.png'))
        self.save_features_btn.setIcon(QtGui.QIcon('common/icons/save-icon.png'))
        
        size = StitchingWidget.BUTTON_ICON_SIZE
        for button in self.control_buttons:
            button.setIconSize(size)
        
    def set_geometry(self):  
        self.img_ctrl_splitter.setMinimumWidth(150)
        self.img_ctrl_splitter.setSizes([300, 300])
        
        self.stitching_controls_group_box.setFixedWidth(130)
        self.ctrl_central_splitter.setSizes([150, self.parent().width()-150])        
    
    def set_layouts(self):
        # Layouts
        
        # Control buttons
        control_buttons_layout = QtGui.QHBoxLayout()
        for button in self.control_buttons:
            control_buttons_layout.addWidget(button)
        self.list_controls_group_box.setLayout(control_buttons_layout)
        
        # Algorithm buttons     
        algorithm_buttons_layout = QtGui.QVBoxLayout()
        for button in self.algo_buttons:
            algorithm_buttons_layout.addWidget(button)
        self.stitching_controls_group_box.setLayout(algorithm_buttons_layout)
        
        # Image list and list controls
        self.img_list_ctrl_widget = QtGui.QWidget(self)        
        list_and_controls_layout = QtGui.QVBoxLayout()
        list_and_controls_layout.addWidget(self.list_controls_group_box)
        list_and_controls_layout.addWidget(self.img_list_label)
        list_and_controls_layout.addWidget(self.img_list)
        self.img_list_ctrl_widget.setLayout(list_and_controls_layout)
        
        # Preview image layout       
        preview_layout = QtGui.QVBoxLayout()
        preview_layout.addWidget(self.preview_text_label)
        preview_layout.addWidget(self.preview_label)
        self.preview_widget.setLayout(preview_layout)
        
        # Image list and Preview image splitter
        self.img_ctrl_splitter.addWidget(self.img_list_ctrl_widget)
        self.img_ctrl_splitter.addWidget(self.preview_widget)     
        
        # Results widget layout
        results_widget_layout = QtGui.QVBoxLayout()
        results_widget_layout.addWidget(self.results_label)
        results_widget_layout.addWidget(self.result_img_label)
        self.results_widget.setLayout(results_widget_layout)
        
        # ALgorithms layout
        self.algorithms_layout = QtGui.QHBoxLayout()
        self.algorithms_layout.addWidget(self.results_widget)
        self.algorithms_layout.addWidget(self.stitching_controls_group_box, 
                                         alignment=QtCore.Qt.AlignTop)
        
        # Stitching layout 
        self.stitching_layout = QtGui.QVBoxLayout()
        self.stitching_layout.addWidget(self.results_label)
        self.stitching_layout.addLayout(self.algorithms_layout)
        self.stitching_layout.addWidget(self.console_label)
        self.stitching_layout.addWidget(self.console)
        self.stitch_widget.setLayout(self.stitching_layout)
        
        # List controls/Preview image and Stitching controls splitter
        self.ctrl_central_splitter.addWidget(self.img_ctrl_splitter)
        self.ctrl_central_splitter.addWidget(self.stitch_widget)
        
        self.main_widget_layout = QtGui.QHBoxLayout(self)    
        self.main_widget_layout.addWidget(self.ctrl_central_splitter)        
        
        self.set_expanding_policy(self.result_img_label)

    def set_event_handlers(self):
        # Buttons' event handlers
        self.add_image_btn.clicked.connect(self.add_image)
        self.remove_image_btn.clicked.connect(self.remove_image)
        self.save_features_btn.clicked.connect(self.save_features)
        self.show_features_btn.clicked.connect(self.show_features)
        
        self.surf_btn.clicked.connect(self.execute_surf)
        self.match_btn.clicked.connect(self.match_features)
        self.ransac_btn.clicked.connect(self.ransac_homography)
        self.stitch_btn.clicked.connect(self.stitch_images)
        self.to_grayscale_btn.clicked.connect(self.to_grayscale)
        
        # Image list event handlers
        self.img_list.itemClicked.connect(self.preview_image_handler)        
        
        self.resizeEvent = self.resize_handler
        
        # Splitter event handlers
        self.img_ctrl_splitter.splitterMoved.connect(self.splitter_moved)
        self.ctrl_central_splitter.splitterMoved.connect(self.splitter_moved)
        
    def set_tooltips(self):
        self.add_image_btn.setToolTip(self.trUtf8(u'Add image'))
        self.remove_image_btn.setToolTip(self.trUtf8(u'Remove image'))
        self.save_features_btn.setToolTip(self.trUtf8(u'Save features to file'))
        self.show_features_btn.setToolTip(self.trUtf8(u'Show features'))
    
    def start_surf_thread(self, img, current_row):
        surf_thread = SURFThread(self)
        surf_thread.image = img
        surf_thread.selected_indices = [current_row]
        surf_thread.finished.connect(lambda thread=surf_thread: self.surf_execution_finished(thread))
        surf_thread.start(QtCore.QThread.TimeCriticalPriority)
        
    def start_matching_thread(self, descriptor_set1, descriptor_set2, selected_indices):
        matching_thread = MatchingThread(self)
        matching_thread.descriptor_set1 = descriptor_set1
        matching_thread.descriptor_set2 = descriptor_set2
        matching_thread.selected_indices = selected_indices
        matching_thread.finished.connect(lambda thread=matching_thread: self.matching_finished(thread))
        matching_thread.start(QtCore.QThread.TimeCriticalPriority)
        
    def start_ransac_thread(self, descriptor_set1, descriptor_set2, selected_indices):
        ransac_thread = RANSACThread(self)
        ransac_thread.descriptor_set1 = descriptor_set1
        ransac_thread.descriptor_set2 = descriptor_set2
        ransac_thread.selected_indices = selected_indices
        ransac_thread.finished.connect(lambda thread=ransac_thread: self.ransac_finished(thread))
        ransac_thread.start(QtCore.QThread.TimeCriticalPriority)
        
    def start_stitching_thread(self, image1, image2):
        stitching_thread = StitchingThread(self)
        stitching_thread.image1 = image1
        stitching_thread.image2 = image2        
        stitching_thread.finished.connect(lambda thread=stitching_thread: self.stitching_finished(thread))
        stitching_thread.start(QtCore.QThread.TimeCriticalPriority)
    
    def surf_execution_finished(self, finished_thread):
        img = finished_thread.image
        descriptors = finished_thread.descriptors
        current_row = finished_thread.selected_indices[0]
        
        draw_features(img, descriptors)
        self.images_with_features[current_row] = img
        self.descriptors[current_row] = descriptors
        self.set_result(img)
        
        item = self.img_list.item(current_row)
        image_description = item.text()
        item.setText(image_description + self.trUtf8(u'\nFeatures: found'))
        self.show_msg_box(self.trUtf8(u'SURF Execution finished!'), 
            self.trUtf8(u'SURF algorithm execution finished for image:\n') + image_description)
        
        self.show_status(self.trUtf8(u'SURF method finished'))
        
    def matching_finished(self, finished_thread):
        i1, i2 = finished_thread.selected_indices
        feature_matches = finished_thread.feature_matches
        
        img1, img2 = self.images[i1], self.images[i2]
        combined_image, shift = nd_combine_images(img1, img2) 
        draw_matches(feature_matches, combined_image, shift, (0, 255, 255))
        
        self.save_image_to_file_by_dialog(combined_image)
        self.set_result(combined_image)
        
        img1_description = self.img_list.item(i1).text()
        img2_description = self.img_list.item(i2).text()
        description = (self.trUtf8('Image 1:\n%1\n\nImage 2:\n%2\n\nMatches count: %3')
            .arg(img1_description).arg(img2_description).arg(len(feature_matches)))
        self.show_msg_box(self.trUtf8(u'Descriptor matching finished!'), 
            self.trUtf8(u'Descriptor matching finished for images:\n') + description)
        
        self.show_status(self.trUtf8(u'Descriptor matching finished'))
        
    def ransac_finished(self, finished_thread):
        i1, i2 = finished_thread.selected_indices
        inliers = finished_thread.inliers
        
        img1, img2 = self.images[i1], self.images[i2]
        combined_image, shift = nd_combine_images(img1, img2) 
        draw_matches(inliers, combined_image, shift, (0, 255, 255))
        
        self.save_image_to_file_by_dialog(combined_image)
        self.set_result(combined_image)
        
        img1_description = self.img_list.item(i1).text()
        img2_description = self.img_list.item(i2).text()
        description = (self.trUtf8('Image 1:\n%1\n\nImage 2:\n%2\n\nInliers count: %3')
            .arg(img1_description).arg(img2_description).arg(len(inliers)))
        self.show_msg_box(self.trUtf8(u'RANSAC finished!'), 
            self.trUtf8(u'RANSAC finished for images:\n') + description)
        
        self.show_status(self.trUtf8(u'RANSAC finished'))
        
    def stitching_finished(self, finished_thread):
        panorama = finished_thread.panorama
        self.save_image_to_file_by_dialog(panorama)
        self.set_result(panorama)
        
        msg = self.trUtf8(u'Stitching finished')
        self.show_msg_box(msg + '!', msg + ' ' + self.trUtf8(u'successfully!'))
        self.show_status(msg)
    
    def console_log(self, txt):
        if txt <> '':
            self.console.appendPlainText('> ' + txt)
    
    def load_stitching_settings(self):
        properties.load_stitching_settings()
        
    def get_selected_img_indices(self):
        return [index.row() for index in self.img_list.selectedIndexes()]
    
    def get_selected_img_descriptions(self):
        indices = self.get_selected_img_indices()
        img_descriptions = [unicode(self.img_list.item(idx).text())
                            for idx in indices]
        return (u',\n\n').join(img_descriptions)
    
    def splitter_moved(self, pos, idx):
        self.preview_image_handler()
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = StitchingWidget()
    form.setWindowTitle(u'SURF')
    form.show()
    sys.exit(app.exec_())