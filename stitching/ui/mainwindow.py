#-*- coding: utf-8 -*-

import time

from PyQt4 import QtGui, QtCore
from PyQt4.Qt import QApplication

from stitching.ui.stitchwidget import StitchingWidget
from stitching.ui.propdialog import PropertiesDialog

class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()    
            
        self.setCentralWidget(StitchingWidget(self))
        
        self.init_actions()
        self.init_menubar()
        self.statusBar()
        self.set_geometry()
        
        self.setWindowIcon(QtGui.QIcon('common/main-window-icon.png'))
        
    def init_actions(self):
        self.exitAction = QtGui.QAction(QtGui.QIcon('icons/exit.png'), self.trUtf8(u'Exit'), self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip(self.trUtf8(u'Exit application'))
        self.exitAction.triggered.connect(self.close)
        
        self.propertiesAction = QtGui.QAction(self.trUtf8(u'Properties'), self)
        self.propertiesAction.setStatusTip(self.trUtf8(u'Set properties'))
        self.propertiesAction.setShortcut('Ctrl+P')
        self.propertiesAction.triggered.connect(self.open_properties_dialog)
        
    def init_menubar(self):
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&' + self.trUtf8(u'File'))
        self.fileMenu.addAction(self.propertiesAction)
        self.menubar.addAction(self.exitAction)        
        
    def set_geometry(self):
        geometry = QApplication.desktop().screenGeometry()
        width = int(0.7*geometry.width())
        height = int(0.7*geometry.height())
        self.setGeometry(50, 50, width, height)
     
    def open_properties_dialog(self):
        dialog = PropertiesDialog(self)
        _ = dialog.exec_()
        
    def load_data(self, splash):
        msg = self.trUtf8(u'Loading')
        for i in range(1, 11):
            time.sleep(0.2)
            percent = '... {0}%'.format(i*10)
            splash.showMessage(msg + percent,
                               QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.black)
            QtGui.qApp.processEvents()
    
if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    form = MainWindow()
    form.setWindowTitle(u'SURF')
    form.show()
    sys.exit(app.exec_())