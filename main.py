import logging.config
import os

from PyQt4 import QtGui, QtCore
from PyQt4.Qt import QTranslator, QLibraryInfo

from stitching.ui.mainwindow import MainWindow
import properties

if __name__ == '__main__':
    config = properties.get_config()
    logging.config.fileConfig(config.get('Global', 'logConfig'))
    logger = logging.getLogger('globalLogger')
    
    try:
        import sys
        app = QtGui.QApplication(sys.argv)
        
        # Load localization settings
        language = config.get('Localization', 'language')
        localize_path = config.get('Localization', 'translationspath')
        filename = config.get('Localization', 'translationnameformat').format(language)
        
        # Qt localization settings
        qt_translator = QTranslator()    
        qt_translator.load('qt_' + language, 
                    QLibraryInfo.location(QLibraryInfo.TranslationsPath))
        app.installTranslator(qt_translator)
        
        # Application localization settings
        app_translator = QTranslator()    
        app_translator.load(os.path.join(localize_path, filename))
        app.installTranslator(app_translator)
        
        # Create main window
        form = MainWindow()
        form.setWindowTitle(u'PyStitch v0.7')
        
        # Create splash screen
        splash = QtGui.QSplashScreen(form, QtGui.QPixmap(config.get('UI', 'splashScreenPic')), 
                                     QtCore.Qt.WindowStaysOnTopHint)
        splash.setFont(QtGui.QFont('Consolas', 16))
        msg = form.trUtf8(u'Loading')
        splash.showMessage(msg + '... 0%',
                            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, QtCore.Qt.black)
        splash.show()
        QtGui.qApp.processEvents()
        
        # Show windows
        form.load_data(splash)
        form.show()
        splash.finish(form)
        form.set_geometry()
        
        sys.exit(app.exec_())
    except Exception as error:
        if not (logger is None):
            logger.exception(error)