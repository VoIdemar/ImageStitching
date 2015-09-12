from PyQt4 import QtCore, QtGui

import properties

class PropertiesDialog(QtGui.QDialog):
    
    def __init__(self, parent=None):
        
        super(PropertiesDialog, self).__init__(parent)

        # OK and Cancel buttons
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        
        buttons.accepted.connect(self.apply_changes)
        buttons.rejected.connect(self.reject)
        
        self.property_grid = self.create_property_grid()
        
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addLayout(self.property_grid)
        self.layout.addWidget(buttons)        
        self.setLayout(self.layout)
        
        self.setWindowTitle(self.trUtf8(u'Global Properties'))
        
    def create_property_grid(self):
        if not properties.is_initialized():
            properties.read_config()
        config = properties.get_config() 
              
        allowed_sections = config.get('UI-AllowedSections', 'allowed').split(',')
        
        property_name_to_edit = {}
        
        grid_layout = QtGui.QGridLayout()
        i = 0
        for section in allowed_sections:
            section_label = QtGui.QLabel(section, self)
            grid_layout.addWidget(section_label, i, 0)
            i += 1
            for option, value in config.items(section):
                option_label = QtGui.QLabel(config.get('UI-PropertyNames', option) + ':', self)
                grid_layout.addWidget(option_label, i, 0)
                
                option_value_edit = QtGui.QLineEdit(value, self)
                grid_layout.addWidget(option_value_edit, i, 1)
                
                property_name_to_edit[(section, option)] = option_value_edit
                i += 1
            i += 1
            
        self.property_name_to_edit = property_name_to_edit
        
        return grid_layout
    
    def apply_changes(self):
        config = properties.get_config()
        for (section, option), edit in self.property_name_to_edit.items():
            config.set(section, option, str(edit.text()))
        properties.save_config()
        
        self.parent().centralWidget().load_stitching_settings()
        self.accept()