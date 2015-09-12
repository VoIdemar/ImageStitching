class NoFeaturesError(Exception):
    def __init__(self, value):
        self.__msg = value
    
    def __str__(self):
        return self.__msg

class IncompatibleImagesError(Exception):
    def __init__(self, value):
        self.__msg = value
        
    def __str__(self):
        return self.__msg
