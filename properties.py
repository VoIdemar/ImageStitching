from ConfigParser import SafeConfigParser

from stitching.estimation.ransac import RansacHomographyEstimator

__parser = None

def read_config():
    global __parser 
    __parser = SafeConfigParser()
    __parser.read('./config/config.ini')

def get_config():
    if not is_initialized():
        read_config()
    return __parser

def is_initialized():
    return not (__parser is None)

def save_config():
    if is_initialized():
        config = get_config()
        with open(config.get('Global', 'configFile'), 'w') as config_file:
            config.write(config_file)
            
def load_stitching_settings():
    if not is_initialized():
        read_config()
    config = get_config()
    
    # Set RANSAC properties
    RansacHomographyEstimator.INITIAL_NUMBER_OF_SAMPLES = config.getint(
            'RANSAC', 'initialNumOfSamples')
    RansacHomographyEstimator.MIN_STD = config.getfloat(
            'RANSAC', 'minStd')
    RansacHomographyEstimator.P = config.getfloat(
            'RANSAC', 'suitableSetProbability')
    RansacHomographyEstimator.DIST_THRESHOLD = config.getfloat(
            'RANSAC', 'distThreshold')
    RansacHomographyEstimator.ADAPTIVE_THRESH_ENABLED = config.getboolean(
            'RANSAC', 'adaptiveThresholdEnabled')
    RansacHomographyEstimator.THRESH_CHANGE_STEP = config.getfloat(
            'RANSAC', 'threshChangeStep')
    RansacHomographyEstimator.THRESH_CHANGE_FREQ = config.getint(
            'RANSAC', 'threshChangeFreq')