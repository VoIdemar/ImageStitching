import cv

SIZES = [1, 1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16]

SIGMA = [1.2, 2.0, 2.8, 3.6, 5.2, 6.8, 10.0, 13.2, 19.6, 26.0, 38.8, 51.6]

OUTLIER_DIST_THRESHOLD = map(lambda sigma: cv.Round(10*(1 + cv.Sqrt(2)*sigma)), 
                             SIGMA)

SCALE_FACTORS = [[0, 0,   0],
                 [1, 1,   1],
                 [1, 1, 0.5],
                 [2, 1,   1],
                 [1, 1, 0.5],
                 [2, 1,   1],
                 [1, 1, 0.5],
                 [2, 1,   1],
                 [1, 1, 0.5],
                 [2, 1,   1],
                 [1, 1,   1]]

OCTAVE_MAP = {9: 1, 15: 1, 21: 1, 27: 2, 39: 2, 51: 3, 75: 3, 99: 4, 147: 4, 195: 5, 291: 5, 387: 5}
  
SCALES = sorted(OCTAVE_MAP.keys())

OCTAVES = sorted(OCTAVE_MAP.values())