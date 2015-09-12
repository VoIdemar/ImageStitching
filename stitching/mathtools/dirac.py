class DiracFunction2D(object):
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def __call__(self, x, y):
        return 1 if (x, y) == (self.a, self.b) else 0