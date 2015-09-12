import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule

detect_features_module = SourceModule("""   

    __global__
    void detectFeatures(bool* mask, float* bottom, float* middle, float* top, float* auxParameters) {
        float threshold = auxParameters[0],
              scaleBottom = auxParameters[1],
              scaleTop = auxParameters[2],
              distThreshold = auxParameters[3];
        
        int rowSize = gridDim.y,
            x = blockIdx.y,
            y = blockIdx.x,
            colSize = gridDim.x;
            
        int scaledBottomX = (int)(scaleBottom*x),
            scaledBottomY = (int)(scaleBottom*y),
            scaledTopX = (int)(scaleTop*x),
            scaledTopY = (int)(scaleTop*y);
              
        int currentIdx = y*rowSize + x;
        float currentValue = middle[currentIdx];
        
        bool isMax = true;
        
        if (abs(currentValue) <= threshold ||
            (distThreshold > x) || (x > (rowSize-distThreshold)) || 
            (distThreshold > y) || (y > (colSize-distThreshold))) 
        {
            isMax = false;
        } 
        else {
            for (int i = -1; i < 2; i++) {
                bool breakExternalLoop = false;
                for (int j = -1; j < 2; j++) {
                    int bottomIdx = (scaledBottomY+i)*rowSize + scaledBottomX + j,
                        middleIdx = (y+i)*rowSize + x + j,
                        topIdx = (scaledTopY+i)*rowSize + scaledTopX + j;  
                                        
                    if (currentValue < bottom[bottomIdx] ||
                        currentValue < middle[middleIdx] ||
                        currentValue < top[topIdx]) 
                    {
                        isMax = false;
                        breakExternalLoop = true;
                        break;
                    }
                }
                if (breakExternalLoop) break;
            }
        }
        
        mask[currentIdx] = isMax;
    }
"""
)

def detect_features(bottom, middle, top, threshold, scale_b, scale_t, dist_threshold):
    b_f32 = bottom.astype(np.float32)
    m_f32 = middle.astype(np.float32)
    t_f32 = top.astype(np.float32)    
    grid_dims = m_f32.shape[::-1]
    mask = np.empty_like(middle, dtype=np.bool)
    aux_parameters = np.array([threshold, scale_b, scale_t, dist_threshold], dtype=np.float32)
    cuda_detect_features = detect_features_module.get_function('detectFeatures')
    cuda_detect_features(
        drv.Out(mask), drv.In(b_f32), drv.In(m_f32), drv.In(t_f32),
        drv.In(aux_parameters), block=(1, 1, 1), grid=grid_dims
    )    
    return np.where(mask)