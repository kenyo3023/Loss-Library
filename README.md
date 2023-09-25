# Loss-Library

### ArcFace Loss
- $L = -log\frac{e^{s \cdot cos(\theta_{y_i}+m)}}{e^{s \cdot cos(\theta_{y_i}+m)} + \sum^N_{j=1,j\neq y_i} e^{s cos\theta_{y_i}}}$


### SphereFace Loss
- $L = -log\frac{e^{cos(m\theta_{y_i})}}{e^{cos(m\theta_{y_i})} + \sum^N_{j=1,j\neq y_i} e^{cos\theta_{y_i}}}$

expand
- $L = -log\frac{e^{\phi(\theta_{y_i})}}{e^{\phi(\theta_{y_i})} + \sum^N_{j=1,j\neq y_i} e^{cos\theta_{y_i}}}$
    - where $\phi(\theta_{y_i}) = (-1)^k cos(m\theta_{y_i}) - 2k$

### AMSoftmax Loss (Additive Margin Softmax Loss)
- $L = -log\frac{e^{s \cdot (cos\theta_{y_i}-m)}}{e^{s \cdot (cos\theta_{y_i}-m)} + \sum^N_{j=1,j\neq y_i} e^{s cos\theta_{y_i}}}$
