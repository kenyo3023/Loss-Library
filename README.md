# Loss-Library

## Introduction

## Features

[ArcFace Loss](https://arxiv.org/pdf/1801.07698.pdf)
[$
\begin{align}
    L = -log\frac{e^{s \cdot cos(\theta_{y_i}+m)}}{e^{s \cdot cos(\theta_{y_i}+m)} + \sum^N_{j=1,j\neq y_i} e^{s cos\theta_{y_i}}}
\end{align}
$](https://github.com/kenyo3023/Loss-Library/blob/main/arcface.py)


[AMSoftmax Loss (Additive Margin Softmax Loss)](https://arxiv.org/pdf/1801.05599.pdf)
$
\begin{align}
    L = -log\frac{e^{s \cdot (cos\theta_{y_i}-m)}}{e^{s \cdot (cos\theta_{y_i}-m)} + \sum^N_{j=1,j\neq y_i} e^{s cos\theta_{y_i}}}
\end{align}
$

[SphereFace Loss](https://arxiv.org/pdf/1704.08063.pdf)

$
\begin{align}
    L = -log\frac{e^{cos(m\theta_{y_i})}}{e^{cos(m\theta_{y_i})} + \sum^N_{j=1,j\neq y_i} e^{cos\theta_{y_i}}}
\end{align}
$

expand
$
\begin{align}
    L = -log\frac{e^{\phi(\theta_{y_i})}}{e^{\phi(\theta_{y_i})} + \sum^N_{j=1,j\neq y_i} e^{cos\theta_{y_i}}}, {\rm where} \phi(\theta_{y_i}) = (-1)^k cos(m\theta_{y_i}) - 2k
\end{align}
$


[SphereFace2 Loss]()

SphereFace2-C (CosFace-type Additive Margin)
$
\begin{align}
    L &= \frac{\lambda}{r} \cdot log (1 + e^{-(r \cdot(g(cos(\theta_y))-m)+b)}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot(g(cos(\theta_y))+m)+b})
\end{align}
$

SphereFace2-A (ArcFace-type Additive Margin)
$
\begin{align}
    L &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot g(cos(\theta_y)) - r \cdot {\rm Detach}(g (cos(min(\pi, \theta_y + m))) - g(cos  (\theta_y))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g(cos(\theta_y)) + {\rm Detach}(g (cos(min(\pi, \theta_y + m))) - g(cos(\theta_y)))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g(cos(\theta_y)) + g (cos(min(\pi, \theta_y + m))) - g(cos(\theta_y))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g (cos(min(\pi, \theta_y + m))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
\end{align}
$

SphereFace2-M (Multiplicative Additive Margin)
$
\begin{align}
    L &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot g(cos(\theta_y)) - r \cdot {\rm Detach}(g (cos(min(m, \frac{\pi}{\theta_y}) \cdot \theta_y)) - g(cos  (\theta_y))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g(cos(\theta_y)) + {\rm Detach}(g (cos(min(m, \frac{\pi}{\theta_y}) \cdot \theta_y)) - g(cos(\theta_y)))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g(cos(\theta_y)) + g (cos(min(m, \frac{\pi}{\theta_y}) \cdot \theta_y)) - g(cos(\theta_y))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g (cos(min(m, \frac{\pi}{\theta_y}) \cdot \theta_y)) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
      &= \frac{\lambda}{r} \cdot log (1 + e^{-r \cdot (g (cos(min(\pi, \theta_y m))) -b}) + \frac{1 - \lambda}{r} \sum^N_{i \neq y} log (1 + e^{r \cdot g(cos(\theta_y))+b}) \\
\end{align}
$
