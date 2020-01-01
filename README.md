# EfficientDet: Scalable and Efficient Object Detection, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [EfficientDet](https://arxiv.org/abs/1911.09070) from the 2019 paper by Mingxing Tan Ruoming Pang Quoc V. Le
Google Research, Brain Team.  The official and original: comming soon.


<img src= "./docs/arch.png"/>

# Issues - 
EfficientDet is has a fully convolutional network ( due to BiFPN being multi scale ). So Corner Point sub-network outputs a feature map, need a loss functon for the corner points.
