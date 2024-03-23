---
title: Pytorch의 Autograd Engine
date: 2024-03-20 13:37:00 +0800
categories: [AI, Pytorch]
tags: [pytorch, autograd, deep learning, machine learning, neural networks, gradient descent, backpropagation, computational graphs, pytorch tutorial]

use_math: true
---

## AutoGrad

### Backpropagation
딥러닝은 input값과 output값을 비교하며 둘의 차이만큼 연산식(parameter)를 수정해 나가는 과정이다. 이를 가능하게 하기 위해서 각 paremeter가 loss function에 얼마나 민감하게 반응하는지 역으로 계산하여 업데이트 해주게 된다. 

예를 들어, 하나의 노드만 있는 네트워크를 상상해보자.

---
참고자료
- *<https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>*

- *<https://medium.com/konvergen/continuing-on-adaptive-method-adadelta-and-rmsprop-1ff2c6029133>*

- *<http://incredible.ai/artificial-intelligence/2017/04/10/Optimizer-Adadelta/>*

- *<https://www.youtube.com/watch?v=NE88eqLngkg&ab_channel=DeepBean>*