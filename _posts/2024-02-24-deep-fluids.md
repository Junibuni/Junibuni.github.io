---
title: [논문리뷰] Deep Fluids
date: 2024-02-24 11:46:00 +0800
categories: [AI, Application]
tags: [ai, deep learning, generative Model, fluid dynamics, graphics, simulation]
use_math: true
---
## AI-CFD
최근 들어 classical simulation 을 가속화 하려는 많은 시도가 있었다. 차원 축소 방법인 Proper Orthogonal Decomposition (POD), Reduced Order Mehod (ROM) 이 그동안 많이 쓰였지만, 두 기법이 사용하는 basis function은 일반적으로 선형 차원축소 방법이다. 선형적인 representation을 비선형적인 manifold 에 올려놓는 작업을 하게 되면 "explosion of subspace dimensionality" 현상이 일어날 수 있다. 해당 문제를 해결하고자 iterative convolution network를 이용하여 효과적으로 데이터를 압축하고, 비선형적인 데이터를 다룰수 있도록 제시한 문헌을 찾아 정리해 본다.

문헌: [Deep Fluids: A Generative Network for Parameterized Fluid Simulations](https://arxiv.org/pdf/1806.02071)

## Introduction
### Traditional Machine Learning / Deep learning
Random forest 모델과 같이 machine learning기법은 정교하게 다듬어진 feature(특징)을 필요로 하기 때문에 일반성이 떨어지게 된다. 딥러닝 방법을 사용해 하나의 time step에 대한 압력장을 구하기 위해 CNN 기법을 사용
여러 time step에 대해 압력장을 예측하는데 LSTM 기반 방법을 사용
위와 같은 기법은 pressure projection 단계에서만 작동하기 때문에 divergence-free 를 만족시키지 못함
여러 ML/DL 연구가 진행되었지만 모두 initial condition 을 받아 하나의 steady state solution을 예측할 뿐 시뮬레이션은 불가능함
또한 모든 연구가 2D에서 진행 됨


---
참고자료
- *<Ladický, L'ubor, et al. "Data-driven fluid simulations using regression forests." ACM Transactions on Graphics (TOG) 34.6 (2015): 1-9.>*