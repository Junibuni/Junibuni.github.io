---
title: Convolution 연산의 종류
date: 2023-12-12 19:57:00 +0800
categories: [AI, Deep Learning]
tags: [convolution, image processing, cnn, deep learning, filtering, feature extraction, matrix convolution, stride, padding, convolution layers, computational efficiency, python]

use_math: true
---

## Convolution Layer
영상에 대한 ***2D Spatial Feature*** 를 추출하는 필터들로 구성되어있다. 하나의 커널은 입력 영상의 각 위치별 반응치를 계산하여 ***2D Response Map***을 출력한다. 수학적으로, 각 위치별로 모든 채널의 픽셀들에 대한 Linear Combination을 수행하는 것이다.

### Convolution Layer vs. FC Layer
FC layer 모델은 모든 인풋 데이터가 모든 아웃풋 데이터에 영향을 준다. 하지만 사람은 물체를 인식할때 이렇게 인식하지 않는다. 사람의 눈, 코, 입과 같은 어떤 물체의 특징의 모양과 위치를 파악해서 사물을 인식한다. 따라서 Convolutional Neural Network(CNN)은 다음을 만족한다.
- Spatial locality: 필터는 주변 픽셀(데이터)를 탐색한다
- Positional invariance: 모든 위치에서 같은 필터를 사용한다

### Memory & Time Cost
Convolution 의 연산량은 \\(CHWK^2N\\)가 되고, 파라미터의 수는 \\(N(CK^2+1)\\)개가 된다.
- C,H,W: 채널, Height, Width
- K: 커널의 크기 (K, K)
- N: 커널의 개수 (output channel)

영상 내의 객체를 정확히 판단하기 위해서는 `Contextual Information`이 중요하다. 객체가 주변의 환경은 어떤지, 다른 객체들은 어떤 환경에 놓여 있는지. 많은 모델들은 충분한 contextual information을 확보하기 위해 넓은 receptive field를 고려한다.

넓은 receptive field를 만족시키기 위해서는 커널의 크기를 키우는 방법과 더 깊은(많은) convolutional layer를 쌓는 방법이 있다. 하지만 이 두가지 방법 모두 연산량이 기하급수적으로 증가한다는 단점이 있다.

#### Expensive Cost
네트워크는 점점 깊어지고, 채널의 수는 증가하게 된다. 이렇게 되면 연산량과 가중치도 비례하여 증가한다. 일반적으로 큰 feature map을 convolution의 입력으로 가져가는 것이 유리하지만, 많은 CNN모델은 computational cost를 고려해서 sup-sampling을 수행한다.

채널의 수를 늘릴수로 더 다양한 필터를 학습 할 수 있게 된다. 하지만 너무 많은 채널수는 오히려 파라미터 수 증가, 학습 속도 저하, 오버피팅과 같은 문제가 생길 수도 있다.

#### Dead Chanels
신경망의 학습 과정에서 출력 결과에 영향을 거의 주지 않는 노드가 나타날 수 있다. 이런 경우, 일반적으로 pruning 을 통해 신경망의 연산량과 파라미터 수를 경량화 할 수 있다.

마찬가지로, CNN에서 이런 현상이 채널 단위에서 일어날 수 있다. 하나의 필터에서, 혹은 전체 신경망 기준으로 불필요한 (학습에 영향을 주지 않는) 채널이 생길 수 있다. 채널의 수는 파라미터 수와 연산량에 직결되기 때문에 엄청난 자원 낭비가 될 수 있다. 채널의 개수는 hyperparmeter 이기 때문에 적당한 채널 개수를 찾는것도 한계가 있다.

#### Low Correlation Between Channels
각 필터는 입력 영상의 모든 채널을 사용하지만, 모든 채널-필터 쌍이 높은 correlation을 가지지 않을 수 있다.

<img src="{{page.img_pth}}convlayer.svg">
_A normal convolutional layer. Yellow blocks represent learned parameters, gray blocks represent feature maps/input images (working memory)._

<img src="{{page.img_pth}}cifar-nin-groupanimation.gif" height="450" width="450">
<img src="{{page.img_pth}}colorbar_conv.svg" width="450">
_The correlations between filters of adjacent layers in a Network-in-Network model trained on CIFAR10, when trained with 1, 2, 4, 8 and 16 filter groups._

출처: [https://blog.yani.ai/filter-group-tutorial/](https://blog.yani.ai/filter-group-tutorial/)

위 그림과 같이 sparse 한 correlation matrix를 볼 수 있다. 따라서 학습 수렴 속도가 저하되고, 불필요한 가중치와 불필요한 연산을 하기 때문에, 학습 시 일종의 노이즈 처럼 작용 될 수 있다.

이러한 문제점 때문에 연산량을 최소화 하면서, 정보손실이 일어나지 않도록 다양한 Convolution 기법들이 등장했다.

