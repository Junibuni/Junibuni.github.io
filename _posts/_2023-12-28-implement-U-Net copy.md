---
title: U-net 구현 - 자동차 파손 인식
date: 2023-12-28 22:29:00 +0800
categories: [AI]
tags: [semantic segmentation, car damage, ai model, computer vision, image processing, deep learning, neural networks, Python, pytorch, image segmentation, machine learning, object detection, data preprocessing, model training, computer science]
use_math: true
---
# 차량 파손 탐지 모델

## Objective
최근 회사 출장 건으로 인해 공유 차량을 빌려 사용할 일이 잦았다. 공유 차량을 빌려 사용할 때, 사용 전 전/측/후면 모두 사진을 찍어 업로드를 해야한다. 이를 이용하여 차량이 파손되었는지 탐지하기 위해 자동화 된 모델을 만들면 좋을 것 같다고 생각되어 탐구해 보기로 했다.

해당 task는 여러 CV 기법 중 segmentatation 기법에 속한다는 것을 알 수 있다. 사진이 주어졌을 때, 파손 부위의 localization도 가능해야 할 뿐더러, 파손의 종류도 classification도 가능해야 하기 때문에 주요 목적은 segmentation으로 볼 수 있다. Segmentation은 instance segmentation과 semantic segmentation으로 분류된다. Instance segmentation은 같은 클래스로 분류 되더라도 다른 instance로 분류되는 반면, semantic segmentation은 같은 클래스 다른 사물 이여도 하나의 object, 즉 하나의 semantic(의미)로 볼 수 있다. 여기서 각 파손 부위별로 나눌 필요는 없고, 파손의 종류만 구별하면 되기 때문에 해당 task의 objective는 `semantic segmentation`으로 볼 수 있다.

## Model selection
Semantic segmentatinon task에서 널리 쓰이는 상용 모델은 크게 두가지가 있다: U-net과 DeepLabV3+ 모델이 있다. 두 모델을 직접 정량적으로 비교해보지 않아 해당 [문헌](https://www.researchgate.net/publication/349292323_MSU-Net_Multi-Scale_U-Net_for_2D_Medical_Image_Segmentation)에서 정량적으로 비교한 데이터를 참고해 보면, 대부분의 부분에서 U-Net이 미세하게 우수한 성능을 내는 것을 볼 수 있다. 구조적으로도 U-Net이 직관적이기 때문에 U-Net모델을 선택 했다. 다음에 기회가 생긴다면 여러 모델(DeepLab, Unet, SegNet등)의 base모델 구현 후 비교할 예정이다. U-Net에 대한 설명은 이전 (포스트)[../U-Net]에 작성해 두었다.

## Dataset
인터넷에 


#3