---
title: U-net 구현 - 자동차 파손 인식
date: 2023-12-28 22:29:00 +0800
categories: [AI, Segmentation]
tags: [semantic segmentation, car damage, ai model, computer vision, image processing, deep learning, neural networks, Python, pytorch, image segmentation, machine learning, object detection, data preprocessing, model training, computer science]
use_math: true
---
# 차량 파손 탐지 모델

## Objective
최근 회사 출장 건으로 인해 공유 차량을 빌려 사용할 일이 잦았다. 공유 차량을 빌려 사용할 때, 사용 전 전/측/후면 모두 사진을 찍어 업로드를 해야한다. 이를 이용하여 차량이 파손되었는지 탐지하기 위해 자동화 된 모델을 만들면 좋을 것 같다고 생각되어 탐구해 보기로 했다.

해당 task는 여러 CV 기법 중 segmentatation 기법에 속한다는 것을 알 수 있다. 사진이 주어졌을 때, 파손 부위의 localization도 가능해야 할 뿐더러, 파손의 종류도 classification도 가능해야 하기 때문에 주요 목적은 segmentation으로 볼 수 있다. Segmentation은 instance segmentation과 semantic segmentation으로 분류된다. Instance segmentation은 같은 클래스로 분류 되더라도 다른 instance로 분류되는 반면, semantic segmentation은 같은 클래스 다른 사물 이여도 하나의 object, 즉 하나의 semantic(의미)로 볼 수 있다. 여기서 각 파손 부위별로 나눌 필요는 없고, 파손의 종류만 구별하면 되기 때문에 해당 task의 objective는 `semantic segmentation`으로 볼 수 있다.

해당 코드는 [github](https://github.com/Junibuni/Study/tree/master/Unet)에 업로드 해두었다.

## Model selection
Semantic segmentatinon task에서 널리 쓰이는 상용 모델은 크게 두가지가 있다: U-net과 DeepLabV3+ 모델이 있다. 두 모델을 직접 정량적으로 비교해보지 않아 해당 [문헌](https://www.researchgate.net/publication/349292323_MSU-Net_Multi-Scale_U-Net_for_2D_Medical_Image_Segmentation)에서 정량적으로 비교한 데이터를 참고해 보면, 대부분의 부분에서 U-Net이 미세하게 우수한 성능을 내는 것을 볼 수 있다. 구조적으로도 U-Net이 직관적이기 때문에 U-Net모델을 선택 했다.

## Dataset
여러 데이터셋을 찾던 중, `roboflow`에서 제공된 [car damage dataset](https://universe.roboflow.com/projet-ai/car-damage-v3)을 선택했다. 다른 데이터셋보다 제공되는 데이터의 양이 많았고(약 2400장), 차의 파손 종류도 함께 분류하고 싶었기 때문에 5개의 class가 있는 해당 데이터셋을 선택했다. 5가지의 class는 다음과 같다.

- Dent
- Glass Break
- Scratch
- Smash
- Background

데이터셋의 형식은 JSON 형식으로 제공되며, 크게 5가지 정보로 이루어져 있다. 

- Info: 데이터셋의 생성 정보
- License: 이미지 파일들의 라이센스 정보
- Categories: 카테고리의 ID, 이름, sub-category

```json
"categories": [
        {
            "id": 0,
            "name": "car-damages",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "dent",
            "supercategory": "car-damages"
        },
        {
            "id": 2,
            "name": "glass_break",
            "supercategory": "car-damages"
        },
        {
            "id": 3,
            "name": "scratch",
            "supercategory": "car-damages"
        },
        {
            "id": 4,
            "name": "smash",
            "supercategory": "car-damages"
        }
]
```

- Images: 이미지들의 고유 ID, 파일명, width, height정보

```json
"images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "43_jpg.rf.0ace9ba486923c24670f7cf0aa54b67f.jpg",
            "height": 640,
            "width": 640,
            "date_captured": "2023-05-12T07:03:37+00:00"
        },
]
```

- Annotations: Image ID, 여러개의 object ID , bbox, segmentation 등 이미지의 상세 정보

### Data Pre-processing
Coco 데이터셋은 train(70%), valid(20%) 그리고 test(10%)으로 나눠져 있다. Coco dataset을 쉽게 조작 할 수 있는 `pycocotools`를 사용하여 id, annotations, masks등 불러오는 기능을 구현해 테스트 해보았다.

| ![]({{page.img_pth}}img_with_mask.png) | ![]({{page.img_pth}}only_mask.png) |


랜덤으로 데이터를 선택하여 image with mask와 mask label을 그려보았다. Mask 이미지를 보면 해당 사진은 2(glass-break)와 4(smash) 정보가 있는 것을 확인 할 수 있다. 추후에 model output을 보기 쉽게 하기 위해 각 카테고리를 특정 색에 매핑을 해주었다. 초록, 파랑, 빨강, 노랑 (background 제외)순으로 1, 2, 3, 4번째 카테고리와 매핑시켜주었다. 


| ![]({{page.img_pth}}unet_train_bar.png) | ![]({{page.img_pth}}unet_valid_bar.png) | ![]({{page.img_pth}}unet_test_bar.png) |

각 카테고리별 데이터 분포는 비교적 고르게 분포해 있었다. 하지만 segmentation task는 다른 task와 다르게 픽셀 단위로 예측을 진행하기 때문에 각 픽셀이 속하는 클래스의 분포도를 출력해 보았다.

<img src="{{page.img_pth}}unet_pixel_bar.png" width="380">

Background의 비중은 약 82%를 차지하고, 나머지는 각각 6%, 3%, 2%, 7%의 비중을 차지한다. 이와 같은 class 불균형은 몇가지 문제의 원인이 될 수 있다. 해당 케이스에서 background 비중이 80%이상을 차지하기 때문에, 대부분의 예측값은 학습에 기여하지 않는, 혹은 쉽게 예측 가능한 easy negative이므로, 비효율적인 학습이 될 수 있다. 또한 background 비중이 압도적으로 크기 때문에, 전체 loss 또는 gradient를 계산할 때 easy negative의 영향이 커지게 된다. 이러한 상황을 인지하고 추후에 cross entropy에 가중치를 부여하는 `focal loss`도 하나의 옵션으로 둔다.

일반적인 `Cross Entropy`는 모델이 잘 예측한 것보다 잘못 예측한 것에 대해 더 큰 페널티를 주는 방식으로 작동한다. 잘 예측하면 보상은 없고 페널티도 없지만, 잘못 예측하게 되면 보상은 없고 페널티가 커지게 된다. `Focal Loss`는 cross entropy 항에 가중치 텀을 추가하여 easy example의 가중치를 줄이고, hard negative example의 학습에 초점을 맞추는 역할을 한다.

\\[
    \text{FL}(p_t)=-\alpha_t(1-p_t)^\gamma \text{log}(p_t)
\\]

위 식을 보면 \\(\gamma = 0\\)일 때 `Cross Entropy Loss`와 같아짐을 알 수 있고, \\(\gamma\\)가 커질수록 easy example에 대한 loss가 줄어듦을 알 수 있다. 잘못 분류되어 \\(p_t\\)가 작아지게 되면, 가중치 (focusing parameter)가 1에 가까워지게 되고, \\(\text{log}(p_t)\\)가 커지게 되어 loss가 커지게 된다. 반대로 \\(p_t\\)가 올바를 예측을 할 경우 1에 가까워지게 되며, focusing parameter가 0에 가깝게 되고 \\(\text{log}(p_t)\\) 값은 줄어들게 된다. 따라서 앞의 focusing parameter는 easy example에 대한 loss 비중을 낮추는 역할을 함을 알 수 있다. 요약하자면, focal loss는 잘맞추는 클래스는 비교적 신경쓰지 않도록 하고 잘못 맞추는 케이스에 대해서 학습을 중점적으로 진행한다. 

### DataLoader
Coco 커스텀 데이터를 이용하기 때문에, 데이터로더를 만들어야한다. `pycocotools`를 사용하여 annotation, Id, image를 읽어와 데이터를 불러오는 모듈을 구현했다. 이때, train set에 대해서 mean과 std를 구하여 데이터를 normalize처리를 해 주었다. Train set의 RGB 채널에 대한 mean 과 std는 [0.5735, 0.5618, 0.5681], [0.2341, 0.2348, 0.2343]로 계산되었다. Mask의 경우, 클래스 개수만큼 one-hot encoded 채널을 만들지 않고, 클래스 인덱스를 이용해 하나의 채널로 만들어주었다. `poly_to_mask`의 코드는 아래와 같이 작성했다.

```python
def poly_to_mask(self, img_info, anns, merge=True):
        anns_img = np.zeros((img_info['height'],img_info['width']))
        for ann in anns:
            anns_img = np.maximum(anns_img, self.coco.annToMask(ann)*ann['category_id'])
        return torch.tensor(anns_img, dtype=torch.int64)
```

배치 단위로 묶이는 사진들을 모두 같은 크기로 만들어주어야 배치크기의 텐서로 만들 수 있다. 이미지 크기가 다르다면 하나의 텐서로 묶일 수 없기 때문에 두가지 방법을 고안해냈다. 첫번째로, image와 mask 모두 배치 단위에서 하나의 사이즈로 resize 하는 것이다. 가장 간편한 방법이지만, resize를 함으로서 segmentaion polygon이 담고 있는 정보의 손실이 생기기 때문에 해당 방법을 사용하지 않았다. 두번째 방법으로는, 배치단위에서 가장 큰 이미지의 사이즈를 추출한 뒤, 나머지 데이터를 사이즈에 맞게 padding 해주는 것이다. 모든 segmentation의 해상도(정보)를 잃지 않으며 배치 단위로 사이즈를 통일화 시켜줄 수 있어, 해당 방법을 선택했다. 구현한 코드는 아래와 같다. 

```python
def torch_divmod(dividend:torch.Tensor, divisor:int):
    quotient = dividend.div(divisor, rounding_mode="floor")
    remainder = torch.remainder(dividend, divisor)
    return quotient, remainder

def collate_fn(batch):
    images, masks = zip(*batch)

    max_height = max(img.size()[1] for img in images)
    max_width = max(img.size()[0] for img in images)

    pad_height = max_height - torch.tensor([img.size()[1] for img in images])
    pad_width = max_width - torch.tensor([img.size()[0] for img in images])

    h_quotient, h_remainder = torch_divmod(pad_width, 2)
    v_quotient, v_remainder = torch_divmod(pad_height, 2)

    pad_left = h_quotient
    pad_right = h_quotient + h_remainder
    pad_top = v_quotient
    pad_bottom = v_quotient + v_remainder

    padded_images = []
    padded_masks = []

    for i, image_mask in enumerate(zip(images, masks)):
        img, mask = image_mask
        padding = (pad_left[i], pad_top[i], pad_right[i], pad_bottom[i])
        padded_img = TF.pad(img, padding, fill=0)
        padded_mask = TF.pad(mask, padding, fill=0)

        padded_images.append(padded_img)
        padded_masks.append(padded_mask)

    return torch.stack(padded_images), torch.stack(padded_masks)
```

데이터 로더 작성 후, pytorch lighting의 `LightningDataModule`로 감싸주었다.

## Implementation / Result
자동차 파손 부위 예측이라는 task를 하기 위해 Unet 기반 다양한 backbone을 사용하여 결과를 비교해 보았다.

### Unet
우선, vanila Unet으로 모델을 구현하여 테스트 해 보았다. 모델은 64, 128, 256, 512, 1024 채널을 가지는 contracting path를 가지고 있고, 반대로 올라오는 expanding path를 가지고 있다. 각 모듈마다 skip connection을 이용하여 contracting과 expanding path를 연결해 주었다. Pretrain 되어있지 않은 모델이라 learning rate를 처음엔 조금 크게 사용하였다. 아래는 `lr=1e-3`, `cross_entropy`, `Adam`을 사용한 결과이다.

```yaml
backbone_name: unet
criterion_params: {}
in_channels: 3
loss_fn: crossentropy
num_classes: 5
optim_params:
  betas: !!python/tuple
  - 0.9
  - 0.999
  lr: 0.001
  weight_decay: 0.0001
optimizer: !!python/name:torch.optim.adam.Adam ''
```

| <img src="{{page.img_pth}}vanila_unet_trainloss.png"> | <img src="{{page.img_pth}}valid_dice_unet.png"> |

<img src="{{page.img_pth}}target_mask_unet_project.png">
*Target Mask*

<img src="{{page.img_pth}}predmask_unet.png">
*Predicted Mask*

Metric으로는 `dice`, `f1`, `acc`, `precision`, `recall`, `jaccard`를 사용하여 로깅했다. 하지만 해당 실험의 경우 모든 metric이 좋지 않게 나왔다. 또한 training loss와 validation loss는 학습함에 따라 감소하기는 하지만 이상적으로 감소하지 않으며, loss의 노이즈가 심하다. 정량적인 값들로 예상했듯이 predicted mask를 보면 위치는 대략적으로 캐치해 내지만, 모양이나 클래스는 매우 부정확한것을 볼 수 있다. Train rate를 `1e-4`로 조금 낮추어 학습을 해 보았다. 

<img src="{{page.img_pth}}unet_losse-4_trainloss.png" width="480">

전보다 감소하는 경향이 명확해졌지만 아직도 loss의 노이즈가 심하고, metrics와 predicted mask의 결과도 개선이 없었다. 마지막으로, learning rate scheduling을 해줌으로서 조금이라도 성능에 향상이 있는지 검토해 보았다. LR scheduler로는 `ReduceLROnPlateau`를 사용하였다. 특별한 이유는 없고, metric이나 validation loss의 향상(감소)가 없을 시 lr을 조금씩 줄여나가는 스케쥴러를 사용하기 위해 선택했다. LR scheduler가 다양하게 있어([블로그](https://sanghyu.tistory.com/113)) 추후에 추가적으로 스터디를 진행해 보아야 겠다. 모든 파라미터를 default로 설정한뒤 진행했다. 이후의 실험은 모두 같은 lr scheduler를 사용하였다. 아래는 스케쥴러를 사용한 뒤의 결과이다.

<img src="{{page.img_pth}}compareallunetloss.png" width="480">

스케쥴러를 사용한 뒤의 결과가 크게 다르지 않았다. Vanila Unet은 기존 문헌에서 흑백채널의 이미지에 대한 실험이였고, 세포의 테두리를 탐지하는 비교적 간단한 테스크였다. 하지만 해당 데이터는 RGB 차원을 가지는 이미지에 대한 테스크이며, mask의 모양도 비교적 복잡하기 때문에 vanila unet의 레이어가 너무 얕거나 피쳐맵의 채널수가 부족하다고 느꼈다. 이 실험에서 validation Loss 자체가 감소하지 않는 현상의 문제보다 train loss의 진동이 심하고 predicted mask의 성능이 좋지 않아 pretrained된 `vgg19`, `resnet50`, `efficientnetb0` backbone을 사용하여 학습을 해보았다. 깊은 레이어를 가진 구조이기도 하며, 더욱 더 세부적인 피처를 추출할 수 있기 때문에 위 모델을 선택했다.

### VGG vs Resnet vs Efficientnet
`vgg19`, `resnet50`, `efficientnetb0` 세가지 모델 모두 convolutional layer를 사용하여 feature extraction을 한다는 특징이 있다. 이 과정에서 인풋 영상의 크기가 반으로 줄어드는 레이어가 있는데, 이를 이용하여 Unet의 backbone으로 사용이 가능하다. 아래는 `torchvision.models`에서 데이터의 크기가 반으로 줄어드는 레이어를 추출하기 위해 저장한 dictionary이다. 예를 들어 `efficientnetb0`모델은 `features.1`, `features.2`, `features.3`, `features.5`, `features.7` 레이어에서 데이터의 사이즈가 반으로 줄어들고, 각 레이어는 `features.7`레이어 부터 [320, 112, 40, 24, 16]채널을 가지게 된다(reversed).

```python
@property
def layer(self):
    #skip1, skip2, skip3, skip4, bottleneck
    layer_concat = {
        "unet": ["module1", "module2", "module3", "module4", "module5"],
        "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
        "efficientnetb0": ["features.1", "features.2", "features.3", "features.5", "features.7"],
        "vgg19": ["12", "25", "38", "51", "52"],
    }
    return layer_concat[self.backbone_name].copy()

@property
def size(self):
    #channel size of: bottleneck, skip4, 3, 2, 1
    size_dict = {
        "unet": [1024, 512, 256, 128, 64],
        "resnet50": [2048, 1024, 512, 256, 64],
        "efficientnetb0": [320, 112, 40, 24, 16],
        "vgg19": [512, 512, 512, 256, 128],
    }
    return size_dict[self.backbone_name].copy()
```

여기서, 각 모델을 backbone으로 불러오고, Unet과 결합하기 위해 다듬어주어야 한다. 위에서 언급 된 레이어들을 추출해야 하고, 원하는 마지막 layer까지 학습 시키도록 설정해야 한다. 처음엔 아래와 같이 `hook` 방법을 사용하여 구현하였다. 해당 방법은 미리 설정해 둔 layer의 결과값을 출력하여 Unet의 skip connection을 가능하게 할 수 있었지만, 원하지 않는 마지막 레이어들은 모두 삭제할 수 없었다. 수동적으로 각 backbone 모델마다 `nn.Identity()`를 사용하여 불필요한 layer를 삭제 할 수 있지만, 다양한 backbone에 대해서 일반화를 시키고자 했기 때문에 리서치를 진행했다. 그러던 중, `torchvision.models.feature_extraction`의 `create_feature_extractor`모듈을 사용하면 원하는 layer를 추출 할 수 있을 뿐만 아니라, 원하는 layer까지 학습이 가능하게 할 수 있었다. 해당 모듈은 모델과 추출하고 싶은 layer를 list로 받고, torch 내부적으로 hook를 걸어 모델을 재건축 하는 것으로 이해했다. 이를 사용하여 원하는 layer를 추출할 수 있음과 동시에 원하는 깊이까지 학습 시키는 방법을 간단하게 구현 할 수 있었다.

```python
def extract_features(model, input_tensor, layer_names):
        extracted_features = {}

        def register_hook(name):
            def hook(module, input, output):
                extracted_features[name] = output
            return hook
        
        hook_handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook_handles.append(module.register_forward_hook(register_hook(name)))

        model(input_tensor)

        for handle in hook_handles:
            handle.remove()

        return extracted_features
```

일단 정량적으로 비교해보면 결과는 아래와 같다. 왼쪽 위부터 오른쪽 아래까지 `UNet`, `VGG19`, `ResNet50`, `EfficientNetb0`모델의 학습 결과이다.

### Resnet with FocalLoss

unet_losse-4_trainloss
random batchidx, stack target and predict, save image

lr pretrained
lr 줄여보고 loss 바꿔보고

모델을 구현 할때 힘들었던점: hook, feature extraction, 모델 안쓰는 부분 자르기
coco dataset 처리하는법
segmentation의 mask는 어떻게 작동하나?

segmentation에 쓰이는 loss들
- ce
- miou
- focal

나머지도 있지만 이렇게 비교하는이유는?
CE는 틀린거에대해서 harsh?한가? https://woochan-autobiography.tistory.com/929
focal 의 장점, 각 loss의 장단점, 선택한 이유 (data 각 class별 분포도 추출)

모델은 백본을 선택한 이유? 일단 vanila unet은 왜 안되는지? featuremap inspect!
백본 써서 좋은이유?
백본 선택한 이유

모델 평가
평가방법 나열

