---
license: mit
language:
- en
metrics:
- accuracy
results:
      - task:
          type: image-classification
        metrics:
          - name: Accuracy
            type: VGG
            value: 0.725
          - name: Accuracy
            type: Resnet18
            value: 0.901
pipeline_tag: image-classification
tags:
- tanks
---

# Russian Tank Destroyed or Not Classifier
Binary Image Classifier for whether a tank is destroyed or not using images from <a href = https://www.oryxspioenkop.com/2022/02/attack-on-europe-documenting-equipment.html>Oryx's</a> collection of confirmed Russian armor losses.

Two models were trained:
1. VGG
2. Finetuned Resnet18

## Model Description and Architecture
Models were trained in Pytorch. VGG was built from scratch while resnet18 was finetuned on pytorch's base resnet18 model with weights from IMAGENET1K_V1. Documentation for pytorch's resnet18 can be found <a href = https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>here</a>.

### VGG Model Architecture:
```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(5, 5))
  (classifier): Sequential(
    (0): Linear(in_features=3200, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=256, out_features=1, bias=True)
    (7): Sigmoid()
  )
)
```

## Training Details

### Dataset:
Training data was sourced from a <a href = https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx>kaggle </a>dataset compiling images from Oryx's site. Only Russian tank images were used for training.

Image filenames in the original dataset contain a list of tags: ['destroyed', 'captured', 'abandoned']. Some images have multiple tags as a result of a vehicle being for example both abandoned then captured, or as a result of an image having multiple vehicles. Images with multiple vehicles or no tags were excluded. Single vehicle images with multiple tags were considered not destroyed if the destroyed tag was not present in the image filename.

After data cleaning and labeling, there were 487 destroyed tanks and 228 not destroyed tanks. These images were split 85/15 into a training and validation set. Imbalanced class representation was by using a weighted random sampler that overweighted the minority "not destroyed" class.

Images were resized and manipulated using the following transformation:

```
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

composed_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
```

### Model Hyperparameters:
VGG:
* Batch Size = 16
* Epochs = 20
* lr = 1e-3
* dropout_rate = 0.3
* Optimizer = Adam
* Loss = BCELoss

Resnet18:
* Batch Size = 16
* Epochs = 46
* lr = 1e-3
* Optimizer = Adam
* Loss = CrossEntropyLoss

### Results:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63df328115266dd945fc01f4/EsT_MvYdLNDOXVkzRqFxt.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63df328115266dd945fc01f4/srkSSvYqa8erEoVqrAjnc.png)

VGG: 73% Accuracy Validation set

Resnet18: 90% Accuracy Validaton set

Has limited ability to correctly recognize the states of vehicles outside of the training scope.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63df328115266dd945fc01f4/jrVjVzh1Im_OLBSn-IB0G.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63df328115266dd945fc01f4/VIDZfkPJDo6gTk0AIGf0w.png)


### Limitations:
Doesn't have the ability to recognize whether the provided image is a tank or not.
<img src="https://huggingface.co/Dingaling01/russian_tank_destroyed_or_not_cnn/resolve/main/brownie.png" alt="drawing" width="250"/>
