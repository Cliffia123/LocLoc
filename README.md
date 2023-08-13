# LocLoc: Low-level Cues and Local-area Guides for Weakly Supervised Object Localization
PyTorch implementation of “LocLoc: Low-level Cues and Local-area Guides for Weakly Supervised Object Localization”.

## 💡 Abstract
Weakly Supervised Object Localization (WSOL) aims to localize objects using only image-level labels while ensuring competitive classification performance. However, previous efforts have prioritized localization over classification accuracy in discriminative features, in which low-level information is neglected. We argue that low-level image representations, such as edges, color, texture, and motions are crucial for accurate detection. That is, using such information further achieves more refined localization, which can be used to promote classification accuracy.
In this paper, we propose a unified framework that simultaneously improves localization and classification accuracy, termed as LocLoc (Low-level Cues and Local-area Guides). It leverages low-level image cues to explore global and local representations for accurate localization and classification. Specifically, we introduce a GrabCut-Enhanced Generator (GEG) to learn global semantic representations for localization based on graph cuts to enhance low-level information based on long-range dependencies captured by the transformer. We further design a Local Feature Digging Module (LFDM) that utilizes low-level cues to guide the learning route of local feature representations for accurate classification.Extensive experiments demonstrate the effectiveness of LocLoc with 84.4(**↑5.2%**) Top-1 Loc., 85.8% Top-1 Cls. on CUB-200-2011 and 57.6% (**↑1.5%**) Top-1 Loc., 78.6% Top-1 Cls. on ILSVRC 2012, indicating that our method achieves competitive performance with a large margin compared to previous approaches.

## 🔑 Requirements
Pytorch>=1.7.0<br>
torchvision>=0.8.1<br>
timm>=0.6.12<br>

## 🎃 Usage

### Start
Download the CUB GrabCut files in [Google Drive](https://drive.google.com/drive/folders/15litgloea5to9qGbgY9pkjNC_WwHRWZw?usp=sharing).

Download the ILSVRC GrabCut files in [Pan.Baidu]()

cd LocLoc

Unfold the GrabCut files folder and put the GrabCut files in datasets

### Datasets
- CUB: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- ILSVRC: https://www.image-net.org/challenges/LSVRC/
```
/path/to/CUB/
    class1/
      img1.jpg
    class2/
      img2.jpg
    class3/
      img3.jpg
```

### Models
You can download the [models](https://drive.google.com/drive/folders/1D9tMZjXqlPVDzRIi_18zRwF4kWwuSfZt?usp=sharing) for evaluation.

### Inference
It is a script example. You can change data_path and models positions according to your directory.
```
python3 main_eval.py \
--eval \
--model deit_small_lctr \
--data-set CUB \
--data-path /data/CUB_200_2011/images \
--resume log/deit_small_patch16_224.pth \
--classifier log/CUB_cls.pth \
--locator log/CUB_loc.pth
```

- To train the CUB/ILSVRC classfier model, you can run:
```
./scripts/train_cub_cls.sh and train_ilsvrc_loc.sh
```

- To train the CUB/ILSVRC generator model, you can run:
```
./scripts/train_cub_loc.sh and train_ilsvrc_loc.sh
```

-  To test the CUB/ILSVRC models, you can run :
```
./scripts/eval_cub.sh and eval_ilsvrc.sh
```
