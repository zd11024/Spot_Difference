# Visual Dialog for Spotting the Differences between Pairs of Similar Images

## Directory Structure

```
Spot_Difference
|-- bottom-up-attention-vqa
|-- checkpoints
		|-- pretrained
				|-- bert-base-uncased
				|-- gpt2
				|-- model_LXRT.pth
		|-- ...
|-- data
		|-- 0206
				|-- spot_diff_train.json
				|-- ...
		|-- img_feat_3ee94.h5
|-- dataloader
		|-- guesser_dataloader.py
		|-- loader_utils.py
		|-- qgen_dataloader.py
|-- lxmert
		|-- ...
|-- model
		|-- guesser.py
		|-- qgen.py
|-- scripts
|-- stat_tools
|-- ...
```

## Pre-Process

### Environment
Setup the environment by running `pip install -r requirements.txt`.

### Pre-Trained Model

1. BERT
2. GPT-2
3. LXMERT: could be download in https://github.com/airsplay/lxmert.

The pre-trained model should be put in checkpoints/pretrained.

### SpotDiff Dataset

1. SpotDiff dialogues: three JSON file, i.e., spot_diff_train.json, spot_diff_val.json, spot_diff_test.json. You could download these files from [Baidu Netdisk](https://pan.baidu.com/s/1bAwEReZkj5gIN2rTMh9hRg?pwd=9hqm).
2. SpotDiff images
- You could download the original images from my [Baidu Netdisk](https://pan.baidu.com/s/1XANouRqIX2DaUL2MUq4Dvg?pwd=t38d).
- Due to the large size of images, I compressed it into four files. You should download these files to your local device and then proceed to merge and decompress them.
- Considering the original image collection is too large, you can only use a subset of it.
1. Image features: are extrated by bottom-up top-down attention. The extracted features could be downloaded [here](https://pan.baidu.com/s/16jC7kbPcAGi3JkdZWXkAiQ?pwd=if64). We extracted butd features by running the code [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch).

## Training

require to modify <work_dir> and <img_feat_file> in the following scripts.

* <work_dir>: the project directory.
* <img_feat_file> the h5 file that contains image features, data/img_feat_3ee94.h5

### QGen

GPT and LXMERT-based VQG model 

```sh
sh scripts/train_<vqg_model_type>_vqg.sh
```

* <vqg_model_type>: gpt, lxrt

### A-Bot

BUTD and LXMERT-based VQA model

```sh
sh scripts/train_<vqa_model_type>_vqa.sh
```

* <vqa_model_type>: butd, lxrt

### Guesser

- Bert-based Guesser

```sh
sh scripts/train_guesser.sh
```

## Evaluation

```sh
sh scripts/self_play_{vqg_model_type}_{vqa_model_type}.sh
```

* <vqg_model_type>: gpt, lxrt
* <vqa_model_type>: butd, lxrt

