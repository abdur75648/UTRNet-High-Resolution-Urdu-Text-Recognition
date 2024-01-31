# UTRNet: High-Resolution Urdu Text Recognition

[![UTRNet](https://img.shields.io/badge/UTRNet:%20High--Resolution%20Urdu%20Text%20Recognition-blueviolet?logo=github&style=flat-square)](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
[![Website](https://img.shields.io/badge/Website-Visit%20Here-darkgreen?style=flat-square)](https://abdur75648.github.io/UTRNet/)
[![arXiv](https://img.shields.io/badge/arXiv-2306.15782-darkred.svg)](https://arxiv.org/abs/2306.15782)
[![SpringerLink](https://img.shields.io/badge/Springer-Page-darkblue.svg)](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_19)
[![SpringerLink](https://img.shields.io/badge/Springer-PDF-blue.svg)](https://rdcu.be/dkbIF)
[![Demo](https://img.shields.io/badge/Demo-Online-brightgreen.svg)](https://abdur75648-urduocr-utrnet.hf.space)

**Official Implementation of the paper *"UTRNet: High-Resolution Urdu Text Recognition In Printed Documents"***

The Poster:

![P2 49-poster](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition/assets/66300465/dea7c7a6-5e13-400f-8ba7-8356a794897d)


## Using This Repository
### Environment
* Python 3.7
* Pytorch 1.9.1+cu111
* Torchvision 0.10.1+cu111
* CUDA 11.4

### Installation
1. Clone the repository
```
git clone https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git
```

2. Install the requirements
```
conda create -n urdu_ocr python=3.7
conda activate urdu_ocr
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Running the code

1. Training
```
python3 train.py --train_data path/to/LMDB/data/folder/train/ --valid_data path/to/LMDB/data/folder/val/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --exp_name UTRNet-Large --num_epochs 100 --batch_size 8

```

2. Testing
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data path/to/LMDB/data/folder/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

3. Character-wise Accuracy Testing
* To create character-wise accuracy table in a CSV file, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 char_test.py --eval_data path/to/LMDB/data/folder/test/ --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

* Visualize the result by running ```char_test_vis```

4. Reading individual images
* To read a single image, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 read.py --image_path path/to/image.png --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/UTRNet-Large/best_norm_ED.pth
```

5. Visualisation of Salency Maps

* To visualize the salency maps for an input image, run the following command

```
python3 vis_salency.py --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/UTRNet-Large/best_norm_ED.pth --vis_dir vis_feature_maps --image_path path/to/image.pngE
```

### Dataset
1. Create your own lmdb dataset
```
pip3 install fire
python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/train
```
The structure of data folder as below.
```
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
test/word_1.png label1
test/word_2.png label2
test/word_3.png label3
...
```

# Downloads
## Trained Models
1. [UTRNet-Large](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EeUZUQsvd3BIsPfqFYvPFcUBnxq9pDl-LZrNryIxtyE6Hw?e=MLccZi)
2. [UTRNet-Small](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EdjltTzAuvdEu-bjUE65yN0BNgCm2grQKWDjbyF0amBcaw?e=yiHcrA)

## Datasets
1. [UTRSet-Real](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EXRKCnnmCrpLo8z6aQ5AP7wBN_NujFaPuDPvlOB0Br8KKg?e=eBBuJX)
2. [UTRSet-Synth](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EUVd7N9q5ZhDqIXrcN_BhMkBKQuc00ivNZ2_jXZArC2f0g?e=Gubr7c)
3. [IIITH (Updated)](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EXg_48rOkoJBqGpXFav2SfYBMLx18zzgQOtj2kNzpeL4bA?e=ef7lLr) ([Original](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-urdu-ocr))
4. [UPTI](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EVCJZL8PRWVJmRfhXSGdK2ABR17Jo_lW5Ji62JeBBevxcA?e=GgYC8R) ([Source](https://ui.adsabs.harvard.edu/abs/2013SPIE.8658E..0NS/abstract))
5. UrduDoc - Will be made available subject to the execution of a no-cost license agreement. Please contact the authors for the same.

## Text Detection (Supplementary)
The text detection inference code & model based on ContourNet is [here](https://github.com/abdur75648/urdu-text-detection). As mentioned in the paper, it may be integrated with UTRNet for a combined text detection+recognition and hence an end-to-end Urdu OCR.

## Synthetic Data Generation using Urdu-Synth (Supplementary)
The [UTRSet-Synth](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EUVd7N9q5ZhDqIXrcN_BhMkBKQuc00ivNZ2_jXZArC2f0g?e=Gubr7c) dataset was generated using a custom-designed robust synthetic data generation module - [Urdu Synth](https://github.com/abdur75648/urdu-synth/). 

## End-To-End Urdu OCR Webtool
This tool was developed by integrating the UTRNet (https://abdur75648.github.io/UTRNe) with a text detection model ([YoloV8](https://docs.ultralytics.com/) finetuned on [UrduDoc](https://paperswithcode.com/dataset/urdudoc)) for end-to-end Urdu OCR.

The application is deployed on Hugging Face Spaces and is available for a live demo. You can access it *[here](https://abdur75648-urduocr-utrnet.hf.space)*. If you prefer to run it locally, you can clone its repository and follow the instructions given there - [Repo](https://github.com/abdur75648/End-To-OCR-UTRNet).

> **Note:** *This version of the application uses a YoloV8 model for text detection. The original version of UTRNet uses ContourNet for this purpose. However, due to deployment issues, we have opted for YoloV8 in this demo. While YoloV8 is as accurate as ContourNet, it offers the advantages of faster processing and greater efficiency.*

![website](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition/assets/66300465/511aeffe-d9b3-41aa-8150-ab91f398ae49)


# Updates
* 01/01/21 - Project Initiated
* 21/11/22 - Abstract accepted at [WIDAFIL-ICFHR 2022](https://icfhr2022.org/wtc.php)
* 12/12/22 - Repository Created
* 20/12/22 - Results Updated
* 19/04/23 - Paper accepted at [ICDAR 2023](https://icdar2023.org/)
* 23/08/23 - Poster presentation at [ICDAR 2023](https://icdar2023.org/)
* 31/08/23 - Webtool made available
* 31/01/24 - Updated Webtool (with YoloV8) made available via HuggingFace [here](https://abdur75648-urduocr-utrnet.hf.space/)

# Acknowledgements
* This repository is based on [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
* We acknowledge the [Rekhta Foundation](https://rekhtafoundation.org/) and the personal collections of [Arjumand Ara](https://du-in.academia.edu/ArjumandAra) for the scanned images and Noor Fatima and Mohammed Usman for the dataset and the manual transcription.
* We also thank all the members of the [Computer Vision Group, IIT Delhi](https://vision-iitd.github.io/) for their support and guidance.

## Contact
* [Abdur Rahman](https://www.linkedin.com/in/abdur-rahman-0b84341a0/)
* [Prof. Arjun Ghosh](https://web.iitd.ac.in/~arjunghosh/)
* [Prof. Chetan Arora](https://www.cse.iitd.ac.in/~chetan/)

## Note
This is an official repository of the project. The copyright of the dataset, code & models belongs to the authors. They are for research purposes only and must not be used for any other purpose without the author's explicit permission.

## Citation
If you use the code/dataset, please cite the following paper:

```BibTeX
@InProceedings{10.1007/978-3-031-41734-4_19,
		author="Rahman, Abdur
		and Ghosh, Arjun
		and Arora, Chetan",
		editor="Fink, Gernot A.
		and Jain, Rajiv
		and Kise, Koichi
		and Zanibbi, Richard",
		title="UTRNet: High-Resolution Urdu Text Recognition in Printed Documents",
		booktitle="Document Analysis and Recognition - ICDAR 2023",
		year="2023",
		publisher="Springer Nature Switzerland",
		address="Cham",
		pages="305--324",
		isbn="978-3-031-41734-4",
		doi="https://doi.org/10.1007/978-3-031-41734-4_19"
}
```

### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/). This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial (academic & research) purposes only and must not be used for any other purpose without the author's explicit permission.
