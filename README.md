# UTRNet: High-Resolution Urdu Text Recognition

[![UTRNet](https://img.shields.io/badge/UTRNet:%20High--Resolution%20Urdu%20Text%20Recognition-blueviolet?logo=github&style=flat-square)](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
[![Website](https://img.shields.io/badge/Website-Visit%20Here-brightgreen?style=flat-square)](https://abdur75648.github.io/UTRNet/)
[![arXiv](https://img.shields.io/badge/arXiv-2306.15782-darkred.svg)](https://arxiv.org/abs/2306.15782)

**Official Implementation of the paper *"UTRNet: High-Resolution Urdu Text Recognition In Printed Documents"***

## Using This Repository
### Environment
* Python 3.7
* Pytorch 1.9.1+cu111
* Torchvision 0.10.1+cu111
* CUDA 11.4

### Installation
1. Clone the repository
```
git clone https://github.com/abdur75648/high-resolution-urdu-text-recognition.git
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
python3 train.py --train_data urdu_data_lmbd/train/real/ --valid_data urdu_data_lmbd/val/ --FeatureExtraction UNet --SequenceModeling DBiLSTM --Prediction CTC --exp_name UNet_DBiLSTM_CTC --num_epochs 100 --batch_size 8 --device_id 3

```

2. Testing
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data path/to/lmdb/data/folder --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/experiment_1/best_norm_ED.pth
```

3. Character-wise Accuracy Testing
* To create character-wise accuracy table in a CSV file, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 char_test.py --eval_data path/to/lmdb/data/folder --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/experiment_1/best_norm_ED.pth
```

* Visualize the result by running ```vis_char_test.py```

4. Reading individual images
* To read a single image, run the following command

```
CUDA_VISIBLE_DEVICES=0 python3 read.py --image_path path/to/lmdb/data/folder/1.png --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model saved_models/experiment_1/best_norm_ED.pth
```

5. Visualisation of Salency Maps

* To visualize the salency maps for an input image, run the following command

```
python3 vis_salency.py --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --saved_model saved_models/experiment_1/best_norm_ED.pth --vis_dir vis_feature_maps --image_path PATH/TO/IMAGE
```

6. Visualisation of CNN Feature Maps

* To visualize the feature maps for an input image, run the following command
    * Add ```from utils import draw_feature_map``` in the beginning of CNN code
    * Add ```draw_feature_map(feature_maps [c,W,H], 'vis_directory', num_channels)``` in the required places in CNN code to visualize the corresponding feature maps

### Dataset
1. Create your own lmdb dataset
```
pip3 install fire
python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/train
python create_lmdb_dataset.py --inputPath data_valid/ --gtFile data_valid/gt.txt --outputPath result/valid
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
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```

# Downloads
1. [Pretrained Model (HRNet-DBiLSTM-CTC)](https://csciitd-my.sharepoint.com/:f:/g/personal/ch7190150_iitd_ac_in/EorhOvQ8q3BLnLqQxtHTztYBReaibafGDOV-B1f4BU9jAQ?e=QceL03)
2. [Datasets](https://csciitd-my.sharepoint.com/:f:/g/personal/ch7190150_iitd_ac_in/EvUxfO15G_JPp0sAod6t0JgBaibJg4JvVqqRTqlw-pPb4w?e=zNnV8E)


# Updates
* 01/01/21 - Project Initiated
* 21/11/22 - Abstract accepted at [WIDAFIL-ICFHR 2022](https://icfhr2022.org/wtc.php)
* 12/12/22 - Repository Created
* 20/12/22 - Results Updated
* 19/04/23 - Paper accepted at [ICDAR 2023](https://icdar2023.org/)

# Acknowledgements
* This repository is based on [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
* We acknowledge the [Rekhta Foundation](https://rekhtafoundation.org/) and the personal collections of [Arjumand Ara](https://du-in.academia.edu/ArjumandAra) for the scanned images and Noor Fatima and Mohammed Usman for the dataset and the manual transcription.
* We also thank all the members of the [Computer Vision Group, IIT Delhi](https://vision-iitd.github.io/) for their support and guidance.

## Contact
* [Abdur Rahman](https://www.linkedin.com/in/abdur-rahman-0b84341a0/)
* [Prof. Arjun Ghosh](https://web.iitd.ac.in/~arjunghosh/)
* [Prof. Chetan Arora](https://www.cse.iitd.ac.in/~chetan/)

## Note
This is an official repository of the project. The copyright of the dataset and the code belongs to the authors. They are for research purposes only and must not be used for any other purpose without the author's explicit permission.

## Citation
If you use the code/dataset, please cite the following paper:

```BibTeX
@article{rahman2023utrnet,
      title={UTRNet: High-Resolution Urdu Text Recognition In Printed Documents}, 
      author={Abdur Rahman and Arjun Ghosh and Chetan Arora},
      journal={arXiv preprint arXiv:2306.15782},
      year={2023},
      eprint={2306.15782},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

