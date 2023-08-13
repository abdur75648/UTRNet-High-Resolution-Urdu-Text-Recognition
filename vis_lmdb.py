"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import lmdb, six, random
from PIL import Image
from tqdm import tqdm

data_dir = "lmdb_ihtr/malayalam/train/real/1L_synth"
env = lmdb.open(data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print("Number of samples: ",nSamples)

def get_item(index):
    assert index <= nSamples, 'index range error'
    with env.begin(write=False) as txn:
        # Label
        label_key = 'label-%09d'.encode() % index
        #print("Label key: ",txn.get(label_key))
        label = txn.get(label_key).decode('utf-8')
        # Image
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
    return (img, label)

for i in range(10):
    idx = random.randint(1,nSamples+1)
    img, label = get_item(idx)
    img.save(str(idx)+".jpg")
    with open("samples.txt","a",encoding="utf-8") as f:
        f.write(str(idx)+".jpg" + "\t" + label + "\n")

# calculate mean label_length and mean aspected ratio of the samples & also draw histograms of label length and aspected ratio
# mean_label_length = 0
# mean_aspected_ratio = 0
# label_lengths = []
# aspected_ratios = []
# for i in tqdm(range(1,nSamples)):
#     img, label = get_item(i)
#     mean_label_length += len(label)
#     mean_aspected_ratio += img.size[0] / img.size[1]
#     label_lengths.append(len(label))
#     aspected_ratios.append(img.size[0] / img.size[1])
# mean_label_length /= nSamples
# mean_aspected_ratio /= nSamples
# print("Mean label length: ",mean_label_length)
# print("Max label length: ",max(label_lengths))
# print("Mean aspected ratio: ",mean_aspected_ratio)
# print("Max aspected ratio: ",max(aspected_ratios))
# import matplotlib.pyplot as plt
# plt.hist(label_lengths, bins=20)
# plt.title("Label length histogram")
# plt.savefig("label_length_histogram.png")
# import matplotlib.pyplot as plt
# plt.hist(aspected_ratios, bins=20)
# plt.title("Aspected ratio histogram")
# plt.savefig("aspected_ratio_histogram.png")
