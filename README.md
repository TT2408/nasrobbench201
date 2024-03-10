# RobNASbenchmark

Code for the ICLR'24 paper called "[Robust NAS benchmark under adversarial training: assessment, theory, and beyond](https://openreview.net/pdf?id=cdUpf6t6LZ)".


 In this project, we will build an adversarially trained NAS benchmark called NAS-RobBench-201 that includes the training result of 6466 network architectures in the NAS-Bench-201 search space. 
 
 Specifically, NAS-BENCH-201 search space consists of 15625 architectures while only 6466 architectures are non-isomorphic. The schematic of each architecture (archidx) can be seen at our [project page](https://tt2408.github.io/nasrobbench201hp/).

The id for all 6466 non-isomorphic architectures can be found at NasBenchID_6466_idx.txt


## Environmental set-up (see 'envirionment.yml' for more detail)
```
Python == 3.9.13
pytorch == 1.12.1
torchvision ==0.9.1
torchmetrics == 0.2.0
pytorch-lightning==1.2.8
lightning-bolts==0.3.2
torchattacks==3.4.0
```

## Dataset
We use the same datasets (cifar 10, cifar 100, imagenet16-120) as in NAS-Bench-201. Please download './ImageNet16' from https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7 into the folder './dataset'.

## Download our benchmark dataset
You can download from https://drive.google.com/drive/folders/1Hhv2R1EfpnQOAyClX09mj7rj1Zbbbj0G?usp=drive_link

## Train the architecture
To conduct adversarial training and evaluate a single architecture with id 5617, execute:

For cifar10
```
python main.py --data cifar10 --epochs 50 --archidx 2581 -seed 0
```

For cifar 100
```
python main.py --data cifar100 --epochs 50 --archidx 2581 -seed 0
```

For imagenet16-120:
```
python main.py --data imagenet --epochs 50 --archidx 2581 -seed 0
```

To train all architecture
```
sh run.sh
```


## Evaluate and access the benchmark
```python
from nasrobbench201 import NasRobBench201
dataset = NasRobBench201(metapath= 'meta.json',
                         datasetpath='NAS-RobBench-201_all.json')

# FGSM accuracy
acc = dataset.get_result('cifar10',architecture_id='1000',metric='val_fgsm_3.0_acc')
print(acc)
#0.6729999780654907

# PGD accuracy
acc = dataset.get_result('cifar10',architecture_id='1000',metric='val_pgd_3.0_acc')
print(acc)
#0.6680999994277954
```

## Reference:
https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/NAS-Bench-201.md

https://github.com/steffen-jung/robustness-dataset


## Cite as:
If you use this code, please cite 
```
@inproceedings{
    wu2024robust,
    title={Robust {NAS} benchmark under adversarial training: assessment, theory, and beyond},
    author={Yongtao Wu and Fanghui Liu and Carl-Johann Simon-Gabriel and Grigorios Chrysos and Volkan Cevher},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=cdUpf6t6LZ}
    }
```