# Few-shot Learning for Crop Mapping from Satellite Image Time Series
The PyTorch implementation code for our paper "Few-shot Learning for Crop Mapping from Satellite Image Time Series".

Our paper has been accepted to the Remote Sensing journal and is publicly available at [this link](https://www.mdpi.com/2072-4292/16/6/1026).

## The Method
We adapted eight few-shot learning methods for crop type mapping in order to address the challenge of scarcity of the labeled samples in the target dataset. These methods include MAML (Episodic-Transductive), Prototypical Networks (Episodic-Inductive), MetaOptNet (Episodic-Inductive), Baseline (NonEpisodic-Inductive), Simpleshot (NonEpisodic-Inductive), Entropy min (NonEpisodic-Transductive), TIM (NonEpisodic-Transductive), and α-TIM (NonEpisodic-Transductive). In our experiments, we map a large diversity of crops from a complex agricultural area situated in Ghana (scenario_1) and infrequent crops cultivated in the selected study areas from France (scenario_2). FSL methods are commonly evaluated using class-balanced unlabeled sets from the target domain data (query sets), leading to overestimated classification results. To enable a realistic evaluation, inspired by the work of Veilleux et al. [1], we used the Dirichlet distribution to model the class proportions in few-shot query sets as random variables. Finally, to handle the variability in lengths of time series data with the capability of capturing information across multiple levels of granularity, we add a Temporal Pyramid Pooling (TPP) layer to the output of our backbone network. This layer generates fixed-length feature representations.

[1] Veilleux, Olivier, et al. "Realistic evaluation of transductive few-shot learning." Advances in Neural Information Processing Systems 34 (2021): 9290-9302.

### The framework based on TIM and α-TIM methods and the feature extractor network: 
<p align="center"><img src="https://github.com/Sina-Mohammadi/FewCrop/blob/main/fig/framework.jpg" width="730" height="650"></p>
<p align="center"><img src="https://github.com/Sina-Mohammadi/FewCrop/blob/main/fig/featureextractor.jpg" width="550" height="420"></p>


## Getting Started
To install the required packages:
```
pip install -r requirements
```
## Data
Download the preprocessed data for the two scenarios from Zenedo using the following links: [scenario_1]([https://zenodo.org/records/10802507/files/scenario_1.rar?download=1]) , [scenario_2]([https://zenodo.org/api/records/10802507/draft/files/scenario_2.rar/content](https://zenodo.org/records/10802507/files/scenario_2.rar?download=1)) - Then after unzipping them, put them in the data_fewcrop folder.

The benchmark dataset is created using the data provided by the following papers: 1) The PASTIS dataset from the paper "Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks", 2) The ZueriCrop dataset from the paper "Crop mapping from image time series: Deep learning with multi-scale label hierarchies", and 3) The Ghana dataset from the paper "Semantic Segmentation of Crop Type in Africa: A Novel Dataset and Analysis of Deep Learning Methods". Please cite them if you use the benchmark dataset in your paper:

```
@article{garnot2021panoptic,
  title={Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks},
  author={Sainte Fare Garnot, Vivien  and Landrieu, Loic },
  journal={ICCV},
  year={2021}
}
```

```
@article{turkoglu2021crop,
  title={Crop mapping from image time series: Deep learning with multi-scale label hierarchies},
  author={Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk},
  journal={Remote Sensing of Environment},
  volume={264},
  pages={112603},
  year={2021},
  publisher={Elsevier}
}
```

```
@inproceedings{m2019semantic,
  title={Semantic segmentation of crop type in Africa: A novel dataset and analysis of deep learning methods},
  author={M Rustowicz, Rose and Cheong, Robin and Wang, Lijing and Ermon, Stefano and Burke, Marshall and Lobell, David},
  booktitle={Proceedings of the IEEE/cvf conference on computer vision and pattern recognition workshops},
  pages={75--82},
  year={2019}
}
```


## Usage
The code is divided into two parts: Navigate to the Episodic folder for the episodic methods and navigate to the NonEpisodic folder for the NonEpisodic methods to see the further instructions.
