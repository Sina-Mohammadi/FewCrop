# Few-shot Learning for Crop Mapping from Satellite Image Time Series
The PyTorch implementation code for our paper "Few-shot Learning for Crop Mapping from Satellite Image Time Series".


Our paper has been accepted to the Remote Sensing journal.

## The Method
We adapted eight few-shot learning methods for crop type mapping in order to address the challenge of scarcity of the labeled samples in the target dataset. These methods include MAML (Episodic-Transductive), Prototypical Networks (Episodic-Inductive), MetaOptNet (Episodic-Inductive), Baseline (NonEpisodic-Inductive), Simpleshot (NonEpisodic-Inductive), Entropy min (NonEpisodic-Transductive), TIM (NonEpisodic-Transductive), and α-TIM (NonEpisodic-Transductive). In our experiments, we map a large diversity of crops from a complex agricultural area situated in Ghana and infrequent crops cultivated in the selected study areas from France.  FSL methods are commonly evaluated using class-balanced unlabeled sets from the target domain data (query sets), leading to overestimated classification results. To enable a realistic evaluation, inspired by the work of Veilleux et al. [1], we used the Dirichlet distribution to model the class proportions in few-shot query sets as random variables. 

<p align="center"><img src="https://github.com/Sina-Mohammadi/FewCrop/blob/main/fig/framework.jpg" width="730" height="650"></p>
<p align="center"><img src="https://github.com/Sina-Mohammadi/FewCrop/blob/main/fig/featureextractor.jpg" width="730" height="650"></p>
