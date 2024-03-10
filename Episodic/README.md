## Usage

After following the instructions on the previous page, you can train the method of your choice using the following code:

```
python src/train.py --base_config 'config/base.yaml' --method 'config/maml.yaml' --opts scenario "scenario_1" num_ways 5 num_support 5
```
Using the code above, we selected the MAML method, scenario_1, 24-way and 20-shot setting.
To see the full list of the configurations, see .yaml files provided in the config folder. These configurations can be modified either by directly editing the .yaml file or by putting them next to --opts in the code above (as what we did for scenario, num_ways, and num_support). As the method, you can select either 'maml.yaml', 'protonet.yaml', 'simpleshot.yaml', or 'metaoptnet.yaml.

After training is completed, you can test the method of your choice (here MAML) using the following code:

```
python src/test.py --base_config 'config/base.yaml' --method 'config/maml.yaml' --opts scenario "scenario_1" num_ways 5 num_support 5 sampling_startegy "dirichlet"
```
In the code above, we used the realistic query sampling strategy to test the method.

## Acknowledgements

Special thanks to the authors of the papers "Fhist: A benchmark for few-shot classification of histological images" and "Realistic evaluation of transductive few-shot learning" for publicly sharing their source code (https://github.com/mboudiaf/Few-shot-histology) and (https://github.com/oveilleux/Realistic_Transductive_Few_Shot) from which this repo was inspired from.


