## Usage

After following the instructions on the previous page, you can train the baseline model using the following code:

```
python3 -m src.main --base_config config/base_config.yaml --method_config config/baseline.yaml
```
Note that for the NonEpisodic methods, we train a single model using standard supervised training via cross-entropy loss. After training is completed, you can test the method of your choice using the following code:

```
python3 -m src.main-test --base_config config/base_config.yaml --method_config config/alpha_tim.yaml --scenario "scanario_1" --n_ways 24 --shots [20] --sampling_startegy "dirichlet"
```
In the code above, we used the realistic query sampling strategy to test the method in scenario_1 using 24-way 20-shot setting.


To see the full list of the configurations, see .yaml files provided in the config folder. These configurations can be modified either by directly editing the .yaml files or by inserting them as above. As the method, you can select either 'baseline.yaml', 'alpha_tim.yaml', 'tim.yaml', 'entropy_min 'or 'simpleshot.yaml'.




## Acknowledgements

Special thanks to the authors of the papers "Realistic evaluation of transductive few-shot learning" for publicly sharing their source code (https://github.com/oveilleux/Realistic_Transductive_Few_Shot) from which this repo was inspired from.

