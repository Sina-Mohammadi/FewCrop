### Usage

After following the instructions on the previous page, you can train the method of your choice using the following code:

```
python src/test.py --base_config 'config/base.yaml' --method 'config/maml.yaml' --opts scenario "scenario_1" num_ways 24 num_support 20
```
Using the code above, we selected the MAML method, scenario_1, 24-way and 20-shot setting.
To see the full list of the configurations, see .yaml files provided in the config folder. These configurations can be modified either by directly editing the .yaml file or by putting them next to --opts in the code above.
