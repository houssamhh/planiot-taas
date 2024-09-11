# Scripts

This directory contains the following scripts:

* `InstantiatePddlTemplates.py`: used to instantiate the domain and problem file templates. 
Usage: ```python InstantiatePddlTemplates.py <response-times-file> <domain-template> <problem-template> <option>```, where:
    * ```<response-times-file>``` is the path to the CSV file containing the response times of subscriptions in the IoT system under different configurations.
    * ```<domain-template>``` is the path to the PDDL domain file template.
    * ```<problem-template>``` is the path to the PDDL problem file template.
    * ```<option>``` can be -o (for overloaded systems) or -e (for emergency scenarios). In other cases, this argument is not needed.

* ```run_planner.sh```: used to run the metric-FF planner and generate the plan for Edge infrastructure configuration. 
Usage: ```run_planner.sh <domain-file> <problem-file> <solution-file>``` where:
	* ```<domain-file>``` is the PDDL domain file.
	* ```<problem-file>``` is the PDDL problem file.
	* ```<solution-file>``` is the solution file that contains the plan.

* ```run_simulation.sh```: used to compose the JMT queueing network, run the simulation, and generate performance metrics files.
Usage: ```Scripts/run_simulation.sh <input-file> <output-file> <simulation-duration> <alias>``` where:
	*  ``<input-file>`` is the JSON file containing the IoT system specifications.
	*  ``<output-file>`` is the file that contains response times of subscriptions under different configurations of the IoT system. If the file does not exist, it will be created.
	* ``<simulation-duration>`` is the duration for which you wish to run the simulation (in seconds). We recommend running the simulation for at least 5 minutes to reach a 95% confidence interval.
	* ``<alias>`` is an alias that represents the configuration of the IoT system. This alias is going to be used when storing the results in the dataset.


