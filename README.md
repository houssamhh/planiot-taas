# PlanIoT

## Project Description
PlanIoT is a framework-based solution that enables adaptive data flow management at the middleware-layer using automated planning methodologies. This is achieved via the following core software components: (i) a queueing network composer; (ii) automated planning modeler; and (iii) an AI planner. 

## Getting Started
This repository contains the following directories:
* ```Code/PlanIoT-SEAMS2023```: contains the *Queueing Network Composer* and the *Dataset Generator* components as a Maven project.
* ```Scenarios```: contains the files needed to run experiments using PlanIoT (e.g., IoT system specifications, PDDL templates). These files were used to run the experiments found in [1].
* ```Scripts```: contains scripts needed to run the Queueing Network Composer, Dataset Generator, PDDL Modeler, and the AI Planner.

### Installation Requirements
This artifact has been prepared for a host that supports X11 forwarding on Docker containers. X11 forwarding is required by the JMT simulator, even when the graphical user interface is not used.
The artifact has been tested on GNU/Linux systems such as Debian or Ubuntu.
For Windows users, you can install a Linux Distribution (Ubuntu/Debian) with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). The artifact has been tested with WSL 2 running Ubuntu.
The artifact has NOT been tested with MacOS. 

### Installing PlanIoT
We provide a Docker container for running PlanIoT. To build and run the container, you need to have Docker installed. You can download Docker [here](https://docs.docker.com/get-docker/).
Alternatively, we provide instructions for directly installing PlanIoT on your machine. However, to avoid dependency problems, we strongly recommend using the Docker container to run PlanIoT.
#### Using  the Docker container
##### Building the container
The Docker container runs Ubuntu version 20.04 with Metric-FF version 2.0 planner and the Java Modelling Tools (JMT) version 1.2.2 queueing simulator.
Start by cloning this repository.
To build the container, execute the following commands:
```
$ cd planiot-seams2023
$ docker build --rm \
--build-arg UID=$(id -u) --build-arg GID=$(id -g) \
--tag planiot:latest -f- . < dockerfile
```

##### Running the Docker container

To run the container and enter into it (with the shell command `/bin/bash`), run the following commands:
```
$ docker run -it --rm \
--user "$(id -u)":"$(id -g)" \
-v $(pwd):/home/planiot/planiot \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /tmp/.docker.xauth:/tmp/.docker.xauth \
--env=DISPLAY=unix$DISPLAY \
--env=XAUTHORITY=/tmp/.docker.xauth \
planiot
planiot@657641f176f4:~$ pwd
/home/planiot
# go to the directory of the repository
planiot@657641f176f4:~$ cd planiot/
planiot@657641f176f4:~$ ls
# the content of the planiot repository
```
Now you are ready to start using PlanIoT.

#### Using PlanIoT
The *Queueing Network Composer* and *Dataset Generator* are provided as a Maven project. To use them, start by building the Maven project:
```
planiot@657641f176f4:~$ (cd Code/PlanIoT-SEAMS2023/;mvn -q install)
```
To illustrate how PlanIoT can be used, we go through an example using the files in the ```Scenarios/medium-load``` directory.

##### Generating a performance metrics dataset
The directory ```Scenarios/medium-load/system-specifications``` contains IoT system specifications for an IoT system with 16 applications, 30 topics, and 80 subscriptions. Each JSON file corresponds to a specific configuration of the IoT system. For example, the file ```default.json``` represents a basic IoT platform where no priorities, drop rates, or resource allocation policies are used. On the other hand, the files ```prioRT.json``` and ```dropVS1.json``` represent the same IoT system while applying priorities to RT applications and a drop rate of 1% to VS applications, respectively.

To create a queueing network, simulate it, and add the results of the simulation to the dataset, run the following command:
```bash ./Scripts/run_simulation.sh <input-file> <output-file> <simulation-duration> <alias>```
where:
*  ``<input-file>`` is the JSON file containing the IoT system specifications,
*  ``<output-file>`` is the response times of subscriptions under different configurations of the IoT system. If the file does not exist, it will be created.
* ``<simulation-duration>`` is the duration for which you wish to run the simulation (in seconds). We recommend running the simulation for at least 5 minutes to reach a 95% confidence interval.
* ``<alias>`` is an alias that represents the configuration of the IoT system. This alias is going to be used when storing the results in the dataset.

For example, if you want to run a simulation for the ``default`` configuration of the system, run the following command:
``` 
$ planiot@657641f176f4:~$ pwd
/home/planiot/planiot
planiot@657641f176f4:~$ bash ./Scripts/run_simulation.sh Scenarios/medium-load/system-specifications/default.json Scenarios/medium-load/dataset/results.csv 300 default
```
This command will generate ```metrics_default.csv``` and ```results.csv``` files in the ```Scenarios/medium-load/dataset``` directory.

To populate the dataset, you need to run simulations for multiple configurations of the system. For example, we use the files in the ```Scenarios/medium-load/system-specifications``` directory to create the dataset found in ```Scenarios/medium-load/dataset```.

*Note: to avoid running simulations for all IoT system specification files, we provide the ```response-times-dataset.csv``` file containing reponse times for the IoT system under different configurations. This file can be used in the next step*

##### Creating the PDDL domain and problem files
Once the dataset is created, we can instantiate the PDDL domain and problem files. To do this, we run the following command:
```
$ python InstantiatePddlTemplates.py <response-times-file> <domain-template> <problem-template>
```
where:
* ```<response-times-file> ```is the file containing the response times for multiple configurations of the IoT system (e.g., in this repository, the file is called ```response-times.csv```)
* ```<domain-template>``` is the PDDL domain file template.
* ```<problem-template>``` is the PDDL problem file template.
Note that PDDL domain and problem file templates can be found in the directory ```pddl-templates``` at the same level of the ```dataset``` directory.
Running this command will create the PDDL domain and problem files needed to run the AI planner, and place them in the ```pddl-files``` directory.


For example, continuing with our medium-loaded system, we can run the following command to instantiate the PDDL domain and problem file templates:
```
$ python Scripts/InstantiatePddlTemplates.py \
Scenarios/medium-load/dataset/response-times-dataset.csv \
Scenarios/medium-load/pddl-templates/domain-template.pddl \
Scenarios/medium-load/pddl-templates/problem-template.pddl
```
The ```pddl-files``` directory will contain a ```domain-generated.pddl``` file and a ```problem-generated.pddl``` file.

Note that the script takes an additional argument when using templates for overloaded systems  (*-o*) or emergency situations (*-e*).

##### Getting plans for Edge infrastructure adaptation
To run the AI planner and get a plan for configuring the Edge infrastructure, we run the ```run_planner.sh``` script.
```
$ bash ./Scripts/run_planner.sh <domain-file> <problem-file> <solution-file>
```
where:
* ```<domain-file>``` is the PDDL domain file
* ```<problem-file>``` is the PDDL problem file
* ```<solution-file>``` is the solution file that contains the plan

For example, we can run the following command to get a plan for our medium-loaded system:
```
$ bash ./Scripts/run_planner.sh \
Scenarios/medium-load/pddl-files/domain-generated.pddl \
Scenarios/medium-load/pddl-files/problem-generated.pddl \
Scenarios/medium-load/plans/solution.pddl
```


### Installing PlanIoT directly on your machine

#### OS installation

We use Ubuntu version 20.04 with the following packages installed: *make*, *gcc*, *libc6-dev*, *wget*, *openjdk-11-jdk maven*, *python*, *flex*, *bison*.

#### Metric-FF installation

Install Metric-FF version 2.0 from (https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html) with the following process:
```
$ wget --progress dot:mega -O Metric-FF-v2.0.tgz \
https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.0.tgz
$ tar xfz Metric-FF-v2.0.tgz
$ rm -f Metric-FF-v2.0.tgz
$ # Changes of constant values in Metric-FF-v2.0/ff.h
$ sed --in-place 's/MAX_LNF_F 25/MAX_LNF_F 150/' Metric-FF-v2.0/ff.h
$ sed --in-place 's/MAX_LNF_EFFS 50/MAX_LNF_EFFS 200/' Metric-FF-v2.0/ff.h
$ (cd Metric-FF-v2.0; make)
```
Configure your shell to have access to the binary `Metric-FF-v2.0/ff`.

#### Java Modelling Tools (JMT) Installation

Install Java Modelling Tool version 1.2.2 from (https://jmt.sourceforge.net/Download.html) with the following process:
```
$ mkdir JMT-v1.2.2
$ cd JMT-v1.2.2
$ wget --progress dot:mega -O jmt-singlejar-1.2.2.jar \
	http://sourceforge.net/projects/jmt/files/jmt/JMT-1.2.2/JMT-singlejar-1.2.2.jar/download
```

Configure your shell to complement the `CLASSPATH` with the JMT archive:

```
export CLASSPATH=$CLASSPATH:/opt/JMT-v1.2.2/jmt-singlejar-1.2.2.jar
```
You can now start [using PlanIoT](#using-planiot).

## Citations
If you use this project, please cite the following paper:
H. Hajj Hassan, G. Bouloukakis, A. Kattepur, D. Conan, and D. Belaïd, “PlanIoT: A Framework for Adaptive Data Flow Management in IoT-enhanced Spaces”, in *IEEE/ACM International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)*, 2023.

[1] H. Hajj Hassan, G. Bouloukakis, A. Kattepur, D. Conan, and D. Belaïd, “PlanIoT: A Framework for Adaptive Data Flow Management in IoT-enhanced Spaces”, in *IEEE/ACM International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)*, 2023.