[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Summary

In this project, an agent is trained to control double-jointed arm to keep a 'fingertip' as close to the moving target as possible for defined amount of time.

![Trained Agent][image1]

In the project's environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of an agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector isa number between -1 and 1.

The project environment is running inside the Unity simulation engine and trained Agent can interact with it using the UnityAgents API.
The variant of Unity simulation environment chosen for this project contains 20 identical agents, each with its own copy of environment. Therefore to achieve the goal of the project all agents of Unity simulation environment must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically, after each episode, rewards that each agent received are added up and the average of these 20 scores is taken as the resulting episode score.

### Project environment setup

1. Follow the initial setup instructions in `../python/README.MD` file.

2. Download the agent execution environment from one of the links below.  You need only select the environment that matches your operating system:

        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS or remote PC (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Unzip (or decompress) the contents of downloaded archive file into source code folder of this project (`../continious-control/`)

3. Set the value of 'file_name' parameter to match a name of unziped Unity environment file in the code line of Continuous_Control.ipynb file.
 ```
 env = UnityEnvironment(file_name='UNITY_ENVIRONMENT_FILE_NAME')
 ```


### Project code execution

To run pretrained agent using provided model weights or train your own agent follow the instructions in `Continuos_Control.ipynb`

