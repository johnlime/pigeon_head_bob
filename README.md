# Pigeon Head-Bob
Code for reproducing results seen in my bachelor's thesis "Modeling Head-Bobbing in Pigeon Locomotion using Reinforcement Learning" ([PDF file](main.pdf)). Refer to the thesis for any details regarding the experiments in the repository.

We used the soft actor critic (SAC) implementation in [RLkit](https://github.com/rail-berkeley/rlkit) and a proximal policy optimization (PPO) implementation built on top of RLkit [RlkitExtension](https://github.com/johnlime/RlkitExtension.git) for training the pigeon models, both of which are included as [sub-repositories](src/rlkit_ppo).

## Abstract
Head-bobbing is a behavior unique to forward locomotion in small birds, mainly pigeons, that consists of a hold phase, where they lock the position of their heads into one position, and a thrust phase, where they move them to a different position. 2 main functionalities of the behavior have been proposed in preliminary research: visual stabilization and induction of motion parallax; however, there is a lack of research that focus on validating their sufficiency by attempting to reproduce it in environments that take physics into account. In our research, we construct a simplified model of pigeons that represent their heads, necks, and bodies and validate the preliminary hypotheses regarding the functionalities of the behavior using reinforcement learning.

## Pigeon OpenAI Gym Environments
We constructed 2 OpenAI Gym environments that utilize the simplified pigeon model
 - `PigeonEnv3Joints` ([code](gym_env/pigeon_gym.py))
   - Pigeon model is tasked to move its head to predefined target locations
   - Head follows a path that represents head-bobbing behavior
   - Reward functions
     - `head_stable_manual_reposition`
       - Negative distance between the target locations and the position of the head relative to a threshold value `max_offset`.
     - `head_stable_manual_reposition_strict_angle`
       - Stricter version of `head_stable_manual_reposition`
       - Rewards are only produced when the angle of the head-tilt is within 30 degrees.

 - `PigeonRetinalEnv` ([code](gym_env/pigeon_gym_retinal.py))
   - Reward functions modeled on the 2 functionalities of the head-bobbing behaviors courtesy of the preliminary hypotheses
     - `motion_parallax`
       - Sum of velocities of external objects within the retina relative to each other
       - Represents motion parallax induction during the thrust phase
     - `retinal_stabilization`
       - Negative sum of velocities of external objects within the retina
       - Represents retinal stabilization during the hold phase

## Getting Started
Clone the repository.
```
git clone https://github.com/johnlime/pigeon_head_bob.git
```

Set the current directory to the repository's main directory.
```
cd pigeon_head_bob
```

### Dependency Installation using Anaconda
The following instructions assume that we are training reinforcement learning models in Linux and conducting testing and visualization of them in MacOS.

#### Linux
Execute the following command for installing dependencies for Linux.
```
conda env create -f conda_env/rlkit-manual-env-linux64gpu.yml
```

Activate the Anaconda environment using the following command.
```
conda activate rlkit-manual
```

#### MacOS
Execute the following command for installing dependencies for the MacOS.
```
conda env create -f conda_env/pybox2d-rlkit-manual-env-mac.yml
```

Activate the Anaconda environment using the following command.
```
conda activate pybox2d-rlkit-manual
```

### Training Reinforcement Learning Controllers
#### Soft Actor Critic
Execute the following command for SAC training.
```

```
