# Pigeon Head-Bob
Code for reproducing results seen in my bachelor's thesis "Modeling Head-Bobbing in Pigeon Locomotion using Reinforcement Learning" ([PDF file](main.pdf)). Refer to the thesis for any details regarding the experiments in the repository.

We used the soft actor critic (SAC) implementation in [RLkit](https://github.com/rail-berkeley/rlkit) by [vitchyr](https://github.com/vitchyr) and a proximal policy optimization (PPO) implementation built on top of RLkit [RlkitExtension](https://github.com/johnlime/RlkitExtension.git) for training the pigeon models, both of which are included as [sub-repositories](src/rlkit_ppo).

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
     - `fifty_fifty`
       - Equally-weighted sum of `retinal stabilization` and `motion_parallax`

## Getting Started
Clone the repository.
```
git clone https://github.com/johnlime/pigeon_head_bob.git
```

Set the current directory to the repository's main directory.
```
cd pigeon_head_bob
```

Add the current directory to `PYTHONPATH`
```
export PYTHONPATH=$PWD
```

### Dependency Installation using Anaconda
The following instructions assume that we are training reinforcement learning models in Linux and conducting testing and visualization of them in MacOS.

#### Linux
Run the following command for installing dependencies for Linux.
```
conda env create -f conda_env/rlkit-manual-env-linux64gpu.yml
```

Activate the Anaconda environment using the following command.
```
conda activate rlkit-manual
```

#### MacOS
Run the following command for installing dependencies for the MacOS.
```
conda env create -f conda_env/pybox2d-rlkit-manual-env-mac.yml
```

Activate the Anaconda environment using the following command.
```
conda activate pybox2d-rlkit-manual
```

#### Minimal Dependencies for Using Pigeon Environments
Optionally, you can choose to construct an Anaconda environment solely for using the 2 pigeon environments (without the RLkit training).
```
conda env create -f conda_env/pigeon_minimal_env.yml
conda activate pigeon-env
```

### Dry Runs of Pigeon Environments
Random policies can be run on the environments via the following command.
```
python run/pigeon_run.py -env <ENVIRONMENT>
```
 - `-env`, `--environment`
   - Name of the environment to run
   - `PigeonEnv3Joints`
   - `PigeonRetinalEnv`

## Training Reinforcement Learning Controllers
### Soft Actor Critic
Run the following command for SAC training.
```
python run/sac.py -env <ENVIRONMENT> -bs <BODY_SPEED> -rc <REWARD_FUNCTION> -mo <MAX_OFFSET>
```
- `-env`, `--environment`
  - Name of the environment to run
  - `PigeonEnv3Joints`
  - `PigeonRetinalEnv`

- `-bs`, `--body_speed`
  - Body speed of the pigeon model

- `-rc`, `--reward_code`
  - Specify reward function associated with the set environment
    - `PigeonEnv3Joints`
      - `head_stable_manual_reposition`
      - `head_stable_manual_reposition_strict_angle`
    - `PigeonRetinalEnv`
      - `motion_parallax`
      - `retinal_stabilization`
      - `fifty_fifty`

- `-mo`, `--max_offset`
  - Specify max offset for aligning head to target
  - Not necessary for `PigeonRetinalEnv`
  - Is set to `0.0` by default

- The resulting data are stored under `src/rlkit_ppo/data/`

### Proximal Policy Optimization
Run the following command for PPO training.
```
python run/ppo.py -env <ENVIRONMENT> -bs <BODY_SPEED> -rc <REWARD_FUNCTION> -mo <MAX_OFFSET>
```
- `-env`, `--environment`
  - Name of the environment to run
  - `PigeonEnv3Joints`

- `-bs`, `--body_speed`
  - Body speed of the pigeon model

- `-rc`, `--reward_code`
  - Specify reward function associated with the set environment
    - `head_stable_manual_reposition`
    - `head_stable_manual_reposition_strict_angle`

- `-mo`, `--max_offset`
  - Specify max offset for aligning head to target

- The resulting data are stored under `src/rlkit_ppo/data/`

## Running the Reinforcement Learning Policies on the Pigeon Environments
Run the following command after training the reinforcement learning policies.
```
python run/pigeon_run.py -env <ENVIRONMENT> -dir <DIRECTORY_PATH> -bs <BODY_SPEED> -rc <REWARD_FUNCTION> -mo <MAX_OFFSET>
```
- `-env`, `--environment`
  - Name of the environment to run
  - `PigeonEnv3Joints`
  - `PigeonRetinalEnv`

- `-dir`, `--snapshot_directory`
  - Path to the snapshot directory is within `src/rlkit_ppo/data/`

- `-bs`, `--body_speed`
  - Body speed of the pigeon model

- `-rc`, `--reward_code`
  - Specify reward function associated with the set environment
    - `PigeonEnv3Joints`
      - `head_stable_manual_reposition`
      - `head_stable_manual_reposition_strict_angle`
    - `PigeonRetinalEnv`
      - `motion_parallax`
      - `retinal_stabilization`
      - `fifty_fifty`

- `-mo`, `--max_offset`
  - Specify max offset for aligning head to target
  - Not necessary for `PigeonRetinalEnv`
  - Is set to `0.0` by default

- `-v`, `--video`
  - Export result to video

## Visualizing the Head Trajectories Generated by the Controllers
Run the following command.
```
python run/pigeon_headtrack.py -env <ENVIRONMENT> -dir <DIRECTORY_PATH> -bs <BODY_SPEED>
```
- `-env`, `--environment`
  - Name of the environment to run
  - `PigeonEnv3Joints`
  - `PigeonRetinalEnv`

- `-dir`, `--snapshot_directory`
  - Path to the snapshot directory within `src/rlkit_ppo/data/`

- `-bs`, `--body_speed`
  - Body speed of the pigeon model

- The resulting visualization is under `<SNAPSHOT_DIRECTORY>/body_trajectory/`
