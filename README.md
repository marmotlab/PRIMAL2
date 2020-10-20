# PRIMAL_2: Pathfinding via Reinforcement and Imitation Multi_agent Learning - Lifelong

## Description
- This branch is for model testing and testing map generation.
- You are currently at continuous testing branch, where agents will be assigned new goals after they finish their current
goals. Continuous testing will end in a fixed number of time step. maxLen of each episode can be changed at get_maxLength.
function in TestingEnv.py.
- We provide M*, PRIMAL1 and PRIMAL2 for testing. You can inherit MstarContinuousPlanner or RL_Planner in TestingEnv.py to 
set up testing for your own planner.
- While using RL_Planner, do make sure your model fits correct num_layers of the network, observation_size and num_future_steps!
-Good luck and have fun! (by Zhiyao Luo)

## Setting up Code
- PRIMAL2_env.yaml contains all necessary packages for PRIMAL2 training and testing

- In addition, there are two C++ stuffs you need to build yourself:

        cd into the od_mstar3 folder.
        python3 setup.py build_ext --inplace
        Check by going back to the root of the git folder, running python3 and "import cpp_mstar"
        
        cd into the astarlib3 folder.
        python3 setup.py build_ext --inplace
        Check by going back to the root of the git folder, running "from astarlib3.astarlib import aStar"

## Running Code
- call 'python TestingEnv.py' to run the testing of a single map
- call 'python TestGenerator.py' to run testing map generation
- call 'python MultiProcessingTestDriver.py' to run a series of testing maps by mp


## Other Links
- fully trained PRIMAL2 model in one-shot environment -  https://www.dropbox.com/s/3nppkpy7psg0j5v/model_PRIMAL2_oneshot_3astarMaps.7z?dl=0
- fully trained PRIMAL2 model in LMAPF environment - https://www.dropbox.com/s/6wjq2bje4mcjywj/model_PRIMAL2_continuous_3astarMaps.7z?dl=0


## Authors

[Mehul Damani](damanimehul24@gmail.com)

[Zhiyao Luo](luozhiyao933@126.com)

[Emerson Wenzel](emersonwenzel@gmail.com)

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)