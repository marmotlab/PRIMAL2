# PRIMAL_2: Pathfinding via Reinforcement and Imitation Multi_agent Learning - Lifelong

## Setting up Code
- Install required dependancies from requirements.txt


## Running Code
- Pick appropriate number of meta-agents and threads (`NUM_THREADS` = `Number of agents in each environment ` ) 
- Name training run via `training_version` in `DRLMAPF_A3C_RNN.py`
- Set desired RL-IL ratio and GIF frequencies 
- Select appropriate Observer in `DRLMAPF_A3C_RNN.py` (PRIMAL2 by default) , but a new observer can also be constructed 
- run DRLMAPF_A3C_RNN.py

## Key Files

- `DRLMAPF_A3C_RNN.py` - The main file which contains the entire training code, worker classes and training parameters. 
- `ACNet.py` - Defines network architecture.
- `Env_Builder.py` - Defines the lower level structure of the Lifelong MAPF environment for PRIMAL2, including the world and agents class.
- `PRIMAL2Env.py` - Defines the high level environment class. 
- `Map_Generator.py` - Algorithm used to generate worlds, parameterized by world size, obstacle density and wall components.
- `PRIMAL2Observer.py` - Defines the decentralized observation of each PRIMAL2 agent.
- `PRIMALObserver.py` - Defines the decentralized observation of our previous work, PRIMAL. 
- `Obsever_Builder.py` - The high level observation class


## Other Links
- fully trained PRIMAL2 model in one-shot environment -  https://www.dropbox.com/s/3nppkpy7psg0j5v/model_PRIMAL2_oneshot_3astarMaps.7z?dl=0
- fully trained PRIMAL2 model in LMAPF environment - https://www.dropbox.com/s/6wjq2bje4mcjywj/model_PRIMAL2_continuous_3astarMaps.7z?dl=0


## Authors

[Mehul Damani](damanimehul24@gmail.com)

[Zhiyao Luo](luozhiyao933@126.com)

[Emerson Wenzel](emersonwenzel@gmail.com)

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)
