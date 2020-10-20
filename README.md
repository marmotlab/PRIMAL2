# PRIMAL_2: Pathfinding via Reinforcement and Imitation Multi_agent Learning - Lifelong

## Setting up Code
- cd into the od_mstar3 folder.
- python3 setup.py build_ext --inplace
- Check by going back to the root of the git folder, running python3 and "import cpp_mstar"


## Running Code
- Pick appropriate number of meta agents via variables `NUM_META_AGENTS` and `NUM_IL_META_AGENTS` in `parameters.py`
- The number of RL meta-agents is implicity defined by the difference between total meta-agents and IL meta-agents (`NUM_RL_META_AGENTS` = `NUM_META_AGENTS` - `NUM_IL_META_AGENTS`)
- Name training run via `training_version` in `parameters.py`
- call `python driver.py`

## Key Files
- `parameters.py` - Training parameters.
- `driver.py` - Driver of program. Holds global network for A3C.
- `Runner.py` - Compute node for training. Maintains a single meta agent.
- `Worker.py` - A single agent in a simulation environment. Majority of episode computation, including gradient calculation, occurs here.
- `Ray_ACNet.py` - Defines network architecture.
- `Env_Builder.py` - Defines the lower level structure of the Lifelong MAPF environment for PRIMAL2, including the world and agents class.
- `PRIMAL2Env.py` - Defines the high level environment class. 
- `Map_Generator2.py` - Algorithm used to generate worlds, parameterized by world size, obstacle density and wall components.
- `PRIMAL2Observer.py` - Defines the decentralized observation of each PRIMAL2 agent.
- `Obsever_Builder.py` - The high level observation class


## Other Links
- fully trained PRIMAL2 model in one-shot environment -  https://www.dropbox.com/s/3nppkpy7psg0j5v/model_PRIMAL2_oneshot_3astarMaps.7z?dl=0
- fully trained PRIMAL2 model in LMAPF environment - https://www.dropbox.com/s/6wjq2bje4mcjywj/model_PRIMAL2_continuous_3astarMaps.7z?dl=0


## Authors

[Mehul Damani](damanimehul24@gmail.com)

[Zhiyao Luo](luozhiyao933@126.com)

[Emerson Wenzel](emersonwenzel@gmail.com)

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)
