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


## Frequently asked questions
1. I got `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"` when running on a server
Running your code starting with `xvfb-run` will solve the problem. You may refer to https://stackoverflow.com/questions/60922076/pyglet-canvas-xlib-nosuchdisplayexception-cannot-connect-to-none-only-happens and relevant issues on StackFlow for help.

2. In one-shot environment, why agent turns black after reaching a goal?
In the one-shot scenario, agent will 'disappear'(i.e., removed from the env). For visualization we keep it as black. Removal of agent who has achieved its goal is necessary, since a lot of narrow corridors in the map could cause unsolvable block and collision. One-shot scenario per se is just a way to test the optimality of the planner. By contrast we do not remove any agents for any reason in continuous env.

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
