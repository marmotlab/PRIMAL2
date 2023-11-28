#include <algorithm>    // std::shuffle
#include <random>      // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include "CBS.h"
#include "SIPP.h"
#include "SpaceTimeAStar.h"


// takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
// also, do the same for ll_min_f_vals and paths_costs (since its already "on the way").
inline void CBS::updatePaths(CBSNode* curr)
{
	for (int i = 0; i < num_of_agents; i++)
		paths[i] = &paths_found_initially[i];
	vector<bool> updated(num_of_agents, false);  // initialized for false

	while (curr != nullptr)
	{
		for (auto & path : curr->paths)
		{
			if (!updated[path.first])
			{
				paths[path.first] = &(path.second);
				updated[path.first] = true;
			}
		}
		curr = curr->parent;
	}
}


void CBS::copyConflicts(const list<shared_ptr<Conflict >>& conflicts,
	list<shared_ptr<Conflict>>& copy, const list<int>& excluded_agents)
{
	for (const auto& conflict : conflicts)
	{
		bool found = false;
		for (auto a : excluded_agents)
		{
			if (conflict->a1 == a || conflict->a2 == a)
			{
				found = true;
				break;
			}
		}
		if (!found)
		{
			assert(!conflict->constraint1.empty());
			assert(!conflict->constraint2.empty());
			copy.push_back(conflict);
		}
	}
}


void CBS::findConflicts(CBSNode& curr, int a1, int a2)
{
	size_t min_path_length = paths[a1]->size() < paths[a2]->size() ? paths[a1]->size() : paths[a2]->size();
	for (size_t timestep = 0; timestep < min_path_length; timestep++)
	{
		int loc1 = paths[a1]->at(timestep).location;
		int loc2 = paths[a2]->at(timestep).location;
		if (loc1 == loc2)
		{
			cout << "conflict" << endl;
			cout << "(" << loc1 << "," << loc2 <<  "," << timestep <<")" << endl;
			shared_ptr<Conflict> conflict(new Conflict());
			if (target_reasoning && paths[a1]->size() == timestep + 1)
			{
				conflict->targetConflict(a1, a2, loc1, timestep);
			}
			else if (target_reasoning && paths[a2]->size() == timestep + 1)
			{
				conflict->targetConflict(a2, a1, loc1, timestep);
			}
			else
			{
				conflict->vertexConflict(a1, a2, loc1, timestep);
			}
			assert(!conflict->constraint1.empty());
			assert(!conflict->constraint2.empty());
			curr.unknownConf.push_back(conflict);
		}
		else if (timestep < min_path_length - 1
				 && loc1 == paths[a2]->at(timestep + 1).location
				 && loc2 == paths[a1]->at(timestep + 1).location)
		{
			shared_ptr<Conflict> conflict(new Conflict());
			conflict->edgeConflict(a1, a2, loc1, loc2, (int)timestep + 1);
			assert(!conflict->constraint1.empty());
			assert(!conflict->constraint2.empty());
			curr.unknownConf.push_back(conflict); // edge conflict
		}
	}
	if (paths[a1]->size() != paths[a2]->size())
	{
		int a1_ = paths[a1]->size() < paths[a2]->size() ? a1 : a2;
		int a2_ = paths[a1]->size() < paths[a2]->size() ? a2 : a1;
		int loc1 = paths[a1_]->back().location;
		for (size_t timestep = min_path_length; timestep < paths[a2_]->size(); timestep++)
		{
			int loc2 = paths[a2_]->at(timestep).location;
			if (loc1 == loc2)
			{
				shared_ptr<Conflict> conflict(new Conflict());
				if (target_reasoning)
					conflict->targetConflict(a1_, a2_, loc1, timestep);
				else
					conflict->vertexConflict(a1_, a2_, loc1, timestep);
				assert(!conflict->constraint1.empty());
				assert(!conflict->constraint2.empty());
				curr.unknownConf.push_front(conflict); // It's at least a semi conflict			
			}
		}
	}
}


void CBS::findConflicts(CBSNode& curr)
{
	clock_t t = clock();
	if (curr.parent != nullptr)
	{
		// Copy from parent
		list<int> new_agents;
		for (const auto& p : curr.paths)
		{
			new_agents.push_back(p.first);
		}
		copyConflicts(curr.parent->conflicts, curr.conflicts, new_agents);
		copyConflicts(curr.parent->unknownConf, curr.unknownConf, new_agents);

		// detect new conflicts
		for (auto it = new_agents.begin(); it != new_agents.end(); ++it)
		{
			int a1 = *it;
			for (int a2 = 0; a2 < num_of_agents; a2++)
			{
				if (a1 == a2)
					continue;
				bool skip = false;
				for (auto it2 = new_agents.begin(); it2 != it; ++it2)
				{
					if (*it2 == a2)
					{
						skip = true;
						break;
					}
				}
				findConflicts(curr, a1, a2);
			}
		}
	}
	else
	{
		for (int a1 = 0; a1 < num_of_agents; a1++)
		{
			for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
			{
				findConflicts(curr, a1, a2);
			}
		}
	}
	// curr.tie_breaking = (int)(curr.unknownConf.size() + curr.conflicts.size());
	runtime_detect_conflicts += (double) (clock() - t) / CLOCKS_PER_SEC;
}


shared_ptr<Conflict> CBS::chooseConflict(const CBSNode& node) const
{
	if (screen == 3)
		printConflicts(node);
	shared_ptr<Conflict> choose;
	if (node.conflicts.empty() && node.unknownConf.empty())
		return nullptr;
	else if (!node.conflicts.empty())
	{
		choose = node.conflicts.back();
		for (const auto& conflict : node.conflicts)
		{
			if (*choose < *conflict)
				choose = conflict;
		}
	}
	else
	{
		choose = node.unknownConf.back();
		for (const auto& conflict : node.unknownConf)
		{
			if (*choose < *conflict)
				choose = conflict;
		}
	}
	return choose;
}


void CBS::computePriorityForConflict(Conflict& conflict, const CBSNode& node)
{
    conflict.secondary_priority = 0; // random
    /*switch (conflict.type) // Earliest
    {
        case conflict_type::STANDARD:
        case conflict_type::RECTANGLE:
        case conflict_type::TARGET:
        case conflict_type::MUTEX:
            conflict.secondary_priority = get<3>(conflict.constraint1.front());
            break;
        case conflict_type::CORRIDOR:
            conflict.secondary_priority = min(get<2>(conflict.constraint1.front()),
                                              get<3>(conflict.constraint1.front()));
            break;
    }*/
}


void CBS::classifyConflicts(CBSNode& node)
{
	// Classify all conflicts in unknownConf
	while (!node.unknownConf.empty())
	{
		shared_ptr<Conflict> con = node.unknownConf.front();
		int a1 = con->a1, a2 = con->a2;
		int a, loc1, loc2, timestep;
		constraint_type type;
		tie(a, loc1, loc2, timestep, type) = con->constraint1.back();
		node.unknownConf.pop_front();


		bool cardinal1 = false, cardinal2 = false;
		if (timestep >= (int) paths[a1]->size())
			cardinal1 = true;
		else //if (!paths[a1]->at(0).is_single())
		{
			mdd_helper.findSingletons(node, a1, *paths[a1]);
		}
		if (timestep >= (int) paths[a2]->size())
			cardinal2 = true;
		else //if (!paths[a2]->at(0).is_single())
		{
			mdd_helper.findSingletons(node, a2, *paths[a2]);
		}

		if (type == constraint_type::EDGE) // Edge conflict
		{
			cardinal1 = paths[a1]->at(timestep).is_single() && paths[a1]->at(timestep - 1).is_single();
			cardinal2 = paths[a2]->at(timestep).is_single() && paths[a2]->at(timestep - 1).is_single();
		}
		else // vertex conflict or target conflict
		{
			if (!cardinal1)
				cardinal1 = paths[a1]->at(timestep).is_single();
			if (!cardinal2)
				cardinal2 = paths[a2]->at(timestep).is_single();
		}

		/*int width_1 = 1, width_2 = 1;

		if (paths[a1]->size() > timestep){
		  width_1 = paths[a1]->at(timestep).mdd_width;
		}

		if (paths[a2]->size() > timestep){
		  width_2 = paths[a2]->at(timestep).mdd_width;
		}
		con -> mdd_width = width_1 * width_2;*/

		if (cardinal1 && cardinal2)
		{
			con->priority = conflict_priority::CARDINAL;
		}
		else if (cardinal1 || cardinal2)
		{
			con->priority = conflict_priority::SEMI;
		}
		else
		{
			con->priority = conflict_priority::NON;
		}

		// Corridor reasoning
		auto corridor = corridor_helper.run(con, paths, node);
		if (corridor != nullptr)
		{
			corridor->priority = con->priority;
			computePriorityForConflict(*corridor, node);
			node.conflicts.push_back(corridor);
			continue;
		}

		// Target Reasoning
		if (con->type == conflict_type::TARGET)
		{
			computePriorityForConflict(*con, node);
			node.conflicts.push_back(con);
			continue;
		}

		// Rectangle reasoning
		if (rectangle_helper.strategy != rectangle_strategy::NR &&
			(int) paths[con->a1]->size() > timestep &&
			(int) paths[con->a2]->size() > timestep && //conflict happens before both agents reach their goal locations
			type == constraint_type::VERTEX && // vertex conflict
			con->priority != conflict_priority::CARDINAL) // not caridnal vertex conflict
		{
			auto mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
			auto mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());
			auto rectangle = rectangle_helper.run(paths, timestep, a1, a2, mdd1, mdd2);
			if (rectangle != nullptr)
			{
				computePriorityForConflict(*rectangle, node);
				node.conflicts.push_back(rectangle);
				continue;
			}
		}

		// Mutex reasoning
		if (mutex_reasoning)
		{
			// TODO mutex reasoning is per agent pair, don't do duplicated work...
			auto mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
			auto mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());

			auto mutex_conflict = mutex_helper.run(paths, a1, a2, node, mdd1, mdd2);

			if (mutex_conflict != nullptr && (*mutex_conflict != *con)) // ignore the cases when mutex finds the same vertex constraints
			{
				computePriorityForConflict(*mutex_conflict, node);
				node.conflicts.push_back(mutex_conflict);
				continue;
			}
		}

		computePriorityForConflict(*con, node);
		node.conflicts.push_back(con);
	}


	// remove conflicts that cannot be chosen, to save some memory
	removeLowPriorityConflicts(node.conflicts);
}


void CBS::removeLowPriorityConflicts(list<shared_ptr<Conflict>>& conflicts) const
{
	if (conflicts.empty())
		return;
	unordered_map<int, shared_ptr<Conflict> > keep;
	list<shared_ptr<Conflict>> to_delete;
	for (const auto& conflict : conflicts)
	{
		int a1 = min(conflict->a1, conflict->a2), a2 = max(conflict->a1, conflict->a2);
		int key = a1 * num_of_agents + a2;
		auto p = keep.find(key);
		if (p == keep.end())
		{
			keep[key] = conflict;
		}
		else if (*(p->second) < *conflict)
		{
			to_delete.push_back(p->second);
			keep[key] = conflict;
		}
		else
		{
			to_delete.push_back(conflict);
		}
	}

	for (const auto& conflict : to_delete)
	{
		conflicts.remove(conflict);
	}
}


bool CBS::findPathForSingleAgent(CBSNode* node, int ag, int lowerbound, int direction)
{
	clock_t t = clock();
	// build reservation table
	// CAT cat(node->makespan + 1);  // initialized to false
	// updateReservationTable(cat, ag, *node);
	// find a path
	Path new_path = search_engines[ag]->findPath(*node, initial_constraints[ag], paths, ag, lowerbound, direction);
	num_LL_expanded += search_engines[ag]->num_expanded;
	num_LL_generated += search_engines[ag]->num_generated;
	runtime_build_CT += search_engines[ag]->runtime_build_CT;
	runtime_build_CAT += search_engines[ag]->runtime_build_CAT;
	runtime_path_finding += (double) (clock() - t) / CLOCKS_PER_SEC;
	if (!new_path.empty())
	{
		assert(!isSamePath(*paths[ag], new_path));
		node->paths.emplace_back(ag, new_path);
		node->g_val = node->g_val - (int) paths[ag]->size() + (int) new_path.size();
		paths[ag] = &node->paths.back().second;
		node->makespan = max(node->makespan, new_path.size() - 1);
		return true;
	}
	else
	{
		return false;
	}
}


bool CBS::generateChild(CBSNode* node, CBSNode* parent)
{
	int direction = 0;
	clock_t t1 = clock();
	node->parent = parent;
	node->g_val = parent->g_val;
	node->makespan = parent->makespan;
	node->depth = parent->depth + 1;
	int agent, x, y, t;
	constraint_type type;
	assert(node->constraints.size() > 0);
	tie(agent, x, y, t, type) = node->constraints.front();

	if (type == constraint_type::LEQLENGTH)
	{
		assert(node->constraints.size() <= 2);
		if ((int)node->constraints.size() == 2) // generated by corridor-target conflict
		{
			int a = get<0>(node->constraints.back()); // it is a G-length constraint or a range constraint on this agent
			int lowerbound = (int)paths[a]->size() - 1;
			// find the direction
			int constrain_location = get<1>(node->constraints.back());

			for (list<pair<int, Path>>::iterator it=node->paths.begin(); it != node->paths.end(); ++it) {
				if ((*it).first == a) {
					Path temp_path = (*it).second;
					for (PathEntry p : temp_path) {
						if (constrain_location == p.location) {
							direction = p.direction;
						}
					}
				}
			}
			direction = search_engines[a]->start_direction;
			if (!findPathForSingleAgent(node, a, lowerbound, direction))
			{
				runtime_generate_child += (double)(clock() - t1) / CLOCKS_PER_SEC;
				return false;
			}
		}
		for (int ag = 0; ag < num_of_agents; ag++)
		{
			if (ag == agent)
			{
				continue;
			}
			for (int i = t; i < (int) paths[ag]->size(); i++)
			{
				if (paths[ag]->at(i).location == x)
				{
					int lowerbound = (int) paths[ag]->size() - 1;
					direction = search_engines[ag]->start_direction;
					if (!findPathForSingleAgent(node, ag, lowerbound, direction))
					{
						runtime_generate_child += (double) (clock() - t1) / CLOCKS_PER_SEC;
						return false;
					}
					break;
				}
			}
		}
	}
	else if (type == constraint_type::POSITIVE_VERTEX)
	{
		assert(node->constraints.size() == 1);
		for (const auto& constraint : node->constraints)
		{
			tie(agent, x, y, t, type) = constraint;
			for (int ag = 0; ag < num_of_agents; ag++)
			{
				if (ag == agent)
				{
					continue;
				}
				if (getAgentLocation(ag, t) == x)
				{
					direction = search_engines[ag]->start_direction;
					if (!findPathForSingleAgent(node, ag, (int) paths[ag]->size() - 1, direction))
					{
						runtime_generate_child += (double) (clock() - t1) / CLOCKS_PER_SEC;
						return false;
					}
				}
			}
		}

	}
	else if (type == constraint_type::POSITIVE_EDGE)
	{
		assert(node->constraints.size() == 1);
		for (int ag = 0; ag < num_of_agents; ag++)
		{
			if (ag == agent)
			{
				continue;
			}
			int curr = getAgentLocation(ag, t);
			int prev = getAgentLocation(ag, t - 1);
			if (prev == x || curr == y ||
				(prev == y && curr == x))
			{
				direction = search_engines[ag]->start_direction;
				if (!findPathForSingleAgent(node, ag, (int) paths[ag]->size() - 1, direction))
				{
					runtime_generate_child += (double) (clock() - t1) / CLOCKS_PER_SEC;
					return false;
				}
			}
		}

	}
	else
	{
		int lowerbound = (int) paths[agent]->size() - 1;
		direction = search_engines[agent]->start_direction;
		if (!findPathForSingleAgent(node, agent, lowerbound, direction))
		{
			runtime_generate_child += (double) (clock() - t1) / CLOCKS_PER_SEC;
			return false;
		}
	}

	assert(!node->paths.empty());
	findConflicts(*node);
	heuristic_helper.computeQuickHeuristics(*node);
	runtime_generate_child += (double) (clock() - t1) / CLOCKS_PER_SEC;
	return true;
}


inline void CBS::pushNode(CBSNode* node)
{
	// update handles
	node->open_handle = open_list.push(node);
	num_HL_generated++;
	node->time_generated = num_HL_generated;
	if (node->g_val + node->h_val <= focal_list_threshold)
		node->focal_handle = focal_list.push(node);
	allNodes_table.push_back(node);
}


void CBS::printPaths() const
{
	cout << "printing paths" << endl;
	for (int i = 0; i < num_of_agents; i++)
	{
		cout << "Agent " << i << " (" << paths_found_initially[i].size() - 1 << " -->" <<
			 paths[i]->size() - 1 << "): ";
		for (const auto& t : *paths[i])
			cout << t.location << "->";
		cout << endl;
	}
}


// adding new nodes to FOCAL (those with min-f-val*f_weight between the old and new LB)
void CBS::updateFocalList()
{
	CBSNode* open_head = open_list.top();
	if (open_head->g_val + open_head->h_val > min_f_val)
	{
		if (screen == 3)
		{
			cout << "  Note -- FOCAL UPDATE!! from |FOCAL|=" << focal_list.size() << " with |OPEN|=" << open_list.size() << " to |FOCAL|=";
		}
		min_f_val = open_head->g_val + open_head->h_val;
		double new_focal_list_threshold = min_f_val * focal_w;
		for (CBSNode* n : open_list)
		{
			if (n->g_val + n->h_val > focal_list_threshold &&
				n->g_val + n->h_val <= new_focal_list_threshold)
				n->focal_handle = focal_list.push(n);
		}
		focal_list_threshold = new_focal_list_threshold;
		if (screen == 3)
		{
			cout << focal_list.size() << endl;
		}
	}
}


void CBS::printResults() const
{
	if (solution_cost >= 0) // solved
		cout << "Optimal,";
	else if (solution_cost == -1) // time_out
		cout << "Timeout,";
	else if (solution_cost == -2) // no solution
		cout << "No solutions,";
	else if (solution_cost == -3) // nodes out
		cout << "Nodesout,";

	cout << solution_cost << "," << runtime << "," <<
		 num_HL_expanded << "," << num_LL_expanded << "," << // HL_num_generated << "," << LL_num_generated << "," <<
		 min_f_val << "," << dummy_start->g_val << "," << dummy_start->g_val + dummy_start->h_val << "," <<
		 endl;
}

void CBS::saveResults(const string& fileName, const string& instanceName) const
{
	std::ifstream infile(fileName);
	bool exist = infile.good();
	infile.close();
	if (!exist)
	{
		ofstream addHeads(fileName);
		addHeads << "runtime,#high-level expanded,#high-level generated,#low-level expanded,#low-level generated," <<
				 "solution cost,min f value,root g value,root f value," <<
				 "#adopt bypasses," <<
				 "standard conflicts,rectangle conflicts,corridor conflicts,target conflicts,mutex conflicts," <<
				 "#merge MDDs,#solve 2 agents,#memoization," <<
				 "runtime of building heuristic graph,runtime of solving MVC," <<
				 "runtime of detecting conflicts," <<
				 "runtime of rectangle conflicts,runtime of corridor conflicts,runtime of mutex conflicts," <<
				 "runtime of building MDDs,runtime of building constraint tables,runtime of building CATs," <<
				 "runtime of path finding,runtime of generating child nodes," <<
                 "runtime_of_computing_second_heuristic,#cluster_solved,#cluster_memorized,#mutex_checking,#mutex_memorized," <<
                 "#second_heuristic_computed,#second_heuristic_increased,total_second_heuristic_increased," <<
                 "total cluster size,#cluster found,#node found bypass,#bypass found,total h value,total fh value,"<<
                 "max num cluster,max cluster size,max second h,max num bypass,"<<
                 "#fh computed,total fh,max fh,"<<
				 "preprocessing runtime,solver name,instance name" << endl;
		addHeads.close();
	}

	ofstream stats(fileName, std::ios::app);
	stats << runtime << "," <<
		  num_HL_expanded << "," << num_HL_generated << "," <<
		  num_LL_expanded << "," << num_LL_generated << "," <<

		  solution_cost << "," << min_f_val << "," << dummy_start->g_val << "," << dummy_start->g_val + dummy_start->h_val << "," <<

		  num_adopt_bypass << "," <<

		  num_standard_conflicts << "," << num_rectangle_conflicts << "," << num_corridor_conflicts << "," <<
		  num_target_conflicts << "," << num_mutex_conflicts << "," <<

		  heuristic_helper.num_merge_MDDs << "," <<
		  heuristic_helper.num_solve_2agent_problems << "," <<
		  heuristic_helper.num_memoization << "," <<
		  heuristic_helper.runtime_build_dependency_graph << "," <<
		  heuristic_helper.runtime_solve_MVC << "," <<

		  runtime_detect_conflicts << "," <<
		  rectangle_helper.accumulated_runtime << "," << corridor_helper.accumulated_runtime << "," << mutex_helper.accumulated_runtime << "," <<
		  mdd_helper.accumulated_runtime << "," << runtime_build_CT << "," << runtime_build_CAT << "," <<
		  runtime_path_finding << "," << runtime_generate_child << "," <<
          heuristic_helper.runtime_of_computing_second_heuristic << "," << heuristic_helper.num_cluster_solved << "," << heuristic_helper.num_cluster_memorized<< "," <<
          heuristic_helper.num_mutex_checking << "," <<heuristic_helper.num_mutex_memorized << "," << heuristic_helper.num_second_heuristic_computed << "," <<
          heuristic_helper.num_second_heuristic_increased << "," <<heuristic_helper.total_second_heuristic_increased << "," <<
          heuristic_helper.total_cluster_size << "," <<heuristic_helper.num_cluster_detected << "," <<
          heuristic_helper.num_node_found_bypass<< "," <<heuristic_helper.num_bypass_found << "," <<
          heuristic_helper.total_h_value << "," <<heuristic_helper.total_fh_value<< ","<<
          heuristic_helper.max_num_cluster << "," <<heuristic_helper.max_cluster_size<< ","<<
          heuristic_helper.max_second_h << "," <<heuristic_helper.max_num_bypass << ","<<
          heuristic_helper.total_fh_computed << ","<<heuristic_helper.total_fh_value << "," <<heuristic_helper.max_fh<< ","<<
    runtime_preprocessing << "," << getSolverName() << "," << instanceName << endl;
	stats.close();
}

void CBS::saveStats(const string &fileName, const string &instanceName) const
{
	ofstream stats(fileName + ".txt", std::ios::app);
	stats << instanceName << endl;
	stats << "agent 1,agent 2,node id,#expanded nodes, h" << endl;
	for (auto ins : heuristic_helper.sub_instances)
	{
		stats << get<0>(ins) << "," << get<1>(ins) << "," << get<2>(ins)->time_generated << "," << get<3>(ins) << "," << get<4>(ins) << endl;
	}
	stats.close();
	cout << "Write " << heuristic_helper.sub_instances.size() << " samples to files" << endl;
}


void CBS::saveCT(const string &fileName) const // write the CT to a file
{
	std::ofstream output;
	output.open(fileName, std::ios::out);
	output << "digraph G {" << endl;
	output << "size = \"5,5\";" << endl;
	output << "center = true;" << endl;
	for (auto node : allNodes_table)
	{
		output << node->time_generated << " [label=\"#" << node->time_generated 
					<< "\ng+h="<< node->g_val << "+" << node->h_val 
					<< "\nd=" << node->tie_breaking << "\"]" << endl;
		if (node == dummy_start)
			continue;
		output << node->parent->time_generated << " -> " << node->time_generated << " [label=\"";
		for (const auto &constraint : node->constraints)
			output << constraint;
		output << "\nAgents ";
        for (const auto &path : node->paths)
            output << path.first << "(+" << path.second.size() - paths_found_initially[path.first].size() << ") ";
        output << "\"]" << endl;
	}
	auto node = goal_node;
	while (node != nullptr)
	{
		output << node->time_generated << " [color=red]" << endl;
		node = node->parent;
	}
	output << "}" << endl;
	output.close();
}

// void CBS::savePaths(const string &fileName) const
// {
//     std::ofstream output;
//     output.open(fileName, std::ios::out);
//     for (int i = 0; i < num_of_agents; i++)
//     {
//         output << "Agent " << i << ": ";
//         for (const auto & t : *paths[i])
//             output << "(" << search_engines[0]->instance.getRowCoordinate(t.location)
//                    << "," << search_engines[0]->instance.getColCoordinate(t.location) << ")->";
//         output << endl;
//     }
//     output.close();
// }
void CBS::savePaths(const string &fileName) const {
	list<list<pair<int,int>>>result = pathMatrix();
	std::ofstream output;
    output.open(fileName, std::ios::out);
	for (const auto & l1 : result) {
		for (const auto & l2 : l1) {
			output << "(" << l2.first << "," << l2.second << ")->";
		}
		output << endl;
	}

}
list<list<pair<int,int>>> CBS::pathMatrix() const {
	list<list<pair<int,int>>> output;
	
	for (int i = 0; i < num_of_agents; i++) {
		list<pair<int,int>> agent_path;
		for (const auto & t : *paths[i]) {
			pair<int,int> new_pair;
			new_pair.first = t.location;
			new_pair.second = t.direction;
			agent_path.emplace_back(new_pair);
		}
		output.emplace_back(agent_path);
	}

	return output;
}


void CBS::printConflicts(const CBSNode &curr)
{
	for (const auto& conflict : curr.conflicts)
	{
		cout << *conflict << endl;
	}
	for (const auto& conflict : curr.unknownConf)
	{
		cout << *conflict << endl;
	}
}


string CBS::getSolverName() const
{
	string name;
	if (disjoint_splitting)
		name += "Disjoint ";
	switch (heuristic_helper.type)
	{
	case heuristics_type::ZERO:
		if (PC)
			name += "ICBS";
		else
			name += "CBS";
		break;
	case heuristics_type::CG:
		name += "CG";
		break;
	case heuristics_type::DG:
		name += "DG";
		break;
	case heuristics_type::WDG:
		name += "WDG";
		break;
	case STRATEGY_COUNT:
		break;
	}
	if (rectangle_helper.getName() != "NR")
		name += "+" + rectangle_helper.getName();
	if (corridor_helper.getName() != "NC")
    	name += "+" + corridor_helper.getName();
	if (target_reasoning)
		name += "+T";
	if (mutex_reasoning)
		name += "+M";
	if (bypass)
		name += "+BP";

    switch (heuristic_helper.ch_type)
    {
        case cluster_heuristics_type::N:
            name += "+N";
            break;
        case cluster_heuristics_type::BP:
            name += "+BP";
            break;
        case cluster_heuristics_type::CH:
            name += "+CH";
            break;
        case cluster_heuristics_type::CHBPNM:
            name += "+CHBPNM";
            break;
        case cluster_heuristics_type::CHBPNS:
            name += "+CHBPNS";
            break;
        case cluster_heuristics_type::CHBPNRM:
            name += "+CHBPNRM";
            break;
        case cluster_heuristics_type::CHBP:
            name += "+CHBP";
            break;
    }
	name += " with " + search_engines[0]->getName();
	return name;
}

bool CBS::solve(double _time_limit, int _cost_lowerbound, int _cost_upperbound)
{
	cout << "call solve" << endl;
	this->min_f_val = _cost_lowerbound;
	this->cost_upperbound = _cost_upperbound;
	this->time_limit = _time_limit;

	if (screen > 0) // 1 or 2
	{
		string name = getSolverName();
		name.resize(35, ' ');
		cout << name << ": ";
	}
	// set timer
	start = clock();

	generateRoot();

	while (!open_list.empty() && !solution_found)
	{
		// CBSNode* curr1 = open_list.top();
		// auto iterator = curr1->paths.begin();
		// advance(iterator, 2);
		// cout << (*iterator).second << endl;

		updateFocalList();
		if (min_f_val >= cost_upperbound)
		{
			solution_cost = (int) min_f_val;
			solution_found = false;
			break;
		}
		runtime = (double) (clock() - start) / CLOCKS_PER_SEC;
		if (runtime > time_limit || num_HL_expanded > node_limit
		    || heuristic_helper.sub_instances.size() >= MAX_NUM_STATS)
		{  // time/node out
			solution_cost = -1;
			solution_found = false;
			break;
		}
		CBSNode* curr = focal_list.top();
		focal_list.pop();
		open_list.erase(curr->open_handle);
		// takes the paths_found_initially and UPDATE all constrained paths found for agents from curr to dummy_start (and lower-bounds)
		updatePaths(curr);

		if (screen > 1)
			cout << endl << "Pop " << *curr << endl;

		if (curr->unknownConf.size() + curr->conflicts.size() == 0) //no conflicts
		{// found a solution (and finish the while look)
			solution_found = true;
			solution_cost = curr->g_val;
			goal_node = curr;
			break;
		}

		// debug
		/*if (curr->time_generated == 15 && num_of_agents > 2)
		{
			int a1 = 51, a2 = 54;
			auto mdd1 = mdd_helper.getMDD(*curr, a1, paths[a1]->size());
			cout << "The mdd of agent " << a1 << endl;
			mdd1->printNodes();
			auto mdd2 = mdd_helper.getMDD(*curr, a2, paths[a2]->size());
			cout << "The mdd of agent " << a2 << endl;
			mdd2->printNodes();
			for (int t = 0; t < min(paths[a1]->size(), paths[a2]->size()); t++)
			{
				if (paths[a1]->at(t).location == paths[a2]->at(t).location)
					rectangle_helper.printOverlapArea(paths, t, a1, a2, mdd1, mdd2);
			}
			cout << "The constraints " << endl;
			curr->printConstraints(a1);
			curr->printConstraints(a2);
		}*/
		/*if (curr->time_generated == 1 && num_of_agents > 2)
		{
			int a1 = 12, a2 = 23;
			auto mdd1 = mdd_helper.getMDD(*curr, a1, paths[a1]->size());
			cout << "The mdd of agent " << a1 << endl;
			mdd1->printNodes();
			auto mdd2 = mdd_helper.getMDD(*curr, a2, paths[a2]->size());
			cout << "The mdd of agent " << a2 << endl;
			mdd2->printNodes();
			for (int t = 0; t < min(paths[a1]->size(), paths[a2]->size()); t++)
			{
				if (paths[a1]->at(t).location == paths[a2]->at(t).location)
					rectangle_helper.printOverlapArea(paths, t, a1, a2, mdd1, mdd2);
			}
			cout << "The constraints " << endl;
			curr->printConstraints(a1);
			curr->printConstraints(a2);
		}*/

		if (!curr->h_computed) // heuristics has not been computed yet
		{

			if (PC) // prioritize conflicts
				classifyConflicts(*curr);
			runtime = (double) (clock() - start) / CLOCKS_PER_SEC;
			bool succ = heuristic_helper.computeInformedHeuristics(*curr, time_limit - runtime);
            if(curr->bypassFound){
                // classify the new conflicts for bypasses.
                classifyConflicts(*curr);
            }
			runtime = (double) (clock() - start) / CLOCKS_PER_SEC;
			if (runtime > time_limit)
			{  // timeout
				solution_cost = -1;
				solution_found = false;
				break;
			}
			if (!succ) // no solution, so prune this node
			{
				cout << "no solution" << endl;
				curr->clear();
				continue;
			}

			// reinsert the node
			curr->open_handle = open_list.push(curr);
			if (curr->g_val + curr->h_val <= focal_list_threshold)
				curr->focal_handle = focal_list.push(curr);
			if (screen == 2)
			{
				cout << "	Reinsert " << *curr << endl;
			}
			continue;
		}



		//Expand the node
		num_HL_expanded++;
		curr->time_expanded = num_HL_expanded;
		bool foundBypass = true;
		while (foundBypass)
		{

			if (curr->unknownConf.size() + curr->conflicts.size() == 0) //no conflicts
			{// found a solution (and finish the while look)
				solution_found = true;
				solution_cost = curr->g_val;
				goal_node = curr;
				break;
			}
			foundBypass = false;
			CBSNode* child[2] = { new CBSNode(), new CBSNode() };

			curr->conflict = chooseConflict(*curr);

			if (disjoint_splitting && curr->conflict->type == conflict_type::STANDARD)
			{
				int first = (bool) (random_tie_breaker());
				if (first) // disjoint splitting on the first agent
				{
					child[0]->constraints = curr->conflict->constraint1;
					int a, x, y, t;
					constraint_type type;
					tie(a, x, y, t, type) = curr->conflict->constraint1.back();
					if (type == constraint_type::VERTEX)
					{
						child[1]->constraints.emplace_back(a, x, y, t, constraint_type::POSITIVE_VERTEX);
					}
					else
					{
						assert(type == constraint_type::EDGE);
						child[1]->constraints.emplace_back(a, x, y, t, constraint_type::POSITIVE_EDGE);
					}
				}
				else // disjoint splitting on the second agent
				{
					child[1]->constraints = curr->conflict->constraint2;
					int a, x, y, t;
					constraint_type type;
					tie(a, x, y, t, type) = curr->conflict->constraint2.back();
					if (type == constraint_type::VERTEX)
					{
						child[0]->constraints.emplace_back(a, x, y, t, constraint_type::POSITIVE_VERTEX);
					}
					else
					{
						assert(type == constraint_type::EDGE);
						child[0]->constraints.emplace_back(a, x, y, t, constraint_type::POSITIVE_EDGE);
					}
				}
			}
			else
			{
				child[0]->constraints = curr->conflict->constraint1;
				child[1]->constraints = curr->conflict->constraint2;
				if (curr->conflict->type == conflict_type::RECTANGLE && rectangle_helper.strategy == rectangle_strategy::DR)
				{
					int i = (bool)(random_tie_breaker());
					for (const auto constraint : child[1 - i]->constraints)
					{
						child[i]->constraints.emplace_back(get<0>(constraint), get<1>(constraint), get<2>(constraint), get<3>(constraint), 
																							constraint_type::POSITIVE_BARRIER);
					}
				}
				else if (curr->conflict->type == conflict_type::CORRIDOR && corridor_helper.getStrategy() == corridor_strategy::DC)
				{
					int i = (bool)(random_tie_breaker());
					assert(child[1 - i]->constraints.size() == 1);
					auto constraint = child[1 - i]->constraints.front();
					child[i]->constraints.emplace_back(get<0>(constraint), get<1>(constraint), get<2>(constraint), get<3>(constraint),
						constraint_type::POSITIVE_RANGE);
				}
			}

			if (screen > 1)
				cout << "	Expand " << *curr << endl <<
					 "	on " << *(curr->conflict) << endl;

			bool solved[2] = { false, false };
			vector<vector<PathEntry>*> copy(paths);

			for (int i = 0; i < 2; i++)
			{
				if (i > 0)
					paths = copy;
				solved[i] = generateChild(child[i], curr);
				if (!solved[i])
				{
					delete child[i];
					continue;
				}
				if (child[i]->g_val + child[i]->h_val == min_f_val && curr->unknownConf.size() + curr->conflicts.size() == 0) //no conflicts
				{// found a solution (and finish the while look)
					break;
				}
				else if (bypass && child[i]->g_val == curr->g_val && child[i]->tie_breaking < curr->tie_breaking) // Bypass1
				{
					if (i == 1 && !solved[0])
						continue;
					foundBypass = true;
					num_adopt_bypass++;
					curr->conflicts = child[i]->conflicts;
					curr->unknownConf = child[i]->unknownConf;
					curr->tie_breaking = child[i]->tie_breaking;
					curr->conflict = nullptr;
					for (const auto& path : child[i]->paths) // update paths
					{
						auto p = curr->paths.begin();
						while (p != curr->paths.end())
						{
							if (path.first == p->first)
							{
								p->second = path.second;
								paths[p->first] = &p->second;
								break;
							}
							++p;
						}
						if (p == curr->paths.end())
						{
							curr->paths.emplace_back(path);
							paths[path.first] = &curr->paths.back().second;
						}
					}
					if (screen > 1)
					{
						cout << "	Update " << *curr << endl;
					}
					break;
				}
			}
			if (foundBypass)
			{
				for (auto & i : child)
				{
					delete i;
					i = nullptr;
				}
                if (PC) // prioritize conflicts
                    classifyConflicts(*curr); // classify the new-detected conflicts
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					if (solved[i])
					{
						pushNode(child[i]);
						if (screen > 1)
						{
							cout << "		Generate " << *child[i] << endl;
						}
					}
				}
			}
		}
		if (curr->conflict != nullptr)
		{

			switch (curr->conflict->type)
			{
			case conflict_type::RECTANGLE:
				num_rectangle_conflicts++;
				break;
			case conflict_type::CORRIDOR:
				num_corridor_conflicts++;
				break;
			case conflict_type::TARGET:
				num_target_conflicts++;
				break;
			case conflict_type::STANDARD:
				num_standard_conflicts++;
				break;
			case conflict_type::MUTEX:
				num_mutex_conflicts++;
				break;
			}
		}
		curr->clear();

	}  // end of while loop


	runtime = (double) (clock() - start) / CLOCKS_PER_SEC;
	if (solution_found && !validateSolution())
	{
		cout << "Solution invalid!!!" << endl;
		printPaths();
		exit(-1);
	}
	if (screen == 2)
        printPaths();
	if (screen > 0) // 1 or 2
		printResults();
	return solution_found;
}



CBS::CBS(vector<SingleAgentSolver*>& search_engines,
		 const vector<ConstraintTable>& initial_constraints,
         vector<Path>& paths_found_initially, int screen) :
		screen(screen), focal_w(1),
		initial_constraints(initial_constraints), paths_found_initially(paths_found_initially),
		search_engines(search_engines),
		mdd_helper(initial_constraints, search_engines),
		rectangle_helper(search_engines[0]->instance),
		mutex_helper(search_engines[0]->instance, initial_constraints),
		corridor_helper(search_engines, initial_constraints),
		heuristic_helper(search_engines.size(), paths, search_engines, initial_constraints, mdd_helper)
{
	num_of_agents = (int) search_engines.size();
	mutex_helper.search_engines = search_engines;
}

CBS::CBS(const Instance& instance, bool sipp, int screen) :
		screen(screen), focal_w(1),
		num_of_agents(instance.getDefaultNumberOfAgents()),
		mdd_helper(initial_constraints, search_engines),
		rectangle_helper(instance),
		mutex_helper(instance, initial_constraints),
		corridor_helper(search_engines, initial_constraints),
		heuristic_helper(instance.getDefaultNumberOfAgents(), paths, search_engines, initial_constraints, mdd_helper)
{
	clock_t t = clock();
	initial_constraints.resize(num_of_agents,
							   ConstraintTable(instance.num_of_cols, instance.map_size));

	search_engines.resize(num_of_agents);
	for (int i = 0; i < num_of_agents; i++)
	{
		if (sipp)
			search_engines[i] = new SIPP(instance, i);
		else
			search_engines[i] = new SpaceTimeAStar(instance, i);
			cout << search_engines[i]->start_direction << endl;

		initial_constraints[i].goal_location = search_engines[i]->goal_location;
	}
	runtime_preprocessing = (double) (clock() - t) / CLOCKS_PER_SEC;

	mutex_helper.search_engines = search_engines;

	if (screen >= 2) // print start and goals
	{
		instance.printAgents();
	}
}

bool CBS::generateRoot()
{	
	cout << "generating root" << endl;
	dummy_start = new CBSNode();
	dummy_start->g_val = 0;
	paths.resize(num_of_agents, nullptr);

	mdd_helper.init(num_of_agents);
	heuristic_helper.init();

	// initialize paths_found_initially
	if (paths_found_initially.empty())
	{
		paths_found_initially.resize(num_of_agents);

		// generate a random permutation of agent indices
		vector<int> agents(num_of_agents);
		for (int i = 0; i < num_of_agents; i++)
		{
			agents[i] = i;
		}

		if (randomRoot)
		{
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(std::begin(agents), std::end(agents), g);
		}

		for (auto i : agents)
		{

			//CAT cat(dummy_start->makespan + 1);  // initialized to false
			//updateReservationTable(cat, i, *dummy_start);
			int starting_direction = search_engines[i]->start_direction;
			paths_found_initially[i] = search_engines[i]->findPath(*dummy_start, initial_constraints[i], paths, i, 0, starting_direction);

			if (paths_found_initially[i].empty())
			{
				cout << "No path exists for agent " << i << endl;
				return false;
			}
			paths[i] = &paths_found_initially[i];
			dummy_start->makespan = max(dummy_start->makespan, paths_found_initially[i].size() - 1);
			dummy_start->g_val += (int) paths_found_initially[i].size() - 1;
			num_LL_expanded += search_engines[i]->num_expanded;
			num_LL_generated += search_engines[i]->num_generated;
		}
	}
	else
	{
		for (int i = 0; i < num_of_agents; i++)
		{
			paths[i] = &paths_found_initially[i];
			dummy_start->makespan = max(dummy_start->makespan, paths_found_initially[i].size() - 1);
			dummy_start->g_val += (int) paths_found_initially[i].size() - 1;
		}
	}

	// generate dummy start and update data structures		
	dummy_start->h_val = 0;
	dummy_start->depth = 0;
	dummy_start->open_handle = open_list.push(dummy_start);
	dummy_start->focal_handle = focal_list.push(dummy_start);

	num_HL_generated++;
	dummy_start->time_generated = num_HL_generated;
	allNodes_table.push_back(dummy_start);
	findConflicts(*dummy_start);
	// We didn't compute the node-selection tie-breaking value for the root node
	// since it does not need it.
	min_f_val = max(min_f_val, (double) dummy_start->g_val);
	focal_list_threshold = min_f_val * focal_w;

	if (screen >= 2) // print start and goals
	{
		printPaths();
	}

	return true;
}

inline void CBS::releaseNodes()
{
	open_list.clear();
	focal_list.clear();
	for (auto node : allNodes_table)
		delete node;
	allNodes_table.clear();
}



/*inline void CBS::releaseOpenListNodes()
{
	while (!open_list.empty())
	{
		CBSNode* curr = open_list.top();
		open_list.pop();
		delete curr;
	}
}*/

CBS::~CBS()
{
	releaseNodes();
	mdd_helper.clear();
}

void CBS::clearSearchEngines()
{
	for (auto s : search_engines)
		delete s;
	search_engines.clear();
}


bool CBS::validateSolution() const
{
	for (int a1 = 0; a1 < num_of_agents; a1++)
	{
		for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
		{
			size_t min_path_length = paths[a1]->size() < paths[a2]->size() ? paths[a1]->size() : paths[a2]->size();
			for (size_t timestep = 0; timestep < min_path_length; timestep++)
			{
				int loc1 = paths[a1]->at(timestep).location;
				int loc2 = paths[a2]->at(timestep).location;
				if (loc1 == loc2)
				{
					cout << "Agents " << a1 << " and " << a2 << " collides at " << loc1 << " at timestep " << timestep << endl;
					return false;
				}
				else if (timestep < min_path_length - 1
						 && loc1 == paths[a2]->at(timestep + 1).location
						 && loc2 == paths[a1]->at(timestep + 1).location)
				{
					cout << "Agents " << a1 << " and " << a2 << " collides at (" <<
						 loc1 << "-->" << loc2 << ") at timestep " << timestep << endl;
					return false;
				}
			}
			if (paths[a1]->size() != paths[a2]->size())
			{
				int a1_ = paths[a1]->size() < paths[a2]->size() ? a1 : a2;
				int a2_ = paths[a1]->size() < paths[a2]->size() ? a2 : a1;
				int loc1 = paths[a1_]->back().location;
				for (size_t timestep = min_path_length; timestep < paths[a2_]->size(); timestep++)
				{
					int loc2 = paths[a2_]->at(timestep).location;
					if (loc1 == loc2)
					{
						cout << "Agents " << a1 << " and " << a2 << " collides at " << loc1 << " at timestep " << timestep << endl;
						return false; // It's at least a semi conflict			
					}
				}
			}
		}
	}
	return true;
}

inline int CBS::getAgentLocation(int agent_id, size_t timestep) const
{
	size_t t = max(min(timestep, paths[agent_id]->size() - 1), (size_t) 0);
	return paths[agent_id]->at(t).location;
}


// used for rapid random  restart
void CBS::clear()
{
	mdd_helper.clear();
	heuristic_helper.clear();
	releaseNodes();
	paths.clear();
	paths_found_initially.clear();
	dummy_start = nullptr;
	goal_node = nullptr;
	solution_found = false;
	solution_cost = -2;
}
