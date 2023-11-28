//#pragma warning(disable: 4996) // I added this line to disable error C4996 caused by CPLEX
//#include <ilcplex/ilocplex.h>
#include "CBSHeuristic.h"
#include "CBS.h"
#include <queue>


void CBSHeuristic::computeQuickHeuristics(CBSNode& node) const // for non-root node
{
	node.h_val = max(0, node.parent->g_val + node.parent->h_val - node.g_val); // pathmax
	set<pair<int, int>> conflicting_agents;
    for (const auto& conflict : node.unknownConf)
    {
        auto agents = make_pair(min(conflict->a1, conflict->a2), max(conflict->a1, conflict->a2));
        if (conflicting_agents.find(agents) == conflicting_agents.end())
        {
            conflicting_agents.insert(agents);
        }
    }
    node.tie_breaking = (int)(node.conflicts.size() + conflicting_agents.size());
	copyConflictGraph(node, *node.parent);
    //inherits the results of mutex propagation from parent.
    copyMutexGraph(node, *node.parent);
    //inherits all cluster from the parent.
    copyCluster(node,*node.parent);
}


bool CBSHeuristic::computeInformedHeuristics(CBSNode& curr, double _time_limit)
{


	curr.h_computed = true;
	// create conflict graph
	start_time = clock();
	this->time_limit = _time_limit;
	int num_of_CGedges;
	vector<int> HG(num_of_agents * num_of_agents, 0); // heuristic graph
    vector<bool> excluded_agents(num_of_agents,false); // Agents need to be excluded for computing cluster heuristic//
	int h = -1;
	switch (type)
	{
	case heuristics_type::ZERO:
		h = 0;
		break;
	case heuristics_type::CG:
		buildCardinalConflictGraph(curr, HG,  excluded_agents, num_of_CGedges);
		// Minimum Vertex Cover
		if (curr.parent == nullptr ||  // root node of CBS tree
		    target_reasoning || corridor_reasoning == corridor_strategy::STC || corridor_reasoning == corridor_strategy::GC ||
                corridor_reasoning == corridor_strategy::DC || disjoint_splitting) // when we are allowed
		    // to replan for multiple agents, the incremental method is not correct any longer.
			h = minimumVertexCover(HG);
		else
        {
		    assert(curr.paths.size() == 1);
            h = minimumVertexCover(HG, curr.parent->h_val, num_of_agents, num_of_CGedges);

        }
		break;
	case heuristics_type::DG:
		if (!buildDependenceGraph(curr, HG, excluded_agents, num_of_CGedges))
			return false;
		// Minimum Vertex Cover
		if (curr.parent == nullptr || // root node of CBS tree
		    target_reasoning || corridor_reasoning == corridor_strategy::STC || corridor_reasoning == corridor_strategy::GC ||
		    corridor_reasoning == corridor_strategy::DC || disjoint_splitting) // when we are allowed to replan for multiple agents, the incremental method is not correct any longer.
			h = minimumVertexCover(HG);
		else
			h = minimumVertexCover(HG, curr.parent->h_val, num_of_agents, num_of_CGedges);
		break;
	case heuristics_type::WDG:
		if (!buildWeightedDependencyGraph(curr, HG,  excluded_agents))
			return false;
		h = minimumWeightedVertexCover(HG);
        if( h > 0 && h < 1000){
            total_fh_computed ++;
            total_fh_value += h;
            if(max_fh < h){
                max_fh = h ;
            }
        }
		break;
	}
	if (h < 0)
		return false;


    int ch = 0;
    clock_t t_compute_second_heuristic = clock();
    uint64_t  bypassIncreased = num_bypass_found;
    switch (ch_type) {
        case cluster_heuristics_type::N: {
            ch = 0;
            break;
        }
        case cluster_heuristics_type::BP:
        case cluster_heuristics_type::CH:
        case cluster_heuristics_type::CHBPNM:
        case cluster_heuristics_type::CHBPNS:
        case cluster_heuristics_type::CHBPNRM:
        case cluster_heuristics_type::CHBP:
        {
            num_second_heuristic_computed ++;
            // the main algorithm of CHBP, which also refers to Algorithm 1 in our paper.
            ch = computeClusterHeuristicAndBypass(curr,excluded_agents,time_limit);
            break;
        }
    }
    runtime_of_computing_second_heuristic += (double)(clock() - t_compute_second_heuristic) / CLOCKS_PER_SEC;
    if (ch < 0)
        return false;

    if(ch > 0 && ch  < 1000){
        num_cluster_detected += curr.cluster_found.size();
        if(max_num_cluster < curr.cluster_found.size()){
            max_num_cluster = curr.cluster_found.size();
        }

        for(auto c : curr.cluster_found){
            total_cluster_size  += c->agent_id.size();
            if(max_cluster_size < c->agent_id.size()){
                max_cluster_size = c->agent_id.size();
            }
        }
        num_second_heuristic_increased++;
        total_second_heuristic_increased += ch ;
        if(max_second_h < ch ){
            max_second_h = ch ;
        }
    }
    if(curr.bypassFound){
        if(max_num_bypass < num_bypass_found -bypassIncreased){
            max_num_bypass = num_bypass_found - bypassIncreased;
        }
        num_node_found_bypass ++;
    }

    h += ch ;
    if( h > 0 && h < 1000){
        total_h_value += h;
    }
	curr.h_val = max(h, curr.h_val);
	return true;
}


void CBSHeuristic::buildCardinalConflictGraph(CBSNode& curr, vector<int>& CG, vector<bool>& excluded_agents, int& num_of_CGedges)
{
	num_of_CGedges = 0;
	for (const auto& conflict : curr.conflicts)
	{
		if (conflict->priority == conflict_priority::CARDINAL)
		{
			int a1 = conflict->a1;
			int a2 = conflict->a2;
			if (!CG[a1 * num_of_agents + a2])
			{
				CG[a1 * num_of_agents + a2] = true;
				CG[a2 * num_of_agents + a1] = true;
                excluded_agents[a1] = true;
                excluded_agents[a2] = true;
				num_of_CGedges++;
			}
		}
	}
	runtime_build_dependency_graph += (double)(clock() - start_time) / CLOCKS_PER_SEC;
}


bool CBSHeuristic::buildDependenceGraph(CBSNode& node, vector<int>& CG, vector<bool>& excluded_agents, int& num_of_CGedges)
{
	for (auto& conflict : node.conflicts)
	{
		int a1 = min(conflict->a1, conflict->a2);
		int a2 = max(conflict->a1, conflict->a2);
		int idx = a1 * num_of_agents + a2;
		if (conflict->priority == conflict_priority::CARDINAL)
		{
			node.conflictGraph[idx] = 1;
		}
		else if(node.conflictGraph.find(idx) == node.conflictGraph.end())
		{
			auto got = lookupTable[a1][a2].find(DoubleConstraintsHasher(a1, a2, &node));
			if (got != lookupTable[a1][a2].end()) // check the lookup table first
			{
				num_memoization++;
				node.conflictGraph[idx] = got->second;
			}
			else
			{
				node.conflictGraph[idx] = dependent(a1, a2, node)? 1 : 0;
				lookupTable[a1][a2][DoubleConstraintsHasher(a1, a2, &node)] = node.conflictGraph[idx];
			}
		}
		if ((clock() - start_time) / CLOCKS_PER_SEC > time_limit) // run out of time
		{
			runtime_build_dependency_graph += (double)(clock() - start_time) / CLOCKS_PER_SEC;
			return false;
		}
	}

	num_of_CGedges = 0;
	for (int i = 0; i < num_of_agents; i++)
	{
		for (int j = i + 1; j < num_of_agents; j++)
		{
			auto got = node.conflictGraph.find(i * num_of_agents + j);
			if (got != node.conflictGraph.end() && got->second > 0)
			{
				CG[i * num_of_agents + j] = got->second;
				CG[j * num_of_agents + i] = got->second;
                excluded_agents[i] = true;
                excluded_agents[j] = true;
				num_of_CGedges++;
			}
		}
	}
	runtime_build_dependency_graph += (double)(clock() - start_time) / CLOCKS_PER_SEC;
	return true;
}


bool CBSHeuristic::buildWeightedDependencyGraph(CBSNode& node, vector<int>& CG, vector<bool>& excluded_agents)
{
	for (const auto& conflict : node.conflicts)
	{
		int a1 = min(conflict->a1, conflict->a2);
		int a2 = max(conflict->a1, conflict->a2);
		int idx = a1 * num_of_agents + a2;
		if (node.conflictGraph.find(idx) != node.conflictGraph.end())
			continue;
		auto got = lookupTable[a1][a2].find(DoubleConstraintsHasher(a1, a2, &node));
		if (got != lookupTable[a1][a2].end()) // check the lookup table first
		{
			num_memoization++;
			node.conflictGraph[idx] = got->second;
		}
		else if (rectangle_reasoning == rectangle_strategy::RM ||
                rectangle_reasoning == rectangle_strategy::GR ||
                rectangle_reasoning == rectangle_strategy::DR ||
                save_stats)
		{
			node.conflictGraph[idx] = solve2Agents(a1, a2, node, false);
			assert(node.conflictGraph[idx] >= 0);
			lookupTable[a1][a2][DoubleConstraintsHasher(a1, a2, &node)] = node.conflictGraph[idx];
		}
		else
		{ 
			bool cardinal = conflict->priority == conflict_priority::CARDINAL;
			if (!cardinal && !mutex_reasoning) // using merging MDD methods before runing 2-agent instance
			{
				cardinal = dependent(a1, a2, node);
			}
			if (cardinal) // run 2-agent solver only for dependent agents
			{
				node.conflictGraph[idx] = solve2Agents(a1, a2, node, cardinal);		
				assert(node.conflictGraph[idx] >= 1);
			}
			else
			{
				node.conflictGraph[idx] = 0;
			}
			lookupTable[a1][a2][DoubleConstraintsHasher(a1, a2, &node)] = node.conflictGraph[idx];
		}

		if (node.conflictGraph[idx] == MAX_COST) // no solution
		{
			return false;
		}
		if ((clock() - start_time) / CLOCKS_PER_SEC > time_limit) // run out of time
		{
			runtime_build_dependency_graph += (double)(clock() - start_time) / CLOCKS_PER_SEC;
			return false;
		}
	}

	for (int i = 0; i < num_of_agents; i++)
	{
		for (int j = i + 1; j < num_of_agents; j++)
		{
			auto got = node.conflictGraph.find(i * num_of_agents + j);
			if (got != node.conflictGraph.end() && got->second > 0)
			{
				CG[i * num_of_agents + j] = got->second;
				CG[j * num_of_agents + i] = got->second;
                excluded_agents[i] = true;
                excluded_agents[j] = true;
			}
		}
	}
	runtime_build_dependency_graph += (double)(clock() - start_time) / CLOCKS_PER_SEC;
	return true;
}


int CBSHeuristic::solve2Agents(int a1, int a2, const CBSNode& node, bool cardinal)
{
    cout << a1 << " and " << a2 << endl; 
	vector<SingleAgentSolver*> engines(2);
	engines[0] = search_engines[a1];

    cout << "path 1 start" << endl;
    cout << engines[0]->start_direction << endl;
	engines[1] = search_engines[a2];
    cout << "path 2 start" << endl;
    cout << engines[1]->start_direction << endl;

	vector<vector<PathEntry>> initial_paths(2);
	initial_paths[0] = *paths[a1];
    cout << "path 1" << endl;
    cout << initial_paths[0] << endl;
	initial_paths[1] = *paths[a2];
    cout << "path 2" << endl;
    cout << initial_paths[1] << endl;
	vector<ConstraintTable> constraints{
		ConstraintTable(initial_constraints[a1]),
		ConstraintTable(initial_constraints[a2]) };
	constraints[0].build(node, a1);
	constraints[1].build(node, a2);
	CBS cbs(engines, constraints, initial_paths, screen);
	cbs.setPrioritizeConflicts(PC);
	cbs.setHeuristicType(heuristics_type::CG);
	cbs.setDisjointSplitting(disjoint_splitting);
	cbs.setBypass(false); // I guess that bypassing does not help two-agent path finding???
	cbs.setRectangleReasoning(rectangle_reasoning);
	cbs.setCorridorReasoning(corridor_reasoning);
	cbs.setTargetReasoning(target_reasoning);
	cbs.setMutexReasoning(mutex_reasoning);
	cbs.setNodeLimit(node_limit);
    cbs.setClusterHeuristicType(cluster_heuristics_type::N);
	double runtime = (double)(clock() - start_time) / CLOCKS_PER_SEC;
	int root_g = (int)initial_paths[0].size() - 1 + (int)initial_paths[1].size() - 1;
	int lowerbound = root_g;
	int upperbound = MAX_COST;
	if (cardinal)
		lowerbound += 1;
	cbs.solve(time_limit - runtime, lowerbound, upperbound);
    initial_paths[0] = *paths[a1];
    cout << "new path 1" << endl;
    cout << initial_paths[0] << endl;
	initial_paths[1] = *paths[a2];
    cout << "new path 2" << endl;
    cout << initial_paths[1] << endl;
	num_solve_2agent_problems++;
	int rst;
	if (cbs.runtime > time_limit - runtime || cbs.num_HL_expanded > node_limit) // time out or node out
		rst = (int)cbs.min_f_val - root_g; // using lowerbound to approximate
	else if (cbs.solution_cost  < 0) // no solution
		rst = MAX_COST;
	else
	{
		rst = cbs.solution_cost - root_g;
	}

	// For statistic study!!!
	/*if (cbs.num_HL_expanded > 1 && rst >= 1)
	{
		cout << a1 << "," << a2 << "," << node.time_generated << "," << cbs.num_HL_expanded << "," << rst << endl;
	}*/
	if (save_stats)
	{
		sub_instances.emplace_back(a1, a2, &node, cbs.num_HL_expanded, rst);
		// cout << sub_instances.size() << endl;
	}

	assert(rst >= 0);
	return rst;
}


int CBSHeuristic::MVConAllConflicts(CBSNode& curr)
{
	auto G = buildConflictGraph(curr);
	return  minimumVertexCover(G);
}


vector<int> CBSHeuristic::buildConflictGraph(const CBSNode& curr) const
{
	vector<int> G(num_of_agents * num_of_agents, 0);
	for (const auto& conflict : curr.conflicts)
	{
		int a1 = conflict->a1;
		int a2 = conflict->a2;
		if (!G[a1 * num_of_agents + a2])
		{
			G[a1 * num_of_agents + a2] = true;
			G[a2 * num_of_agents + a1] = true;
		}
	}
	return G;
}

int CBSHeuristic::minimumVertexCover(const vector<int>& CG)
{
    clock_t t = clock();
	int rst = 0;
	std::vector<bool> done(num_of_agents, false);
	for (int i = 0; i < num_of_agents; i++)
	{
		if (done[i])
			continue;
		std::vector<int> indices;
		indices.reserve(num_of_agents);
		std::queue<int> Q;
		Q.push(i);
		done[i] = true;
		while (!Q.empty())
		{
			int j = Q.front(); Q.pop();
			indices.push_back(j);
			for (int k = 0; k < num_of_agents; k++)
			{
				if (CG[j * num_of_agents + k] > 0)
				{
					if (!done[k])
					{
						Q.push(k);
						done[k] = true;
					}
				}
				else if (CG[k * num_of_agents + j] > 0)
				{
					if (!done[k])
					{
						Q.push(k);
						done[k] = true;
					}
				}
			}
		}
		if ((int) indices.size() == 1) //one node -> no edges -> mvc = 0
			continue;
		else if ((int) indices.size() == 2) // two nodes -> only one edge -> mvc = 1
		{
			rst += 1; // add edge weight
			continue;
		}

		std::vector<int> subgraph(indices.size() * indices.size(), 0);
		int num_edges = 0;
		for (int j = 0; j < (int) indices.size(); j++)
		{
			for (int k = j + 1; k < (int) indices.size(); k++)
			{
				subgraph[j * indices.size() + k] = CG[indices[j] * num_of_agents + indices[k]];
				subgraph[k * indices.size() + j] = CG[indices[k] * num_of_agents + indices[j]];
				if (subgraph[j * indices.size() + k] > 0)
					num_edges++;
			}
		}
        if ((int)indices.size() > DP_node_threshold)
        {
            rst += greedyMatching(subgraph, (int)indices.size());
            double runtime = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            if (runtime > time_limit)
            {
                runtime_solve_MVC += (double)(clock() - t) / CLOCKS_PER_SEC;
                return -1; // run out of time
            }
        }
        else {
            for (int i = 1; i < (int) indices.size(); i++) {
                if (KVertexCover(subgraph, (int) indices.size(), num_edges, i, (int) indices.size())) {
                    rst += i;
                    break;
                }
                double runtime = (double) (clock() - start_time) / CLOCKS_PER_SEC;
                if (runtime > time_limit)
                    return -1; // run out of time
            }
        }
	}
    runtime_solve_MVC += (double)(clock() - t) / CLOCKS_PER_SEC;
	return rst;
}


int CBSHeuristic::minimumVertexCover(const std::vector<int>& CG, int old_mvc, int cols, int num_of_CGedges)
{
    assert(old_mvc >= 0);
	clock_t t = clock();
	int rst = 0;
	if (num_of_CGedges < 2)
		return num_of_CGedges;
	// Compute #CG nodes that have edges
	int num_of_CGnodes = 0;
	for (int i = 0; i <  cols; i++)
	{
		for (int j = 0; j <  cols; j++)
		{
			if (CG[i * cols + j] > 0)
			{
				num_of_CGnodes++;
				break;
			}
		}
	}

	if (num_of_CGnodes > DP_node_threshold)
	{
        runtime_solve_MVC += (double)(clock() - t) / CLOCKS_PER_SEC;
		return minimumVertexCover(CG);
	}
	else
	{
		if (KVertexCover(CG, num_of_CGnodes, num_of_CGedges, old_mvc - 1, cols))
			rst = old_mvc - 1;
		else if (KVertexCover(CG, num_of_CGnodes, num_of_CGedges, old_mvc, cols))
			rst = old_mvc;
		else
			rst = old_mvc + 1;
	}
	runtime_solve_MVC += (double)(clock() - t) / CLOCKS_PER_SEC;
	return rst;
}

// Whether there exists a k-vertex cover solution
bool CBSHeuristic::KVertexCover(const std::vector<int>& CG, int num_of_CGnodes, int num_of_CGedges, int k, int cols)
{
	double runtime = (double) (clock() - start_time) / CLOCKS_PER_SEC;
	if (runtime > time_limit)
		return true; // run out of time
	if (num_of_CGedges == 0)
		return true;
	else if (num_of_CGedges > k * num_of_CGnodes - k)
		return false;

	std::vector<int> node(2);
	bool flag = true;
	for (int i = 0; i < cols - 1 && flag; i++) // to find an edge
	{
		for (int j = i + 1; j < cols && flag; j++)
		{
			if (CG[i * cols + j] > 0)
			{
				node[0] = i;
				node[1] = j;
				flag = false;
			}
		}
	}
	for (int i = 0; i < 2; i++)
	{
		std::vector<int> CG_copy(CG.size());
		CG_copy.assign(CG.cbegin(), CG.cend());
		int num_of_CGedges_copy = num_of_CGedges;
		for (int j = 0; j < cols; j++)
		{
			if (CG_copy[node[i] * cols + j] > 0)
			{
				CG_copy[node[i] * cols + j] = 0;
				CG_copy[j * cols + node[i]] = 0;
				num_of_CGedges_copy--;
			}
		}
		if (KVertexCover(CG_copy, num_of_CGnodes - 1, num_of_CGedges_copy, k - 1, cols))
			return true;
	}
	return false;
}

int CBSHeuristic::greedyMatching(const std::vector<int>& CG,  int cols)
{
	int rst = 0;
	std::vector<bool> used(cols, false);
	while(true)
	{
		int maxWeight = 0;
		int ep1, ep2;
		for (int i = 0; i < cols; i++)
		{
			if(used[i])
				continue;
			for (int j = i + 1; j < cols; j++)
			{
				if (used[j])
					continue;
				else if (maxWeight < CG[i * cols + j])
				{
					maxWeight = CG[i * cols + j];
					ep1 = i;
					ep2 = j;
				}
			}
		}
		if (maxWeight == 0)
			return rst;
		rst += maxWeight;
		used[ep1] = true;
		used[ep2] = true;
	}
}

int CBSHeuristic::minimumWeightedVertexCover(const vector<int>& HG)
{
	clock_t t = clock();
	int rst = weightedVertexCover(HG);
	runtime_solve_MVC += (double)(clock() - t) / CLOCKS_PER_SEC;
	return rst;
}


int CBSHeuristic::weightedVertexCover(const std::vector<int>& CG)
{
	int rst = 0;
	std::vector<bool> done(num_of_agents, false);
	for (int i = 0; i < num_of_agents; i++)
	{
		if (done[i])
			continue;
		std::vector<int> range;
		std::vector<int> indices;
		range.reserve(num_of_agents);
		indices.reserve(num_of_agents);
		int num = 0;
		int num_states = 1;
		std::queue<int> Q;
		Q.push(i);
		done[i] = true;
		while (!Q.empty())
		{
			int j = Q.front(); Q.pop();
			range.push_back(0); // TODO::this line can be deleted?
			indices.push_back(j);
			for (int k = 0; k < num_of_agents; k++)
			{
				if (CG[j * num_of_agents + k] > 0)
				{
					range[num] = std::max(range[num], CG[j * num_of_agents + k]);
					if (!done[k])
					{
						Q.push(k);
						done[k] = true;
					}
				}		
				else if (CG[k * num_of_agents + j] > 0)
				{
					range[num] = std::max(range[num], CG[k * num_of_agents + j]);
					if (!done[k])
					{
						Q.push(k);
						done[k] = true;
					}
				}
			}
			num++;
		}
		if (num  == 1) // no edges
			continue;
		else if (num == 2) // only one edge
		{
			rst += std::max(CG[indices[0] * num_of_agents + indices[1]], CG[indices[1] * num_of_agents + indices[0]]); // add edge weight
			continue;
		}
		std::vector<int> G(num * num, 0);
		for (int j = 0; j < num; j++)
		{
			for (int k = j + 1; k < num; k++)
			{
				G[j * num + k] = std::max(CG[indices[j] * num_of_agents + indices[k]], CG[indices[k] * num_of_agents + indices[j]]);
			}
		}
		//if (num > DP_threshold) // solve by ILP
		//	rst += ILPForWMVC(G, range);
        if (num > DP_node_threshold) // solve by greedy matching
        {
            rst += greedyMatching(G, (int)range.size());
        }
		else // solve by dynamic programming
		{
			std::vector<int> x(num); 
			int best_so_far = MAX_COST;
			rst += DPForWMVC(x, 0, 0, G, range, best_so_far);
		}
		double runtime = (double)(clock() - start_time) / CLOCKS_PER_SEC;
		if (runtime > time_limit)
			return -1; // run out of time
	}

	//test
	/*std::vector<int> x(N, 0);
	std::vector<int> range(N, 0);
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			range[i] = std::max(range[i], CG[i * N + j]);
			range[j] = std::max(range[j], CG[i * N + j]);
		}
	}
	int best_so_far = INT_MAX;
	int rst2 = DPForWMVC(x, 0, 0, CG, range, best_so_far);
	if( rst != rst2)
		std::cout << "ERROR" <<std::endl;*/

	return rst;
}

// recusive component of dynamic programming for weighted vertex cover
int CBSHeuristic::DPForWMVC(std::vector<int>& x, int i, int sum, const std::vector<int>& CG, const std::vector<int>& range, int& best_so_far)
{
	if (sum >= best_so_far)
		return MAX_COST;
	double runtime = (double)(clock() - start_time) / CLOCKS_PER_SEC;
	if (runtime > time_limit)
		return -1; // run out of time
	else if (i == (int)x.size())
	{
		best_so_far = sum;
		return sum;
	}
	else if (range[i] == 0) // vertex i does not have any edges.
	{
		int rst = DPForWMVC(x, i + 1, sum, CG, range, best_so_far);
		if (rst < best_so_far)
		{
			best_so_far = rst;
		}
		return best_so_far;
	}

	int cols = x.size();
	// find minimum cost for this vertex
	int min_cost = 0;
	for (int j = 0; j < i; j++)
	{
		if (min_cost + x[j] < CG[j * cols + i]) // infeasible assignment
		{
			min_cost = CG[j * cols + i] - x[j]; // cost should be at least CG[i][j] - x[j];
		}
	}


	int best_cost = -1;
	for (int cost = min_cost; cost <= range[i]; cost++)
	{
		x[i] = cost;
		int rst = DPForWMVC(x, i + 1, sum + x[i], CG, range, best_so_far);
		if (rst < best_so_far)
		{
			best_so_far = rst;
			best_cost = cost;
		}
	}
	if (best_cost >= 0)
	{
		x[i] = best_cost;
	}

	return best_so_far;
}


void CBSHeuristic::copyConflictGraph(CBSNode& child, const CBSNode& parent) const
{
	//copy conflict graph
	if (type == heuristics_type::DG || type == heuristics_type::WDG)
	{
		unordered_set<int> changed;
		for (const auto& p : child.paths) {
			changed.insert(p.first);
		}
		for (auto e : parent.conflictGraph) {
			if (changed.find(e.first / num_of_agents) == changed.end() &&
				changed.find(e.first % num_of_agents) == changed.end())
				child.conflictGraph[e.first] = e.second;
		}

	}
}

bool CBSHeuristic::dependent(int a1, int a2, CBSNode& node) // return true if the two agents are dependent
{
	const MDD* mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size()); // get mdds
	const MDD* mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());
	if (mdd1->levels.size() > mdd2->levels.size()) // swap
		std::swap(mdd1, mdd2);
	num_merge_MDDs++;
	return !SyncMDDs(*mdd1, *mdd2);
}

// return true if the joint MDD exists.
bool CBSHeuristic::SyncMDDs(const MDD &mdd, const MDD& other) // assume mdd.levels <= other.levels
{
	if (other.levels.size() <= 1) // Either of the MDDs was already completely pruned already
		return false;

	SyncMDD copy(mdd);
	if (copy.levels.size() < other.levels.size())
	{
		size_t i = copy.levels.size();
		copy.levels.resize(other.levels.size());
		for (; i < copy.levels.size(); i++)
		{
			SyncMDDNode* parent = copy.levels[i - 1].front();
			auto node = new SyncMDDNode(parent->location, parent);
			parent->children.push_back(node);
			copy.levels[i].push_back(node);

		}
	}
	// Cheaply find the coexisting nodes on level zero - all nodes coexist because agent starting points never collide
	copy.levels[0].front()->coexistingNodesFromOtherMdds.push_back(other.levels[0].front());

	// what if level.size() = 1?
	for (size_t i = 1; i < copy.levels.size(); i++)
	{
		for (auto node = copy.levels[i].begin(); node != copy.levels[i].end();)
		{
			// Go over all the node's parents and test their coexisting nodes' children for co-existance with this node
			for (auto parent = (*node)->parents.begin(); parent != (*node)->parents.end(); parent++)
			{
				//bool validParent = false;
				for (const MDDNode* parentCoexistingNode : (*parent)->coexistingNodesFromOtherMdds)
				{
					for (const MDDNode* childOfParentCoexistingNode : parentCoexistingNode->children)
					{
						if ((*node)->location == childOfParentCoexistingNode->location || // vertex conflict
							((*node)->location == parentCoexistingNode->location &&
							(*parent)->location == childOfParentCoexistingNode->location)) // edge conflict
							continue;
						//validParent = true;

						auto it = (*node)->coexistingNodesFromOtherMdds.cbegin();
						for (; it != (*node)->coexistingNodesFromOtherMdds.cend(); ++it)
						{
							if (*it == childOfParentCoexistingNode)
								break;
						}
						if (it == (*node)->coexistingNodesFromOtherMdds.cend())
						{
							(*node)->coexistingNodesFromOtherMdds.push_back(childOfParentCoexistingNode);
						}
					}
				}
				//if (!validParent)
				//{
				//	// delete the edge, and continue up the levels if necessary
				//	SyncMDDNode* p = *parent;
				//	parent = (*node)->parents.erase(parent);
				//	p->children.remove((*node));
				//	if (p->children.empty())
				//		copy.deleteNode(p);
				//}
				//else
				//{
				//	parent++;
				//}
			}
			if ((*node)->coexistingNodesFromOtherMdds.empty())
			{
				// delete the node, and continue up the levels if necessary
				SyncMDDNode* p = *node;
				node++;
				copy.deleteNode(p, i);
			}
			else
				node++;
		}
		if (copy.levels[i].empty())
		{
			copy.clear();
			return false;
		}
	}
	copy.clear();
	return true;
}




// The implementation of our CHBP starts from here:


void CBSHeuristic::copyCluster(CBSNode &child, const CBSNode &parent) const {
    if(ch_type != cluster_heuristics_type::N) {
        // inherit all clusters based on the cost of current paths.
        for (const auto &cluster: parent.cluster_found) {
            bool valid = true;
            for(int i = 0; i < cluster->agent_id.size(); i ++){
                // filter out the cluster if the path cost changed.
                if(paths[cluster->agent_id[i]]->size() != cluster->path_cost[i]){
                    valid = false;
                    break;
                }
            }
            if(valid) child.cluster_found.emplace_back(cluster);
        }
    }
}
void CBSHeuristic::copyMutexGraph(CBSNode &child, const CBSNode &parent) const {
    // copy mutex graph, this graph is maintained on every node, to avoid lookups of centralized database.
    if(applyMutexCache){
        unordered_set<int> changed;
        for (const auto& p : child.paths) {
            changed.insert(p.first);
        }
        // only copy the edges if the path of agent has not been modified.
        for (const auto& e : parent.mutexGraph) {
            if (changed.find(e.first / num_of_agents) == changed.end() &&
                changed.find(e.first % num_of_agents) == changed.end())
                child.mutexGraph[e.first] = e.second;
        }
    }
}



void CBSHeuristic::filterCluster(CBSNode &node, int& cluster_heuristic, vector<bool>& excluded_agents) {
    // We already filter out the clusters which contain agents with different path cost.
    // We only check the whether agent is excluded here.
    for (auto it = node.cluster_found.begin(); it != node.cluster_found.end(); )
    {
        bool found = true;
        for (int i = 0; i < (*it)->agent_id.size(); i++) {
            if (excluded_agents[(*it)->agent_id[i]]) {
                // do not inherit cluster, if agent is already excluded.
                found = false;
                break;
            }
        }
        if(found){
            cluster_heuristic += (*it)->increased_cost;
            for(auto i : (*it)->agent_id){
                // mark as excluded for all agents in cluster.
                excluded_agents[i] = true;
            }
            ++it;
        }else{
            // remove the invalid cluster.
            node.cluster_found.erase(it);
        }
    }

}


int CBSHeuristic::getMaxConflictAgent(vector<bool>& processed_agents, vector<bool>& excluded_agents, vector<bool>& SG){
    int max_agent = -1;
    size_t max_size = 0;
    for(int cn = 0; cn < num_of_agents; cn ++){
        // only consider the not processed or not excluded agents.
        if(processed_agents[cn] || excluded_agents[cn]) continue;
        int size = 0;
        int row_index = cn*num_of_agents;
        for(int check_agent = 0; check_agent < num_of_agents; check_agent ++){
            // counting the conflicting agents that have not been excluded.
            if(SG[row_index+check_agent] && !excluded_agents[check_agent]){
                size++;
            }
        }
        if(size > max_size){
            // record the agent with the maximum number of conflicts.
            max_size = size;
            max_agent = cn;
        }
    }
    if(max_agent != -1){
        //-------------------------------Algorithm 1 line 14 -------------------------------
        processed_agents[max_agent] = true;
        //-------------------------------Algorithm 1 line 14 -------------------------------
    }
    return max_agent;
}

void CBSHeuristic::removeEdgesFromSG( int agent_id, vector<bool>& SG){
    // remove all edges of agent_id from SG.
    for(int i = num_of_agents*agent_id; i < num_of_agents*(agent_id +1); i ++ ){
        if(SG[i]){
            SG[(i - num_of_agents*agent_id) * num_of_agents + agent_id ] = false;
            SG[i] = false;
        }
    }
}


int CBSHeuristic::solveClusterAgents(const unordered_set<int>& cluster_set,  const CBSNode& node, bool cardinal){
    // Exactly same with solving 2 agents in WDG.
    num_cluster_solved ++;
    size_t number_of_agents = cluster_set.size();
    vector<SingleAgentSolver*> engines(number_of_agents);
    int index = 0;
    for(auto& agent_id: cluster_set){
        engines[index] = search_engines[agent_id];
        index ++;
    }
    vector<vector<PathEntry>> initial_paths(number_of_agents);
    index = 0;
    for(auto& agent_id: cluster_set){
        initial_paths[index] = *paths[agent_id];
        index ++;
    }
    vector<ConstraintTable> constraints;
    for(auto& agent_id: cluster_set){
        constraints.emplace_back(initial_constraints[agent_id]);
    }
    index = 0;
    for(auto& agent_id: cluster_set){
        constraints[index].build(node, agent_id);
        index ++;
    }


    CBS cbs(engines, constraints, initial_paths, screen);
    cbs.setPrioritizeConflicts(PC);
    cbs.setHeuristicType(heuristics_type::CG);
    cbs.setDisjointSplitting(disjoint_splitting);
    cbs.setBypass(false);
    cbs.setRectangleReasoning(rectangle_reasoning);
    cbs.setCorridorReasoning(corridor_reasoning);
    cbs.setTargetReasoning(target_reasoning);
    cbs.setMutexReasoning(mutex_reasoning);
    cbs.setNodeLimit(node_limit);
    cbs.setClusterHeuristicType(cluster_heuristics_type::N);
    double runtime = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    int root_g = 0;
    for(const auto& p : initial_paths){
        root_g += (int)p.size()-1;
    }
    // must at lease increase 1;
    int lowerbound = root_g;
    int upperbound = MAX_COST;
    if (cardinal)
        lowerbound += 1;
    cbs.solve(time_limit - runtime, lowerbound, upperbound);
    int rst;
    if (cbs.runtime > time_limit - runtime || cbs.num_HL_expanded > node_limit) // time out or node out
        rst = (int)cbs.min_f_val - root_g; // using lowerbound to approximate
    else if (cbs.solution_cost  < 0) // no solution
        rst = MAX_COST;
    else
    {
        rst = cbs.solution_cost - root_g;
    }
    assert(rst >= 0);
    return rst;
}

void CBSHeuristic::computeClusterHeuristic(CBSNode& node, const unordered_set<int>& cluster_set ){
    // create the cluster.
    shared_ptr<Cluster> cluster(new Cluster());
    cluster->agent_id.resize(cluster_set.size());
    cluster->path_cost.resize(cluster_set.size());
    // insert path cost and agent id
    int index = 0;
    for(auto& agent_id : cluster_set){
        cluster->agent_id[index] = agent_id;
        cluster->path_cost[index] = paths[agent_id]->size();
        index ++;
    }

    if(applyHeuristicCache){
        auto MCH = MultiConstraintsHasher(cluster->agent_id, &node);
        // hash the key base on the constraints of all agents in a cluster.
        auto got = CHLookupTable[MCH.agents[0]][MCH.agents[1]].find(MCH);
        if (got != CHLookupTable[MCH.agents[0]][MCH.agents[1]].end()) // check the lookup table first
        {
            // lookup the centralized database.
            num_cluster_memorized++;
            cluster->increased_cost  = got->second;
        } else {
            // solve the cluster and cache the results.
            cluster->increased_cost = solveClusterAgents(cluster_set,node,true);
            CHLookupTable[MCH.agents[0]][MCH.agents[1]][MCH] = cluster->increased_cost;
        }
    }else{
        // don't apply caching, directly solve the cluster.
        cluster->increased_cost = solveClusterAgents(cluster_set,node,true);
    }

    node.cluster_found.push_back(cluster);
}
bool CBSHeuristic::findMutexBetweenMDDs(MDD *main_mdd, MDD *secondary_mdd, vector<tuple<int, int, int>> &main_mutex,
                                        vector<tuple<int, int, int>> &secondary_mutex) {
    // assume main_mdd.size() >  secondary_mdd.size();
    num_mutex_checking++;
    assert(main_mutex.empty() && secondary_mutex.empty());
    // perform mutex propagation
    cp_helper.set_MDD(main_mdd, secondary_mdd);
    cp_helper.init_mutex();
    cp_helper.fwd_mutex_prop();
    int mdd_size = secondary_mdd->levels.size();
    if(!cp_helper.fwd_mutexes.empty() ) {
        // get mutex nodes for equal-sized part of MDDs.
        cp_helper.get_mutex_node(main_mutex,secondary_mutex);
    }
    // get mutex nodes for target part of MDDs.
    int goal_location = secondary_mdd->levels[mdd_size - 1].front()->location;
    for (int i = mdd_size - 1; i < main_mdd->levels.size(); i++) {
        for (auto &curr_ptr_i: main_mdd->levels[i]) {
            if (curr_ptr_i->location == goal_location) {
                main_mutex.emplace_back(make_tuple(i, curr_ptr_i->location, -1));
            }
        }
    }
    return true;
}


bool CBSHeuristic::mutexCheckingWithCache(CBSNode& node, int main_agent, int check_agent, const Path& main_path,bool& found_incompatible_nodes, bool& path_not_existed ){
    shared_ptr<mutex_nodes> mdd_mutex;
    int idx1 = main_agent * num_of_agents + check_agent;
    // find incompatible MDD nodes from mutex graph.
    auto got_mutex = node.mutexGraph.find(idx1);
    if (got_mutex != node.mutexGraph.end()){
        // get incompatible MDD nodes if exists.
        mdd_mutex = got_mutex->second;
    }else {
        // otherwise, check the centralized database.
        int a1 = min(main_agent, check_agent);
        int a2 = max(main_agent, check_agent);
        int idx3 = a1 * num_of_agents + a2;
        int idx4 = a2 * num_of_agents + a1;
        auto DCH = DoubleConstraintsHasher(a1, a2, &node);
        auto got = mutexLookupTable[a1][a2].find(DCH);
        if (got != mutexLookupTable[a1][a2].end()) // check the lookup table first
        {
            // get incompatible MDD nodes from centralized database if exists.
            num_mutex_memorized ++;
            node.mutexGraph[idx3] = got->second.first;
            node.mutexGraph[idx4] = got->second.second;
            mdd_mutex = main_agent < check_agent ? got->second.first : got->second.second;
        } else {
            // otherwise, perform mutex propagation
            auto mdd1 = mdd_helper.getMDD(node, a1, paths[a1]->size());
            auto mdd2 = mdd_helper.getMDD(node, a2, paths[a2]->size());
            shared_ptr<mutex_nodes> mdd1_mutex(new mutex_nodes());
            shared_ptr<mutex_nodes> mdd2_mutex(new mutex_nodes());
            if (mdd1->levels.size() < mdd2->levels.size()) {
                //perform mutex propagation.
                findMutexBetweenMDDs(mdd2, mdd1, *mdd2_mutex, *mdd1_mutex);
            } else {
                //perform mutex propagation.
                findMutexBetweenMDDs(mdd1, mdd2, *mdd1_mutex, *mdd2_mutex);
            }
            node.mutexGraph[idx3] = mdd1_mutex;
            node.mutexGraph[idx4] = mdd2_mutex;
            mutexLookupTable[a1][a2][DCH] = make_pair(mdd1_mutex, mdd2_mutex);
            mdd_mutex = main_agent < check_agent ? mdd1_mutex : mdd2_mutex;
        }
    }

    // deleting incompatible node and edges.
    for( const auto& mutex : *mdd_mutex){
        if( get<2>(mutex) == -1 ) {
            // delete incompatible MDD nodes.
            for (auto &main_node_ptr: mdd_soft_copy.levels[get<0>(mutex)]) {
                if (main_node_ptr->location == get<1>(mutex)) {
                    found_incompatible_nodes = true;
                    mdd_soft_copy.deleteNode(main_node_ptr,main_path,path_not_existed);
                    break;
                }
            }
        }else{
            // delete incompatible MDD edges, these edges can only be singleton edges.
            for (auto &main_node_ptr: mdd_soft_copy.levels[get<0>(mutex)]) {
                if (main_node_ptr->location == get<1>(mutex)) {
                    for(auto & child_ptr : main_node_ptr->children){
                        if(child_ptr->location == get<2>(mutex)){
                            mdd_soft_copy.deleteEdge(main_node_ptr, child_ptr, main_path, path_not_existed);
                            found_incompatible_nodes = true;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        if(mdd_soft_copy.levels[get<0>(mutex)].empty()) {
            //return true if MDD becomes empty
            return true;
        }
    }
    return false;
}

bool CBSHeuristic::mutexChecking(CBSNode& node, int check_agent, const Path& main_path,bool& found_incompatible_nodes, bool& path_not_existed ){
    // only perform mutex propagation for mdd_m -> mdd_c.
    num_mutex_checking ++;
    auto check_mdd = mdd_helper.getMDD(node,  check_agent, paths[ check_agent]->size());
    int mdd_size = min(mdd_soft_copy.levels.size(), check_mdd->levels.size());

    // for the case that mdd_m > mdd_c, check target part.
    if(mdd_soft_copy.levels.size() > check_mdd->levels.size()){
        int goal_location = check_mdd->levels[mdd_size-1].front()->location;
        for(int i = mdd_size-1; i < mdd_soft_copy.levels.size(); i ++){
            //check target part and remove incompatible nodes.
            for (auto& curr_ptr_i: mdd_soft_copy.levels[i]) {
                if(curr_ptr_i->location == goal_location){
                    found_incompatible_nodes = true;
                    mdd_soft_copy.deleteNode(curr_ptr_i,main_path,path_not_existed);
                    break;
                }
            }
            if(mdd_soft_copy.levels[i].empty()) {
                //return true if MDD becomes empty
                return true ;
            }
        }
    }
    // perform mutex propagation.
    smdd_cp_helper.set_MDD(&mdd_soft_copy, check_mdd);
    smdd_cp_helper.init_mutex();
    smdd_cp_helper.fwd_mutex_prop();
    if(!smdd_cp_helper.fwd_mutexes.empty()){
        for(int i = 1; i < mdd_size; i ++){
            // detect incompatible MDD nodes.
            vector<SoftMDDNode*> to_be_deleted;
            for (auto& curr_ptr_i: mdd_soft_copy.levels[i]) {
                bool is_mutex_node = true;
                for (auto& it: check_mdd->levels[i]) {
                    if (! smdd_cp_helper.has_fwd_mutex(curr_ptr_i, it)){
                        is_mutex_node = false;
                        break;
                    }
                }
                if(is_mutex_node){
                    found_incompatible_nodes = true;
                    to_be_deleted.emplace_back(curr_ptr_i);
                }
            }
            // remove incompatible MDD nodes.
            for(auto curr_ptr_i : to_be_deleted){
                mdd_soft_copy.deleteNode(curr_ptr_i,main_path,path_not_existed);
            }
            //detect singleton incompatible edges.
            if( check_mdd->levels[i].size() == 1 &&  check_mdd->levels[i].front()->parents.size() == 1){
                for (auto& curr_ptr_i: mdd_soft_copy.levels[i]) {
                    if(curr_ptr_i ->location == check_mdd->levels[i].front()->parents.front()->location){
                        for(auto& parent_ptr_i: curr_ptr_i->parents){
                            if(parent_ptr_i->location == check_mdd->levels[i].front()->location){
                                // remove incompatible MDD edge.
                                mdd_soft_copy.deleteEdge(parent_ptr_i,curr_ptr_i, main_path, path_not_existed);
                                found_incompatible_nodes = true;
                                break;
                            }
                        }
                    }
                }
            }
            if(mdd_soft_copy.levels[i].empty()) {
                //return true if MDD becomes empty
                return true;
            }
        }
    }
    return false;
}

void CBSHeuristic::getPathAndConflictAgentFromMDD(SoftMDD& main_mdd, Path& main_path, unordered_set<int>& CA){
    // extracting a path from the current mdd.
    size_t mdd_size =  main_mdd.levels.size();
    main_mdd.levels[0].front()->min_total_conflict = main_mdd.levels[0].front()->num_of_conflict;
    // perform BFS from start to goal.
    for( int i = 0;  i < mdd_size; i ++){
        for(auto mdd_node : main_mdd.levels[i]){
            mdd_node->pre_node = nullptr;
            for( auto par_mdd_node : mdd_node->parents){
                if(mdd_node->pre_node == nullptr){
                    // set predecessor, when first visit.
                    mdd_node->min_total_conflict = par_mdd_node->min_total_conflict + mdd_node->num_of_conflict;
                    size_t idx = (1 + par_mdd_node->location ) * map_size + mdd_node->location;
                    if(main_mdd.edge_conflict_agent[i].find(idx) != main_mdd.edge_conflict_agent[i].end()){
                        mdd_node->min_total_conflict += main_mdd.edge_conflict_agent[i][idx].size();
                    }
                    mdd_node->pre_node = par_mdd_node;
                }else{
                    // update the predecessor, iff the number of conflicts reduced.
                    if(mdd_node->min_total_conflict > par_mdd_node->min_total_conflict + mdd_node->num_of_conflict){
                        mdd_node->min_total_conflict = par_mdd_node->min_total_conflict + mdd_node->num_of_conflict;
                        size_t idx = (1 + par_mdd_node->location ) * map_size + mdd_node->location;
                        if(main_mdd.edge_conflict_agent[i].find(idx) != main_mdd.edge_conflict_agent[i].end()){
                            mdd_node->min_total_conflict += main_mdd.edge_conflict_agent[i][idx].size();
                        }
                        mdd_node->pre_node = par_mdd_node;
                    }
                }
            }
        }
    }


    // backward extracting the path and conflicting agents CA;
    int cur_index = main_mdd.levels.size()-1;
    SoftMDDNode* cur_node_ptr = main_mdd.levels[main_mdd.levels.size()-1].front();
    SoftMDDNode* pre_node_ptr = cur_node_ptr;
    while(cur_index >= 0 ){
        main_path[cur_index].location = cur_node_ptr->location;
        for(auto agent : cur_node_ptr->node_conflict_agent){
            // add vertex conflict agents.
            CA.insert(agent);
        }
        if(cur_node_ptr != pre_node_ptr){
            size_t idx = (1 + cur_node_ptr->location ) * map_size + pre_node_ptr->location;
            if(main_mdd.edge_conflict_agent[cur_index+1].find(idx) != main_mdd.edge_conflict_agent[cur_index+1].end()){
                for(auto agent : main_mdd.edge_conflict_agent[cur_index+1][idx]){
                    // add edge conflict agents.
                    CA.insert(agent);
                }
            }
        }
        pre_node_ptr = cur_node_ptr;
        cur_node_ptr = cur_node_ptr->pre_node;
        cur_index --;
    }
}


void CBSHeuristic::labelConflict2MDD(SoftMDD& main_mdd, int curr_agent){
    // for every path in the current plan, label the conflicts on the mdd.
    main_mdd.edge_conflict_agent.clear();
    main_mdd.edge_conflict_agent.resize(main_mdd.levels.size());
    size_t mdd_size =  main_mdd.levels.size();
    for (int ag = 0; ag < paths.size(); ag++)
    {
        if (ag == curr_agent || paths[ag] == nullptr)
            continue;

        size_t min_path_length = paths[ag]->size() < mdd_size ? paths[ag]->size() : mdd_size;
        // labelling the equal-sized part of MDD.
        for (size_t timestep = 0; timestep < min_path_length; timestep++)
        {
            int path_loc = paths[ag]->at(timestep).location;
            for( auto mdd_node : main_mdd.levels[timestep] ){
                if( mdd_node->location == path_loc){
                    // labelling vertex conflicts.
                    mdd_node->node_conflict_agent.push_back(ag);
                    mdd_node->num_of_conflict ++;
                }else if(timestep < min_path_length - 1 && mdd_node->location == paths[ag]->at(timestep + 1).location){
                    // labelling edge conflicts.
                    for(auto children_mdd : mdd_node->children){
                        if( children_mdd->location == path_loc){
                            size_t idx = (1 + mdd_node->location ) * map_size + children_mdd->location;
                            if( main_mdd.edge_conflict_agent[timestep + 1].find(idx) != main_mdd.edge_conflict_agent[timestep + 1].end()){
                                main_mdd.edge_conflict_agent[timestep + 1][idx].push_back(ag);
                            }else{
                                main_mdd.edge_conflict_agent[timestep + 1].insert(make_pair(idx,vector<int>{ag}));
                            }
                        }
                    }
                }
            }
        }
        // labelling target parts for p < mdd_c.
        if( paths[ag]->size() < mdd_size){
            int path_loc = paths[ag]->back().location;
            for (size_t timestep = min_path_length; timestep < mdd_size; timestep++) {
                for( auto mdd_node : main_mdd.levels[timestep] ){
                    if( mdd_node->location == path_loc){
                        mdd_node->node_conflict_agent.push_back(ag);
                        mdd_node->num_of_conflict ++;
                    }
                }
            }
        }
        // labelling target parts for mdd_c > p.
        if( paths[ag]->size() > mdd_size){
            // for mdd_c > mdd_m, label all conflicts on the target of mdd_m.
            int mdd_loc = main_mdd.levels[mdd_size-1].front()->location;
            for (size_t timestep = min_path_length; timestep <paths[ag]->size(); timestep++) {
                if(mdd_loc == paths[ag]->at(timestep).location){
                    main_mdd.levels[mdd_size-1].front()->node_conflict_agent.push_back(ag);
                    main_mdd.levels[mdd_size-1].front()->num_of_conflict ++;
                }
            }
        }
    }
    // set labelled,  never label it twice in one CBS node.
    main_mdd.is_labeled  = true;
}

result_type CBSHeuristic::findClusterOrBypass(CBSNode& curr, int a_m, Path& bypass_path, vector<bool>& excluded_agents,vector<bool>& SG){
    //--------------------BEGIN: Algorithm 2 line 1 (copy path and conflict agent of a_m)-----------------------------------------------
    // set current path p_m to the path of a_m.
    bypass_path = *paths[a_m];
    // processed agents: PA.
    vector<bool> processed_agents(num_of_agents,false);
    // record whether path has been modified.
    bool path_modified = false;
    // cluster_set records the cluster agents.
    unordered_set<int> cluster_set;
    // set of conflicting agents with current path of a_m: CA.
    unordered_set<int> CA;

    //get non-excluded CA, get the original number of conflicts in current CBS node.
    size_t original_num_of_conflict= 0;
    for(int i = a_m*num_of_agents; i < (a_m+1)*num_of_agents; i++){
        if(SG[i]){
            if(!excluded_agents[i%num_of_agents]) CA.insert(i%num_of_agents);
            original_num_of_conflict++;
        }
    }
    size_t curr_num_of_conflict = original_num_of_conflict;
    //----------------------END: Algorithm 2 line 1 (copy path and conflict agent of a_m)-----------------------------------------------

    //------------------------BEGIN: Algorithm 2 line 2 (get the MDD of a_m)------------------------------------------------------------
    // copy the MDD of a_m
    mdd_soft_copy.is_labeled = false;
    mdd_soft_copy.levels.clear();
    mdd_soft_copy.fastCopyMDD(*mdd_helper.getMDD(curr, a_m, paths[a_m]->size()));
    //----------------------- END: Algorithm 2 line 2 (get the MDD of a_m)--------------------------------------------------------------

    // main loop
    for(;;){
        bool path_not_existed   = false;
        for(int conflict_agent : CA){
            //------------------------BEGIN: Algorithm 2 line 5 (mark agent as processed)-----------------------------------------------
            processed_agents[conflict_agent] = true;
            //------------------------END: Algorithm 2 line 5 (mark agent as processed)-------------------------------------------------

            //---------BEGIN: Algorithm 2 line 6 - 10 (perform mutex propagation and remove incompatible nodes)-------------------------
            bool found_incompatible_nodes = false;
            bool mdd_empty = false;
            if(applyMutexCache){
                // perform mutex propagation with memorization technique: a_m -> conflict_agent.
                mdd_empty = mutexCheckingWithCache(curr,a_m,conflict_agent,bypass_path,found_incompatible_nodes,path_not_existed);
            }else{
                // perform incremental mutex propagation:  a_m -> conflict_agent.
                mdd_empty = mutexChecking(curr,conflict_agent,bypass_path,found_incompatible_nodes,path_not_existed);
            }
            //---------END: Algorithm 2 line 6 - 10 (perform mutex propagation and remove incompatible nodes)---------------------------

            //--------------------------BEGIN: Algorithm 2 line 11-12 (return cluster)--------------------------------------------------
            // mdd of a_m is empty.
            if(mdd_empty) {
                if(!applyMutexHeuristic) return result_type::FNCNP;
                cluster_set.insert(a_m);
                cluster_set.insert(conflict_agent);
                //--------BEGIN: Algorithm 1 line 9 (mark cluster as excluded)----------------------------------------------------------
                for(auto cluster_agent : cluster_set){
                    // mark all agents in cluster as excluded.
                    excluded_agents[cluster_agent] = true;
                }
                //--------END: Algorithm 1 line 9 (mark cluster as excluded)------------------------------------------------------------
                if(applySolvingCluster){
                    //---------BEGIN: Algorithm 1 line 10 (compute lower-bound of cluster)----------------------------------------------
                    // solve the cluster to find the lower-bound (i.e., the increased cost).
                    computeClusterHeuristic(curr,cluster_set);
                    //---------END: Algorithm 1 line 10 (compute lower-bound of cluster)------------------------------------------------
                    return result_type::FC;
                }else{
                    shared_ptr<Cluster> cluster(new Cluster());
                    cluster->agent_id.resize(cluster_set.size());
                    cluster->path_cost.resize(cluster_set.size());
                    int index = 0;
                    for(auto& agent_id : cluster_set){
                        cluster->agent_id[index] = agent_id;
                        cluster->path_cost[index] = paths[agent_id]->size();
                        index ++;
                    }
                    //---------BEGIN: Algorithm 1 line 10 (compute lower-bound of cluster)----------------------------------------------
                    // simply set the increased cost to 1.
                    cluster->increased_cost = 1;
                    //---------END: Algorithm 1 line 10 (compute lower-bound of cluster)------------------------------------------------

                    // push to the back of current node and return found cluster.
                    curr.cluster_found.push_back(cluster);
                    return result_type::FC;
                }
            }
            //-----------------------END: Algorithm 2 line 11-12 (return cluster)-------------------------------------------------------

            // insert conflicting agent to cluster, iff found incompatible nodes.
            if(found_incompatible_nodes) cluster_set.insert(conflict_agent);
            // break the current loop if the current path is not existed in the modified MDD.
            if(path_not_existed) break;
        }

        if(path_not_existed){
            //--------BEGIN: Algorithm 2 line 13-15 (get min conflict path from MDD and update CA)--------------------------------------
            // change the current path to an alternative path.
            path_modified = true;
            if(!mdd_soft_copy.is_labeled){
                // label all conflicts on the MDD.
                labelConflict2MDD(mdd_soft_copy, a_m);
            }

            CA.clear();
            // get an alternative path from MDD which minimize the number of conflicts, and its conflicting agents: CA.
            getPathAndConflictAgentFromMDD(mdd_soft_copy,bypass_path,CA);
            curr_num_of_conflict = CA.size();
            // remove the processed agent and excluded agent in CA.
            for(auto it = begin(CA); it != end(CA);)
            {
                if (processed_agents[*it] || excluded_agents[*it])
                {
                    it = CA.erase(it);
                }
                else
                    ++it;
            }
            //--------END: Algorithm 2 line 13-15 (get min conflict path from MDD and update CA)----------------------------------------
        }else{
            //----------------------BEGIN: Algorithm 2 line 16 (return Bypass if CA reduced)--------------------------------------------
            if(path_modified && curr_num_of_conflict < original_num_of_conflict){
                // found bypass, if the number of conflicts of a_m reduced.
                if(applyBypass){
                    curr.tie_breaking = curr.tie_breaking + curr_num_of_conflict - original_num_of_conflict;
                }
                return result_type::FBP;
            }else{
                // return found nothing.
                return result_type::FNCNP;
            }
            //----------------------END: Algorithm 2 line 16 (return Bypass if CA reduced)----------------------------------------------
        }
    }

}

int CBSHeuristic::computeClusterHeuristicAndBypass(CBSNode& curr, vector<bool>& excluded_agents, double time_limit){
    //--------------BEGIN: Algorithm 1 line 2 (inherit cluster from parent)-------------------------------------------------------------
    int cluster_heuristic = 0;
    // all clusters from the parent node are already copied into the current node, we only need to filter them here.
    filterCluster(curr,cluster_heuristic,excluded_agents);
    //--------------END: Algorithm 1 line 2 (inherit cluster from parent)---------------------------------------------------------------

    //------------------BEGIN: Algorithm 1 line 4 (build conflict state graph)----------------------------------------------------------
    // create conflict state graph, SG, of the current node.
    vector<bool> SG(num_of_agents* num_of_agents, false);
    for(const auto& c: curr.conflicts){
        SG[c->a1 * num_of_agents + c->a2] = true;
        SG[c->a2 * num_of_agents + c->a1] = true;
    }
    //------------------END: Algorithm 1 line 4 (build conflict state graph)------------------------------------------------------------
    // processed agent: PA
    vector<bool> processed_agent(num_of_agents,false);
    // agent which has maximal number of conflicts;
    int a_m = -1;
    // bypass results of R;
    Path bypass_path;
    for(;;){
        // run out of time
        if ((clock() - start_time) / CLOCKS_PER_SEC > time_limit)
        {
            return -1;
        }

        //-----------------------BEGIN: Algorithm 1 line 5 (select a_m from SG)-------------------------------------------------------
        // get a_m from SG that is not been processed or excluded.
        // Algorithm 1 line 11:  a_m is marked as PA inside this function.
        a_m = getMaxConflictAgent(processed_agent,excluded_agents,SG);
        if(a_m == -1) break;
        //-----------------------END: Algorithm 1 line 5 (select a_m from SG)---------------------------------------------------------

        //-----------------------BEGIN: Algorithm 1 line 6 (detect cluster or bypass)---------------------------------------------------
        // detecting cluster and bypass of a_m, this function refers to Algorithm 2 of our paper.
        auto result = findClusterOrBypass(curr,a_m,bypass_path,excluded_agents,SG);
        //-----------------------ENd: Algorithm 1 line 6 (detect cluster or bypass)-----------------------------------------------------

        if(result == result_type::FC){
            //-----------------------BEGIN: Algorithm 1 line 7-10 (found cluster)-------------------------------------------------------
            // Algorithm 1 line 8-9: The cluster found is directly appended to current node N and immediately marked in EA, during the cluster detection.
            // here, we only increase the heuristic value accordingly.
            cluster_heuristic += curr.cluster_found.back()->increased_cost;
            //-----------------------END: Algorithm 1 line 7-10 (found cluster)---------------------------------------------------------
        }
        else if (result == result_type::FBP){
            //-----------------------BEGIN: Algorithm 1 line 11-13 (found bypass)-------------------------------------------------------
            // found a bypass for agent a_m
            if(!applyBypass) continue;
            num_bypass_found++;
            // remove all conflicts of a_m which already been classified.
            for(auto itr = curr.conflicts.begin(); itr != curr.conflicts.end();){
                if((*itr)->a1 == a_m || (*itr)->a2 == a_m ){
                    curr.conflicts.erase(itr++);
                }else{
                    ++itr;
                }
            }

            // remove all conflicts of a_m which already not been classified.
            for(auto itr = curr.unknownConf.begin(); itr != curr.unknownConf.end();){
                if((*itr)->a1 == a_m || (*itr)->a2 == a_m ){
                    curr.unknownConf.erase(itr++);
                }else{
                    ++itr;
                }
            }

            //add bypass.
            bool found = false;
            for(auto& p :  curr.paths){
                if(p.first == a_m){
                    // if the path of a_m has been changed due to the traditional bypass technique.
                    // directly modify the path of a_m.
                    p.second = bypass_path;
                    // change the current bypass to our new bypass.
                    paths[a_m] = &p.second;
                    found = true;
                    break;
                }
            }
            if(!found){
                // path of a_m has not been changed before, add bypass to the changed paths.
                curr.paths.emplace_back(make_pair(a_m, bypass_path));
                // change the current path to bypass.
                paths[a_m] = &curr.paths.back().second;
            }

            // Base on the bypass, remove all edges of a_m from the conflict state graph SG.
            removeEdgesFromSG(a_m,SG);

            // update the new conflicts and insert new edges to the conflict state graph SG.
            // these new conflicts are detected by our labelling method. i.e., we directly get these conflicts from MDD.
            for(auto conflict_agent : mdd_soft_copy.levels[mdd_soft_copy.levels.size()-1].front()->node_conflict_agent){
                // insert new target conflicts.
                shared_ptr<Conflict> conflict(new Conflict());
                for(int i = 0; i < paths[conflict_agent]->size(); i++){
                    if(paths[conflict_agent]->at(i).location ==mdd_soft_copy.levels[mdd_soft_copy.levels.size()-1].front()->location ){
                        if (target_reasoning)
                        {
                            conflict->targetConflict(a_m, conflict_agent,  paths[conflict_agent]->at(i).location, i);
                        }else {
                            conflict->vertexConflict(a_m, conflict_agent, paths[conflict_agent]->at(i).location,i);
                        }
                        break;
                    }
                }
                curr.unknownConf.push_back(conflict);
                // update the conflict state graph SG
                SG[conflict_agent  * num_of_agents + a_m] = true;
                SG[a_m * num_of_agents + conflict_agent ] = true;
            }

            int timestep = mdd_soft_copy.levels.size()-2;
            SoftMDDNode* pre_node_ptr = mdd_soft_copy.levels[mdd_soft_copy.levels.size()-1].front();
            SoftMDDNode* cur_node_ptr = pre_node_ptr->pre_node;
            while(timestep >= 0 ){
                for(auto conflict_agent : cur_node_ptr->node_conflict_agent){
                    // insert new vertex conflicts
                    shared_ptr<Conflict> conflict(new Conflict());
                    if (target_reasoning && cur_node_ptr->location == paths[conflict_agent]->back().location && timestep >= paths[conflict_agent]->size()-1)
                    {
                        conflict->targetConflict(conflict_agent, a_m, bypass_path[timestep].location, timestep);
                    }
                    else
                    {
                        conflict->vertexConflict(a_m, conflict_agent, bypass_path[timestep].location, timestep);
                    }
                    curr.unknownConf.push_back(conflict);
                    // update the conflict state graph SG
                    SG[conflict_agent  * num_of_agents + a_m] = true;
                    SG[a_m * num_of_agents + conflict_agent ] = true;
                }
                if(cur_node_ptr != pre_node_ptr){
                    // insert new edge conflicts.
                    size_t idx = (1 + cur_node_ptr->location ) * map_size + pre_node_ptr->location;
                    if(mdd_soft_copy.edge_conflict_agent[timestep+1].find(idx) != mdd_soft_copy.edge_conflict_agent[timestep+1].end()){
                        for(auto conflict_agent : mdd_soft_copy.edge_conflict_agent[timestep+1][idx]){
                            shared_ptr<Conflict> conflict(new Conflict());
                            conflict->edgeConflict(a_m, conflict_agent, bypass_path[timestep].location, bypass_path[timestep+1].location, (int)timestep+1);
                            curr.unknownConf.push_back(conflict);
                            // update the conflict state graph SG
                            SG[conflict_agent  * num_of_agents + a_m] = true;
                            SG[a_m * num_of_agents + conflict_agent ] = true;
                        }
                    }
                }
                pre_node_ptr = cur_node_ptr;
                cur_node_ptr = cur_node_ptr->pre_node;
                timestep--;
                //-----------------------END: Algorithm 1 line 11-13 (found bypass)-----------------------------------------------------------
            }
            // mark the current node to already found bypass, all new conflicts will be classified later.
            if (!curr.bypassFound) curr.bypassFound  = true;
        }
    }
    //-------------------------BEGIN: Algorithm 1 line 14 (return heuristic)------------------------------------------------------------------
    return cluster_heuristic;
    //-------------------------END: Algorithm 1 line 14 (return heuristic)--------------------------------------------------------------------
}
