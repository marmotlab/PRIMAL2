#include "SIPP.h"


void SIPP::updatePath(const LLNode* goal, vector<PathEntry>& path)
{
	path.resize(goal->timestep + 1);
	// num_of_conflicts = goal->num_of_conflicts;

	const auto* curr = goal;
	while (curr->parent != nullptr) // non-root node
	{
		const auto* prev = curr->parent;
		int t = prev->timestep + 1;
		while (t < curr->timestep)
		{
			path[t].location = prev->location; // wait at prev location
			t++;
		}
		path[curr->timestep].location = curr->location; // move to curr location
		curr = prev;
	}
	assert(curr->timestep == 0);
	path[0].location = curr->location;
}


// find path by SIPP
// Returns a shortest path that satisfies the constraints of the given node while
// minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
// lowerbound is an underestimation of the length of the path in order to speed up the search.
Path SIPP::findPath(const CBSNode& node, const ConstraintTable& initial_constraints,
					const vector<Path*>& paths, int agent, int lowerbound, int direction)
{
	Path path;
	auto t = clock();
	ReservationTable reservation_table(initial_constraints);
	reservation_table.build(node, agent);
	runtime_build_CT = (double) (clock() - t) / CLOCKS_PER_SEC;
	int holding_time = reservation_table.getHoldingTime();
	t = clock();
	reservation_table.buildCAT(agent, paths);
	runtime_build_CAT = (double) (clock() - t) / CLOCKS_PER_SEC;
	if (reservation_table.length_min >= MAX_TIMESTEP ||
		reservation_table.length_min > reservation_table.length_max) // the agent cannot reach its goal location
		return path;
	num_expanded = 0;
	num_generated = 0;
	Interval interval = reservation_table.get_first_safe_interval(start_location);
	if (get<0>(interval) > 0)
		return path;

	// generate start and add it to the OPEN list
	auto start = new SIPPNode(start_location, start_direction, 0, my_heuristic[start_location], nullptr, 0, interval, 0, false);

	num_generated++;
	start->open_handle = open_list.push(start);
	start->focal_handle = focal_list.push(start);
	start->in_openlist = true;
	allNodes_table.insert(start);
	min_f_val = (int) start->getFVal();
	lower_bound = max(holding_time, max(min_f_val, max(reservation_table.length_min, lowerbound)));


	while (!open_list.empty())
	{
		updateFocalList(); // update FOCAL if min f-val increased
		SIPPNode* curr = focal_list.top();
		focal_list.pop();
		open_list.erase(curr->open_handle);
		curr->in_openlist = false;
		num_expanded++;
		cout << curr->location << endl;

		// check if the popped node is a goal node
		if (curr->location == goal_location && // arrive at the goal location
			!curr->wait_at_goal && // not wait at the goal location
			curr->timestep >= holding_time) // the agent can hold the goal location afterward
		{
			updatePath(curr, path);
			break;
		}

		for (int next_location : instance.getNeighbors(curr->location)) // move to neighboring locations
		{
			for (auto interval : reservation_table.get_safe_intervals(
					curr->location, next_location, curr->timestep + 1, get<1>(curr->interval) + 1))
			{
				generateChild(interval, curr, next_location, reservation_table, lower_bound);
			}
		}  // end for loop that generates successors
		   
		// wait at the current location
		bool found = reservation_table.find_safe_interval(interval, curr->location, get<1>(curr->interval));
		if (found)
		{
			generateChild(interval, curr, curr->location, reservation_table, lower_bound);
		}
	}  // end while loop

	// no path found
	releaseNodes();
	return path;
}

void SIPP::updateFocalList()
{
	auto open_head = open_list.top();
	if (open_head->getFVal() > min_f_val)
	{
		int new_min_f_val = (int) open_head->getFVal();
		int new_lower_bound = max(lower_bound, new_min_f_val);
		for (auto n : open_list)
		{
			if (n->getFVal() > lower_bound && n->getFVal() <= new_lower_bound)
				n->focal_handle = focal_list.push(n);
		}
		min_f_val = new_min_f_val;
		lower_bound = new_lower_bound;
	}
}


inline SIPPNode* SIPP::popNode()
{
	auto node = focal_list.top(); focal_list.pop();
	open_list.erase(node->open_handle);
	node->in_openlist = false;
	num_expanded++;
	return node;
}


inline void SIPP::pushNode(SIPPNode* node)
{
	node->open_handle = open_list.push(node);
	node->in_openlist = true;
	num_generated++;
	if (node->getFVal() <= lower_bound)
		node->focal_handle = focal_list.push(node);
}


void SIPP::releaseNodes()
{
	open_list.clear();
	focal_list.clear();
	for (auto node: allNodes_table)
		delete node;
	allNodes_table.clear();
}


void SIPP::generateChild(const Interval& interval, SIPPNode* curr, int next_location,
						 const ReservationTable& reservation_table, int lower_bound)
{
	// compute cost to next_id via curr node
	int next_timestep = max(curr->timestep + 1, (int) get<0>(interval));
	int next_g_val = next_timestep;
	int next_h_val = my_heuristic[next_location];
	if (next_g_val + next_h_val > reservation_table.length_max)
		return;
	int next_conflicts = curr->num_of_conflicts + get<2>(interval);

	// generate (maybe temporary) node
	auto next = new SIPPNode(next_location, 0, next_g_val, next_h_val, curr, next_timestep, interval, next_conflicts, false);
	if (next_location == goal_location && curr->location == goal_location)
		next->wait_at_goal = true;
	// try to retrieve it from the hash table
	auto it = allNodes_table.find(next);
	if (it == allNodes_table.end())
	{
		pushNode(next);
		allNodes_table.insert(next);
		return;
	}
	// update existing node's if needed (only in the open_list)

	auto existing_next = *it;
	if (existing_next->getFVal() > next->getFVal() || // if f-val decreased through this new path
		(existing_next->getFVal() == next->getFVal() &&
		 existing_next->num_of_conflicts > next->num_of_conflicts)) // or it remains the same but there's fewer conflicts
	{
		if (!existing_next->in_openlist) // if its in the closed list (reopen)
		{
			existing_next->copy(*next);
			pushNode(existing_next);
		}
		else
		{
			bool add_to_focal = false;  // check if it was above the focal bound before and now below (thus need to be inserted)
			bool update_in_focal = false;  // check if it was inside the focal and needs to be updated (because f-val changed)
			bool update_open = false;
			if ((next_g_val + next_h_val) <= lower_bound)
			{  // if the new f-val qualify to be in FOCAL
				if (existing_next->getFVal() > lower_bound)
					add_to_focal = true;  // and the previous f-val did not qualify to be in FOCAL then add
				else
					update_in_focal = true;  // and the previous f-val did qualify to be in FOCAL then update
			}
			if (existing_next->getFVal() > next_g_val + next_h_val)
				update_open = true;

			existing_next->copy(*next);  // update existing node

			if (update_open)
				open_list.increase(existing_next->open_handle);  // increase because f-val improved
			if (add_to_focal)
				existing_next->focal_handle = focal_list.push(existing_next);
			if (update_in_focal)
				focal_list.update(existing_next->focal_handle);  // should we do update? yes, because number of conflicts may go up or down			
		}
	}

	delete next;  // not needed anymore -- we already generated it before
}

// TODO:: currently this is implemented in A*, not SIPP
int SIPP::getTravelTime(int end, const ConstraintTable& constraint_table, int upper_bound)
{
	int length = MAX_TIMESTEP;
	if (constraint_table.length_min >= MAX_TIMESTEP || constraint_table.length_min > constraint_table.length_max || // the agent cannot reach its goal location
		constraint_table.constrained(start_location, 0)) // the agent cannot stay at its start location
	{
		return length;
	}
	auto root = new SIPPNode(start_location, start_direction, 0, compute_heuristic(start_location, end), nullptr, 0, Interval(0, 1, 0));
	root->open_handle = open_list.push(root);  // add root to heap
	allNodes_table.insert(root);       // add root to hash_table (nodes)
	SIPPNode* curr = nullptr;
	while (!open_list.empty())
	{
		curr = open_list.top(); open_list.pop();
		if (curr->location == end)
		{
			length = curr->g_val;
			break;
		}
		list<int> next_locations = instance.getNeighbors(curr->location);
		next_locations.emplace_back(curr->location);
		for (int next_location : next_locations)
		{
			int next_timestep = curr->timestep + 1;
			int next_g_val = curr->g_val + 1;
			if (constraint_table.latest_timestep <= curr->timestep)
			{
				if (curr->location == next_location)
				{
					continue;
				}
				next_timestep--;
			}
			if (!constraint_table.constrained(next_location, next_timestep) &&
				!constraint_table.constrained(curr->location, next_location, next_timestep))
			{  // if that grid is not blocked
				int next_h_val = compute_heuristic(next_location, end);
				if (next_g_val + next_h_val >= upper_bound) // the cost of the path is larger than the upper bound
					continue;
				auto next = new SIPPNode(next_location, 0, next_g_val, next_h_val, nullptr, next_timestep,
										 Interval(next_timestep, next_timestep + 1, 0));
				auto it = allNodes_table.find(next);
				if (it == allNodes_table.end())
				{  // add the newly generated node to heap and hash table
					next->open_handle = open_list.push(next);
					allNodes_table.insert(next);
				}
				else
				{  // update existing node's g_val if needed (only in the heap)
					delete next;  // not needed anymore -- we already generated it before
					auto existing_next = *it;
					if (existing_next->g_val > next_g_val)
					{
						existing_next->g_val = next_g_val;
						existing_next->timestep = next_timestep;
						open_list.increase(existing_next->open_handle);
					}
				}
			}
		}
	}
	releaseNodes();
	return length;
	/*int length = INT_MAX;
	// generate a heap that can save nodes (and a open_handle)
	pairing_heap< SIPPNode*, compare<SIPPNode::compare_node> > open_list;
	// boost::heap::pairing_heap< AStarNode*, boost::heap::compare<LLNode::compare_node> >::handle_type open_handle;
	unordered_set<SIPPNode*, SIPPNode::NodeHasher, SIPPNode::eqnode> nodes;

	Interval interval = reservation_table.get_first_safe_interval(start);
	assert(get<0>(interval) == 0);
	auto root = new SIPPNode(start, 0, instance.getManhattanDistance(start, end), nullptr, 0, interval);
	root->open_handle = open_list.push(root);  // add root to heap
	nodes.insert(root);       // add root to hash_table (nodes)

	while (!open_list.empty())
	{
		auto curr = open_list.top(); open_list.pop();
		if (curr->location == end)
		{
			length = curr->g_val;
			break;
		}
		for (int next_location : instance.getNeighbors(curr->location))
		{
			if ((curr->location == blocked.first && next_location == blocked.second) ||
				(curr->location == blocked.second && next_location == blocked.first)) // use the prohibited edge
			{
				continue;
			}

			for (auto interval : reservation_table.get_safe_intervals(
				curr->location, next_location, curr->timestep + 1, get<1>(curr->interval) + 1))
			{
				int next_timestep = max(curr->timestep + 1, (int)get<0>(interval));
				int next_g_val = next_timestep;
				int next_h_val = instance.getManhattanDistance(next_location, end);
				if (next_g_val + next_h_val >= upper_bound) // the cost of the path is larger than the upper bound
					continue;
				auto next = new SIPPNode(next_location, next_g_val, next_h_val, nullptr, next_timestep, interval);
				auto it = nodes.find(next);
				if (it == nodes.end())
				{  // add the newly generated node to heap and hash table
					next->open_handle = open_list.push(next);
					nodes.insert(next);
				}
				else {  // update existing node's g_val if needed (only in the heap)
					delete(next);  // not needed anymore -- we already generated it before
					auto existing_next = *it;
					if (existing_next->g_val > next_g_val)
					{
						existing_next->g_val = next_g_val;
						existing_next->timestep = next_timestep;
						open_list.update(existing_next->open_handle);
					}
				}
			}
		}
	}
	open_list.clear();
	for (auto node : nodes)
	{
		delete node;
	}
	nodes.clear();
	return length;*/
}