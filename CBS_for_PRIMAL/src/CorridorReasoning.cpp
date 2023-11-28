#include "CorridorReasoning.h"
#include "Conflict.h"
#include <memory>
#include "SpaceTimeAStar.h"
#include "SIPP.h"

shared_ptr<Conflict> CorridorReasoning::run(const shared_ptr<Conflict>& conflict,
	const vector<Path*>& paths, const CBSNode& node)
{
	clock_t t = clock();
    shared_ptr<Conflict> corridor = nullptr;
    switch(strategy)
    {
        case corridor_strategy::NC:
            return corridor;
        case corridor_strategy::C:
            corridor = findCorridorConflict(conflict, paths, node);
            break;
        case corridor_strategy::PC:
            corridor = findCorridorConflict(conflict, paths, node);
            if (corridor == nullptr)
                corridor = findPseudoCorridorConflict(conflict, paths, node);
            break;
        case corridor_strategy::STC:
            corridor = findCorridorTargetConflict(conflict, paths, node);
            break;
        case corridor_strategy::GC:
        case corridor_strategy::DC:
            corridor = findCorridorTargetConflict(conflict, paths, node);
            if (corridor == nullptr)
                corridor = findPseudoCorridorConflict(conflict, paths, node);
            break;
    }
	accumulated_runtime += (double)(clock() - t) / CLOCKS_PER_SEC;
	return corridor;
}

// the version in the paper
shared_ptr<Conflict> CorridorReasoning::findCorridorConflict(const shared_ptr<Conflict>& conflict,
                                                             const vector<Path*>& paths, const CBSNode& node)
{
    int a[2] = { conflict->a1, conflict->a2 };
    int  agent, loc1, loc2, timestep;
    constraint_type type;
    tie(agent, loc1, loc2, timestep, type) = conflict->constraint1.back();
    int curr = -1;
    if (search_engines[0]->instance.getDegree(loc1) == 2)
    {
        curr = loc1;
        if (loc2 >= 0)
            timestep--;
    }
    else if (loc2 >= 0 && search_engines[0]->instance.getDegree(loc2) == 2)
        curr = loc2;
    if (curr <= 0)
        return nullptr;

    int enter_time[2];
    for (int i = 0; i < 2; i++)
        enter_time[i] = getEnteringTime(*paths[a[i]], *paths[a[1 - i]], timestep);
    if (enter_time[0] > enter_time[1])
    {
        int temp = enter_time[0]; enter_time[0] = enter_time[1]; enter_time[1] = temp;
        temp = a[0]; a[0] = a[1]; a[1] = temp;
    }
    int enter_location[2];
    for (int i = 0; i < 2; i++)
        enter_location[i] = paths[a[i]]->at(enter_time[i]).location;
    if (enter_location[0] == enter_location[1])
        return nullptr;
    for (int i = 0; i < 2; i++)
    {
        bool found = false;
        for (int time = enter_time[i]; time < (int)paths[a[i]]->size() && !found; time++)
        {
            if (paths[a[i]]->at(time).location == enter_location[1 - i])
                found = true;
        }
        if (!found)
            return nullptr;
    }
    pair<int, int> edge; // one edge in the corridor
    int corridor_length = getCorridorLength(*paths[a[0]], enter_time[0], enter_location[1], edge);
    if (corridor_length < 2)
        return nullptr;
    int t3, t3_, t4, t4_;
    ConstraintTable ct1(initial_constraints[a[0]]);
    ct1.build(node, a[0]);
    t3 = search_engines[a[0]]->getTravelTime(enter_location[1], ct1, MAX_TIMESTEP);
    ct1.insert2CT(edge.first, edge.second, 0, MAX_TIMESTEP); // block the corridor in both directions
    ct1.insert2CT(edge.second, edge.first, 0, MAX_TIMESTEP);
    t3_ = search_engines[a[0]]->getTravelTime(enter_location[1], ct1, t3 + 2 * corridor_length + 1);
    ConstraintTable ct2(initial_constraints[a[1]]);
    ct2.build(node, a[1]);
    t4 = search_engines[a[1]]->getTravelTime(enter_location[0], ct2, MAX_TIMESTEP);
    ct2.insert2CT(edge.first, edge.second, 0, MAX_TIMESTEP); // block the corridor in both directions
    ct2.insert2CT(edge.second, edge.first, 0, MAX_TIMESTEP);
    t4_ = search_engines[a[1]]->getTravelTime(enter_location[0], ct2, t3 + corridor_length + 1);

    if (abs(t3 - t4) <= corridor_length && t3_ > t3 && t4_ > t4)
    {
        int t1 = std::min(t3_ - 1, t4 + corridor_length);
        int t2 = std::min(t4_ - 1, t3 + corridor_length);
        list<Constraint> C1, C2;
        C1.emplace_back(a[0], enter_location[1], 0, t1, constraint_type::RANGE);
        C2.emplace_back(a[1], enter_location[0], 0, t2, constraint_type::RANGE);
        if (blocked(*paths[a[0]], C1.front()) &&
            blocked(*paths[a[1]], C2.front()))
        {
            shared_ptr<Conflict> corridor = make_shared<Conflict>();
            corridor->corridorConflict(a[0], a[1], C1, C2);
            return corridor;
        }
    }

    return nullptr;
}

shared_ptr<Conflict> CorridorReasoning::findPseudoCorridorConflict(const shared_ptr<Conflict>& conflict,
	const vector<Path*>& paths, const CBSNode& node)
{
	int  agent, loc1, loc2, timestep;
	constraint_type type;
	tie(agent, loc1, loc2, timestep, type) = conflict->constraint1.back();
	int endpoint1, endpoint2, lowerbound1, lowerbound2; // the timestep of the range constraint at the endpoint should be >= the lowerbound
	if (loc2 < 0) // vertex conflict
	{
		if (paths[conflict->a1]->size() <= timestep + 1 || paths[conflict->a2]->size() <= timestep + 1)
			return nullptr;
		if (paths[conflict->a1]->at(timestep - 1).location != paths[conflict->a2]->at(timestep + 1).location ||
			paths[conflict->a2]->at(timestep - 1).location != paths[conflict->a1]->at(timestep + 1).location)
			return nullptr;
		if (!paths[conflict->a1]->at(timestep - 1).is_single() || 
			!paths[conflict->a1]->at(timestep).is_single() ||
			!paths[conflict->a1]->at(timestep + 1).is_single() ||
			!paths[conflict->a2]->at(timestep - 1).is_single() ||
			!paths[conflict->a2]->at(timestep).is_single() ||
			!paths[conflict->a2]->at(timestep + 1).is_single())
			return nullptr;
		endpoint1 = paths[conflict->a1]->at(timestep + 1).location;
		endpoint2 = loc1;
		lowerbound1 = timestep + 1;
		lowerbound2 = timestep;
	}
	else // edge conflict
	{
		if (!paths[conflict->a1]->at(timestep - 1).is_single() ||
			!paths[conflict->a1]->at(timestep).is_single() ||
			!paths[conflict->a2]->at(timestep - 1).is_single() ||
			!paths[conflict->a2]->at(timestep).is_single())
			return nullptr;
		endpoint1 = loc2;
		endpoint2 = loc1;
		lowerbound1 = timestep;
		lowerbound2 = timestep;
	}

	auto t = getTimeRanges(conflict->a1, conflict->a2, endpoint1, endpoint2, endpoint2, endpoint1,
		lowerbound1, lowerbound2, 1, node); // return (-1, -1) if doesn't exist

	if (t.first >= 0)
	{
		shared_ptr<Conflict> corridor_conflict = make_shared<Conflict>();
		list<Constraint> C1, C2;
		C1.emplace_back(conflict->a1, endpoint1, 0, t.first, constraint_type::RANGE);
		C2.emplace_back(conflict->a2, endpoint2, 0, t.second, constraint_type::RANGE);
		corridor_conflict->corridorConflict(conflict->a1, conflict->a2, C1, C2);
		num_pesudo_corridors++;
		return corridor_conflict;
	}
	return nullptr;
}

shared_ptr<Conflict> CorridorReasoning::findCorridorTargetConflict(const shared_ptr<Conflict>& conflict,
	const vector<Path*>& paths, const CBSNode& node)
{
	assert(conflict->constraint1.size() == 1);
	int  agent, loc1, loc2, timestep;
	constraint_type type;
	tie(agent, loc1, loc2, timestep, type) = conflict->constraint1.back();
	auto corridor = findCorridor(loc1, loc2);
	if (corridor.empty())
		return nullptr;
	int corridor_length = (int)corridor.size() - 1;
	int a[2] = { conflict->a1, conflict->a2 };
	int entry[2] = { -1, -1 }, exit[2] = { -1, -1 }, start[2] = {-1, -1}, goal[2] = { -1, -1 }; // store the corresponding node indices in the corridor
	int goal_time[2] = { -1, -1 };
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < (int)corridor.size(); j++)
		{
			if (paths[a[i]]->front().location == corridor[j]) // the start location is inside the corridor
			{
				start[i] = j; 
				break;
			}
		}
		if (start[i] == -1) // the start location is not inside the corridor
		{
			for (int t = min((int)paths[a[i]]->size(), timestep) - 1; t >= 0; t--) // find the entry point
			{
				if (paths[a[i]]->at(t).location == corridor.front())
				{
					entry[i] = 0;
					break;
				}
				else if (paths[a[i]]->at(t).location == corridor.back())
				{
					entry[i] = (int)corridor.size() - 1;
					break;
				}
			}
		}
		for (int j = 0; j < (int)corridor.size(); j++)
		{
			if (paths[a[i]]->back().location == corridor[j]) // the goal location is inside the corridor
			{
				goal[i] = j;
				goal_time[i] = (int)paths[a[i]]->size() - 1; 
				break;
			}
		}
		if (goal[i] == -1)
		{
			for (int t = timestep; t < (int)paths[a[i]]->size() - 1; t++) // find the exit point
			{
				if (paths[a[i]]->at(t).location == corridor.front())
				{
					exit[i] = 0;
					// goal[i] = 0;
					goal_time[i] = t;
					break;
				}
				else if (paths[a[i]]->at(t).location == corridor.back())
				{
					exit[i] = (int)corridor.size() - 1;
					// goal[i] = exit[i];
					goal_time[i] = t;
					break;
				}
			}
		}
	}
	if ((max(start[0], entry[0]) - max(start[1], entry[1])) * (max(goal[0], exit[0]) - max(goal[1], exit[1])) >= 0) // the start(entry) and goal(exit) locations need to be swaped
		return nullptr;

	list<Constraint> C1, C2;
	if (goal[0] >= 0 || goal[1] >= 0) // The goal location of one of the agents are inside the corridor
	{ // the is a corridor-target conflict no matter the two agents move in the same direction or not
		int middle_agent = (goal[0] >= 0) ? 0 : 1; // will add length constraints to this agent 
		if (start[1] == goal[1] && start[1] >= 0)  // if the start and goal locations of one agent are the same (and, of course, inside the corridor). 
			middle_agent = 1; // We prioritize this agent to be the middle agent 
		ConstraintTable ct1(initial_constraints[a[middle_agent]]);
		ct1.build(node, a[middle_agent]);
		ConstraintTable ct2(initial_constraints[a[1 - middle_agent]]);
		ct2.build(node, a[1 - middle_agent]);
		auto t1 = search_engines[a[middle_agent]]->getTravelTime(corridor.front(), ct1, MAX_TIMESTEP) - 1;
		auto t2 = search_engines[a[1 - middle_agent]]->getTravelTime(corridor.front(), ct2, MAX_TIMESTEP);
		auto l1 = max(t1, t2) + goal[middle_agent];
		t1 = search_engines[a[middle_agent]]->getTravelTime(corridor.back(), ct1, l1 - corridor_length + goal[middle_agent]) - 1;
		t2 = search_engines[a[1 - middle_agent]]->getTravelTime(corridor.back(), ct2, l1 - corridor_length + goal[middle_agent]);
		l1 = min(l1, max(t1, t2) + corridor_length - goal[middle_agent]);
		if (l1 < (int)paths[a[middle_agent]]->size() - 1) // the length constraint for the left child node will not change the paths of the agents
			return nullptr;
		C1.emplace_back(a[middle_agent], corridor[goal[middle_agent]], -1, l1, constraint_type::GLENGTH);
		C2.emplace_back(a[middle_agent], corridor[goal[middle_agent]], -1, l1, constraint_type::LEQLENGTH);
		int dir = max(goal[1 - middle_agent], exit[1 - middle_agent]) - max(start[1 - middle_agent], entry[1 - middle_agent]);
		assert(dir != 0); // the start and goal locations of the other agent cannot be the same, otherwise, the two agents didn't swap their relative locations
		dir = dir / abs(dir);
		int idx = max(exit[1 - middle_agent], goal[1 - middle_agent]); // the index of the exit/goal location
		auto edge = make_pair(corridor[idx], corridor[idx - dir]);
		ct2.insert2CT(edge.first, edge.second, 0, MAX_TIMESTEP); // block the corridor in both directions
		ct2.insert2CT(edge.second, edge.first, 0, MAX_TIMESTEP);
		auto l2 = search_engines[a[1 - middle_agent]]->getTravelTime(edge.first, ct2, MAX_TIMESTEP) - 1;
		if (goal[1 - middle_agent] >= 0) // The goal location of the other agent is also inside the corridor
		{
			if (l2 < (int)paths[a[1 - middle_agent]]->size() - 1) // the length constraint below will not change the path of the agent
				return nullptr;
			C2.emplace_back(a[1 - middle_agent], edge.first, -1, l2, constraint_type::GLENGTH); // add length constraint to C2
		}
		else
		{
			if (goal_time[1 - middle_agent] >= l2) // the range constraint does not block the current path
				return nullptr;
			C2.emplace_back(a[1 - middle_agent], edge.first, 0, l2, constraint_type::RANGE); // add range constraint to C2
		}
		shared_ptr<Conflict> corridor_conflict = make_shared<Conflict>();
		corridor_conflict->corridorConflict(a[middle_agent], a[1 - middle_agent], C1, C2);
		return corridor_conflict;
	}
	else
	{
		int dir = exit[0] - max(start[0], entry[0]);
		assert(dir != 0); // the start and goal locations of the agent cannot be the same, otherwise, the two agents didn't swap their relative locations
		dir = dir / abs(dir);
		auto t = getTimeRanges(a[0], a[1], corridor[exit[0]], corridor[exit[1]], 
			corridor[exit[0] - dir], corridor[exit[1] + dir], goal_time[0], goal_time[1], corridor_length, node);

		if (t.first >= 0)
		{
			shared_ptr<Conflict> corridor_conflict = make_shared<Conflict>();
			C1.emplace_back(a[0], corridor[exit[0]], 0, t.first, constraint_type::RANGE);
			C2.emplace_back(a[1], corridor[exit[1]], 0, t.second, constraint_type::RANGE);
			corridor_conflict->corridorConflict(a[0], a[1], C1, C2);
			return corridor_conflict;
		}
	}
	return nullptr;
}



vector<int> CorridorReasoning::findCorridor(int loc1, int loc2)
{
    list<int> rst;
    if (search_engines[0]->instance.getDegree(loc1) == 2)
    {
        rst.push_back(loc1);
    }
    else if (loc2 >= 0 && search_engines[0]->instance.getDegree(loc2) == 2)
    {
        rst.push_back(loc2);
    }
    if (rst.empty())
        return vector<int>();

    auto root = rst.front();
    auto prev = root;
    auto curr = search_engines[0]->instance.getNeighbors(root).front();
    rst.push_front(curr);
    auto neighbors = search_engines[0]->instance.getNeighbors(curr);
    while (neighbors.size() == 2)
    {
        auto next = (neighbors.front() == prev)? neighbors.back() : neighbors.front();
        rst.push_front(next);
        prev = curr;
        curr = next;
        neighbors = search_engines[0]->instance.getNeighbors(next);
    }
    prev = root;
    curr = search_engines[0]->instance.getNeighbors(root).back();
    rst.push_back(curr);
    neighbors = search_engines[0]->instance.getNeighbors(curr);
    while (neighbors.size() == 2)
    {
        auto next = (neighbors.front() == prev) ? neighbors.back() : neighbors.front();
        rst.push_back(next);
        prev = curr;
        curr = next;
        neighbors = search_engines[0]->instance.getNeighbors(next);
    }

    // When k=2, it might just be a corner cell, which we do not want to recognize as a corridor
    /*if (rst.size() == 3 &&
        search_engines[0]->instance.getColCoordinate(rst.front()) != search_engines[0]->instance.getColCoordinate(rst.back()) &&
        search_engines[0]->instance.getRowCoordinate(rst.front()) != search_engines[0]->instance.getRowCoordinate(rst.back()))
    {
        rst.clear();
    }*/

    return vector<int>(rst.begin(), rst.end());
}


int CorridorReasoning::getEnteringTime(const vector<PathEntry>& path, const vector<PathEntry>& path2, int t)
{
	if (t >= (int) path.size())
		t = (int) path.size() - 1;
	int loc = path[t].location;
	while (loc != path.front().location && loc != path2.back().location &&
		   search_engines[0]->instance.getDegree(loc) == 2)
	{
		t--;
		loc = path[t].location;
	}
	return t;
}


int CorridorReasoning::getCorridorLength(const vector<PathEntry>& path, int t_start, int loc_end, pair<int, int>& edge)
{
	int curr = path[t_start].location;
	int next;
	int prev = -1;
	int length = 0; // distance to the start location
	int t = t_start;
	bool moveForward = true;
	bool updateEdge = false;
	while (curr != loc_end)
	{
		t++;
		next = path[t].location;
		if (next == curr) // wait
			continue;
		else if (next == prev) // turn around
			moveForward = !moveForward;
		if (moveForward)
		{
			if (!updateEdge)
			{
				edge = make_pair(curr, next);
				updateEdge = true;
			}
			length++;
		}
		else
			length--;
		prev = curr;
		curr = next;
	}
	return length;
}


pair<int, int> CorridorReasoning::getTimeRanges(int a1, int a2, int endpoint1, int endpoint2,
	int from1, int from2, int lowerbound1, int lowerbound2, int corridor_length, const CBSNode& node)
{
	int t[2] = {-1, -1}, tprime[2] = {-1, -1};

	ConstraintTable ct1(initial_constraints[a1]);
	ct1.build(node, a1);
	t[0] = search_engines[a1]->getTravelTime(endpoint1, ct1, MAX_TIMESTEP);
	if (t[0] + corridor_length < lowerbound2)
		return make_pair(-1, -1);
	ConstraintTable ct2(initial_constraints[a2]);
	ct2.build(node, a2);
	t[1] = search_engines[a2]->getTravelTime(endpoint2, ct2, MAX_TIMESTEP);
	if (t[1] + corridor_length < lowerbound1)
		return make_pair(-1, -1);
	ct1.insert2CT(from1, endpoint1, 0, MAX_TIMESTEP); // block the corridor in both directions
	ct1.insert2CT(endpoint1,from1, 0, MAX_TIMESTEP); // TODO:: Is this correct? Can we block the entire horizon?
	tprime[0] = search_engines[a1]->getTravelTime(endpoint1, ct1, t[1] + corridor_length + 1);
	if (tprime[0] - 1 < lowerbound1)
		return make_pair(-1, -1);
	ct2.insert2CT(from2, endpoint2, 0, MAX_TIMESTEP); // block the corridor in both directions
	ct2.insert2CT(endpoint2, from2, 0, MAX_TIMESTEP);
	tprime[1] = search_engines[a2]->getTravelTime(endpoint2, ct2, t[0] + corridor_length + 1);
	if (tprime[1] - 1 < lowerbound2)
		return make_pair(-1, -1);
	int t1 = std::min(tprime[0] - 1, t[1] + corridor_length);
	int t2 = std::min(tprime[1] - 1, t[0] + corridor_length);
	return make_pair(t1, t2);
}

bool CorridorReasoning::blocked(const Path& path, const Constraint& constraint)
{
	int a, loc, t1, t2;
	constraint_type type;
	tie(a, loc, t1, t2, type) = constraint;
	assert(type == constraint_type::RANGE);
	for (int t = t1; t < t2; t++)
	{
		if ((t >= (int)path.size() && loc == path.back().location) ||
			(t >= 0 && path[t].location == loc))
			return true;
	}
	return false;
}

