#include "RectangleReasoning.h"
#include <memory>
#include <queue>
#include <stack>


shared_ptr<Conflict> RectangleReasoning::run(const vector<Path*>& paths, int timestep,
											 int a1, int a2, const MDD* mdd1, const MDD* mdd2)
{
	clock_t t = clock();
	shared_ptr<Conflict> rectangle = nullptr;
	switch(strategy)
	{
	case rectangle_strategy::NR:
		return rectangle;
	case rectangle_strategy::R:
        rectangle = findRectangleConflict(paths, timestep, a1, a2);
        break;
	case rectangle_strategy::RM:
		rectangle = findRectangleConflict(paths, timestep, a1, a2, mdd1, mdd2);
		break;
	case rectangle_strategy::GR:
	case rectangle_strategy::DR:
		rectangle = findGenerealizedRectangleConflict(paths, timestep, a1, a2, mdd1, mdd2);
		break;
	}
	accumulated_runtime += (double)(clock() - t) / CLOCKS_PER_SEC;
	return rectangle;
}

// for R
shared_ptr<Conflict> RectangleReasoning::findRectangleConflict(const vector<Path*>& paths, int timestep, int a1, int a2)
{
    auto s1 = instance.getCoordinate(paths[a1]->front().location);
    auto g1 = instance.getCoordinate(paths[a1]->back().location);
    auto s2 = instance.getCoordinate(paths[a2]->front().location);
    auto g2 = instance.getCoordinate(paths[a2]->back().location);
    if (!isRectangleConflict(s1, s2, g1, g2, (int)paths[a1]->size() - 1, (int)paths[a2]->size() - 1))
        return nullptr;
    auto Rs = getRs(s1, s2, g1);
    auto Rg = getRg(s1, g1, g2);
    int Rg_t = abs(Rg.first - s1.first) + abs(Rg.second - s1.second);
    list<Constraint> constraint1;
    list<Constraint> constraint2;
    addBarrierConstraints(a1, a2, Rs, Rg, s1, s2, Rg_t, constraint1, constraint2);
    if (!blocked(*paths[a1], constraint1) || !blocked(*paths[a2], constraint2))
        return nullptr;
    auto rectangle = make_shared<Conflict>();
    rectangle->rectangleConflict(a1, a2, constraint1, constraint2);
    int type = classifyRectangleConflict(s1, s2, g1, g2);
    if (type == 2)
        rectangle->priority = conflict_priority::CARDINAL;
    else if (type == 1)
        rectangle->priority = conflict_priority::SEMI;
    else
        rectangle->priority = conflict_priority::NON;
    return rectangle;
}

// for RM
shared_ptr<Conflict> RectangleReasoning::findRectangleConflict(const vector<Path*>& paths, int timestep,
                                                               int a1, int a2, const MDD* mdd1, const MDD* mdd2)
{
    shared_ptr<Conflict> rectangle = nullptr;
    //Rectangle reasoning for semi and non cardinal vertex conflicts
    list<int> s1s = getStartCandidates(*paths[a1], timestep);
    list<int> g1s = getGoalCandidates(*paths[a1], timestep);
    list<int> s2s = getStartCandidates(*paths[a2], timestep);
    list<int> g2s = getGoalCandidates(*paths[a2], timestep);
    pair<int, int> location = instance.getCoordinate(paths[a1]->at(timestep).location);

    // Try all possible combinations
    int type = -1;
    int area = 0;
    for (int t1_start : s1s)
    {
        for (int t1_end : g1s)
        {
            auto s1 = instance.getCoordinate(paths[a1]->at(t1_start).location);
            auto g1 = instance.getCoordinate(paths[a1]->at(t1_end).location);
            if (instance.getManhattanDistance(s1, g1) !=  t1_end - t1_start)
                continue;
            for (int t2_start : s2s)
            {
                for (int t2_end : g2s)
                {
                    auto s2 = instance.getCoordinate(paths[a2]->at(t2_start).location);
                    auto g2 = instance.getCoordinate(paths[a2]->at(t2_end).location);
                    if (instance.getManhattanDistance(s2, g2) != t2_end - t2_start)
                        continue;
                    if (!isRectangleConflict(s1, s2, g1, g2))
                        continue;
                    auto Rg = getRg(s1, g1, g2);
                    auto Rs = getRs(s1, s2, g1);
                    int new_area = (abs(Rs.first - Rg.first) + 1) * (abs(Rs.second - Rg.second) + 1);
                    auto new_type = classifyRectangleConflict(s1, s2, g1, g2, Rg);
                    if (new_type > type || (new_type == type && new_area > area))
                    {
                        int Rg_t = timestep + abs(Rg.first - location.first) + abs(Rg.second - location.second);
                        list<Constraint> constraint1;
                        list<Constraint> constraint2;
                        bool succ = addModifiedBarrierConstraints(a1, a2, Rs, Rg, s1, s2,
                                                                  Rg_t, mdd1, mdd2, constraint1, constraint2);
                        if (succ && blocked(*paths[a1], constraint1) && blocked(*paths[a2], constraint2))
                        {
                            type = new_type;
                            area = new_area;
                            rectangle = make_shared<Conflict>();
                            rectangle->rectangleConflict(a1, a2, constraint1, constraint2);
                            if (type == 2)
                            {
                                rectangle->priority = conflict_priority::CARDINAL;
                                return rectangle;
                            }
                            else if (type == 1) // && !findRectangleConflict(parent.parent, *conflict))
                                rectangle->priority = conflict_priority::SEMI;
                            else //if (type == 0 && !findRectangleConflict(parent.parent, *conflict))
                                rectangle->priority = conflict_priority::NON;
                        }
                    }
                }
            }
        }
    }

    return rectangle;
}
// for GR
shared_ptr<Conflict> RectangleReasoning::findGenerealizedRectangleConflict(const vector<Path*>& paths, int timestep,
                                                                           int a1, int a2, const MDD* mdd1, const MDD* mdd2)
{
	int conflict_location = paths[a1]->at(timestep).location;
	MDDNode multiple_visits(-1, nullptr);
	// project MDDs to the 2D space
	vector<MDDNode*> visit_times1(instance.map_size, nullptr);
	vector<MDDNode*> visit_times2(instance.map_size, nullptr);
	projectMDD2Map(visit_times1, mdd1, &multiple_visits);
	projectMDD2Map(visit_times2, mdd2, &multiple_visits);
	if (visit_times1[conflict_location]  == &multiple_visits || visit_times2[conflict_location] == &multiple_visits)
		return nullptr;

	// find the purple area and its boundary nodes
	set<pair<int, int> > entry1, entry2; // entry edges
	list<int> exit1, exit2; // exit locations
	vector<bool> overlap_area(instance.map_size, false);
	pair<int, int> Rs(paths[a1]->at(timestep - 1).location, conflict_location), Rg(conflict_location, conflict_location);
	findOverlapArea(conflict_location, overlap_area, visit_times1, visit_times2, entry1, entry2, exit1, exit2, Rs, Rg, &multiple_visits);
	// TODO:: it seems that we do not need exits.
	if (Rs.second == Rg.first) // The overlap area only contains a single location
		return  nullptr;

	//scan the perimeter
	int stage = 0; // 0: initial stage; 1: Rs->R1; 2:Rs->R2; 3:R1->Rs; 4: R2->Rs
	pair<int, int> R1 = Rs, R2 = Rs;
	if(!scanPerimeter(stage, overlap_area, Rs, Rg, R1, R2, entry1, entry2))
		return nullptr;

	// check holes
	if (!checkHoles(entry1, entry2, overlap_area, Rs))
		return nullptr;
	if (!checkHoles(entry2, entry1, overlap_area, Rs))   // TODO:: it seems that we do not need this step?
		return nullptr;


	// generate constraints
	list<Constraint> constraint1, constraint2;
	if (stage == 0 || stage == 2 || stage == 3) // counter-clockwise Rs->R2->Rg->R1->Rs
	{
		constraint1 = generateConstraint(overlap_area, visit_times1, a1, R2, Rg);
		constraint2 = generateConstraint(overlap_area, visit_times2, a2, Rg, R1);
	}
	else // counter-clockwise Rs->R1->Rg->R2->Rs
	{
		constraint1 = generateConstraint(overlap_area, visit_times1, a1, Rg, R2);
		constraint2 = generateConstraint(overlap_area, visit_times2, a2, R1, Rg);
	}
	if (!blocked(*paths[a1], constraint1) || !blocked(*paths[a2], constraint2))
		return nullptr;

	bool cardinal1 = blocked(*mdd1, constraint1);
	bool cardinal2 = blocked(*mdd2, constraint2);
	auto rectangle = make_shared<Conflict>();
	rectangle->rectangleConflict(a1, a2, constraint1, constraint2);
	if (cardinal1 && cardinal2)
		rectangle->priority = conflict_priority::CARDINAL;
	else if (cardinal1 || cardinal2) 
		rectangle->priority = conflict_priority::SEMI;
	else
		rectangle->priority = conflict_priority::NON;
	return rectangle;
}


bool RectangleReasoning::blocked(int location, const list<Constraint>& constraints)
{
	for (auto constraint : constraints)
	{
		if (get<1>(constraint) == location)
		{
			return true;
		}
	}
	return false;
}

bool RectangleReasoning::blocked(const MDD& mdd, const list<Constraint>& constraints)
{
	if (blocked(mdd.levels.back().back()->location, constraints))
		return true;
	std::stack<MDDNode*> open_list;
	set<MDDNode*> visited;
	open_list.push(mdd.levels.back().back());
	while (!open_list.empty())
	{
		auto curr = open_list.top();
		open_list.pop();
		if (curr == mdd.levels.front().front())
			return false;
		for (auto next : curr->parents)
		{
			// assert(next->location >= 0 && next->location < instance.map_size);
			if (visited.find(next) == visited.end() && !blocked(next->location, constraints))
			{
				open_list.push(next);
				visited.insert(next);
			}
		}
	}
	return true;
}


//Identify rectangle conflicts for CR/R
bool RectangleReasoning::isRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
											 const pair<int, int>& g1, const pair<int, int>& g2, int g1_t, int g2_t)
{
	return g1_t == abs(s1.first - g1.first) + abs(s1.second - g1.second) &&  // Manhattan-optimal
		   g2_t == abs(s2.first - g2.first) + abs(s2.second - g2.second) && // Manhattan-optimal
		   (s1.first - g1.first) * (s2.first - g2.first) >= 0 &&  //Move in the same direction
		   (s1.second - g1.second) * (s2.second - g2.second) >= 0; //Move in the same direction
}

//Identify rectangle conflicts for RM
bool RectangleReasoning::isRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2, const pair<int, int>& g1, const pair<int, int>& g2)
{
    return !(s1 == s2 || // A standard cardinal conflict
             s1 == g1 || s2 == g2 || // s1 = g1 or  s2 = g2
             (s1.first - g1.first) * (s2.first - g2.first) < 0 ||
             (s1.second - g1.second) * (s2.second - g2.second) < 0 || // Not move in the same direction
             ((s2.first - s1.first) * (s1.first - g1.first) < 0 &&
              (s2.second - s1.second) * (s1.second - g1.second) < 0) || // s1 always in the middle
             ((s1.first - s2.first) * (s2.first - g2.first) < 0 &&
              (s1.second - s2.second) * (s2.second - g2.second) < 0) || // s2 always in the middle
             (s1.first == g1.first && s2.second == g2.second) || (s1.second == g1.second && s2.first == g2.first)); // area = 1 <==> a cardinal vertex conflct
}

//Classify rectangle conflicts for CR/R
// Return 2 if it is a cardinal rectangle conflict
// Return 1 if it is a semi-cardinal rectangle conflict
// Return 0 if it is a non-cardinal rectangle conflict
int RectangleReasoning::classifyRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
												  const pair<int, int>& g1, const pair<int, int>& g2)
{
	int cardinal1 = 0, cardinal2 = 0;
	if ((s1.first - s2.first) * (g1.first - g2.first) <= 0)
		cardinal1++;
	if ((s1.second - s2.second) * (g1.second - g2.second) <= 0)
		cardinal2++;
	return cardinal1 + cardinal2;
}

//Classify rectangle conflicts for RM
// Return 2 if it is a cardinal rectangle conflict
// Return 1 if it is a semi-cardinal rectangle conflict
// Return 0 if it is a non-cardinal rectangle conflict
int RectangleReasoning::classifyRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
												  const pair<int, int>& g1, const pair<int, int>& g2,
												  const pair<int, int>& Rg)
{
	int cardinal1 = 0, cardinal2 = 0;
	if ((s1.first == s2.first && (s1.second - s2.second) * (s2.second - Rg.second) >= 0) ||
		(s1.first != s2.first && (s1.first - s2.first) * (s2.first - Rg.first) < 0))
	{
		if (Rg.first == g1.first)
			cardinal1 = 1;
		if (Rg.second == g2.second)
			cardinal2 = 1;
	}
	else
	{
		if (Rg.second == g1.second)
			cardinal1 = 1;
		if (Rg.first == g2.first)
			cardinal2 = 1;
	}

	return cardinal1 + cardinal2;
}

//Compute rectangle corner Rs
pair<int, int> RectangleReasoning::getRs(const pair<int, int>& s1, const pair<int, int>& s2, const pair<int, int>& g1)
{
	int x, y;
	if (s1.first == g1.first)
		x = s1.first;
	else if (s1.first < g1.first)
		x = max(s1.first, s2.first);
	else
		x = min(s1.first, s2.first);
	if (s1.second == g1.second)
		y = s1.second;
	else if (s1.second < g1.second)
		y = max(s1.second, s2.second);
	else
		y = min(s1.second, s2.second);
	return make_pair(x, y);
}

//Compute rectangle corner Rg
pair<int, int> RectangleReasoning::getRg(const pair<int, int>& s1, const pair<int, int>& g1, const pair<int, int>& g2)
{
	int x, y;
	if (s1.first == g1.first)
		x = g1.first;
	else if (s1.first < g1.first)
		x = min(g1.first, g2.first);
	else
		x = max(g1.first, g2.first);
	if (s1.second == g1.second)
		y = g1.second;
	else if (s1.second < g1.second)
		y = min(g1.second, g2.second);
	else
		y = max(g1.second, g2.second);
	return make_pair(x, y);
}

//Compute start candidates for RM
list<int> RectangleReasoning::getStartCandidates(const vector<PathEntry>& path, int timestep)
{
	list<int> starts;
	for (int t = 0; t <= timestep; t++) //Find start that is single and Manhattan-optimal to conflicting location
	{
		if (path[t].is_single() && instance.getManhattanDistance(path[t].location, path[timestep].location) == timestep - t)
			starts.push_back(t);
	}
	return starts;
}

//Compute goal candidates for RM
list<int> RectangleReasoning::getGoalCandidates(const vector<PathEntry>& path, int timestep)
{
	list<int> goals;
	for (int t = (int)path.size() - 1; t >= timestep; t--) //Find end that is single and Manhattan-optimal to conflicting location
	{
		if (path[t].is_single() && instance.getManhattanDistance(path[t].location, path[timestep].location) == t - timestep)
			goals.push_back(t);
	}
	return goals;
}

bool RectangleReasoning::addModifiedBarrierConstraints(int a1, int a2, 
	const pair<int, int>& Rs, const pair<int, int>& Rg,
	const pair<int, int>& s1, const pair<int, int>& s2, int Rg_t,
	const MDD* mdd1, const MDD* mdd2,
	list<Constraint>& constraint1, list<Constraint>& constraint2)
{
	if (s1.first == s2.first)
	{
		if ((s1.second - s2.second) * (s2.second - Rg.second) >= 0)
		{
			// first agent moves horizontally and second agent moves vertically
			if (!addModifiedVerticalBarrierConstraint(a1, mdd1, Rg.second, Rs.first, Rg.first, Rg_t, constraint1))
				return false;
			if (!addModifiedHorizontalBarrierConstraint(a2, mdd2, Rg.first, Rs.second, Rg.second, Rg_t, constraint2))
				return false;
		}
		else
		{
			// first agent moves vertically and second agent moves horizontally
			if (!addModifiedHorizontalBarrierConstraint(a1, mdd1, Rg.first, Rs.second, Rg.second, Rg_t, constraint1))
			{
				return false;
			}
			if (!addModifiedVerticalBarrierConstraint(a2, mdd2, Rg.second, Rs.first, Rg.first, Rg_t, constraint2))
			{
				return false;
			}
		}
	}
	else if ((s1.first - s2.first) * (s2.first - Rg.first) >= 0)
	{
		// first agent moves vertically and second agent moves horizontally
		if (!addModifiedHorizontalBarrierConstraint(a1, mdd1, Rg.first, Rs.second, Rg.second, Rg_t, constraint1))
		{
			return false;
		}
		if (!addModifiedVerticalBarrierConstraint(a2, mdd2, Rg.second, Rs.first, Rg.first, Rg_t, constraint2))
		{
			return false;
		}
	}
	else
	{
		// first agent moves horizontally and second agent moves vertically
		if (!addModifiedVerticalBarrierConstraint(a1, mdd1, Rg.second, Rs.first, Rg.first, Rg_t, constraint1))
		{
			return false;
		}
		if (!addModifiedHorizontalBarrierConstraint(a2, mdd2, Rg.first, Rs.second, Rg.second, Rg_t, constraint2))
		{
			return false;
		}
	}
	return true;
}

// add a horizontal modified barrier constraint
bool RectangleReasoning::addModifiedHorizontalBarrierConstraint(int agent, const MDD* mdd, int x,
																int Ri_y, int Rg_y, int Rg_t, list<Constraint>& constraints)
{
	int sign = Ri_y < Rg_y ? 1 : -1;
	int Ri_t = Rg_t - abs(Ri_y - Rg_y);
	int t1 = -1;
	int t_min = max(Ri_t, 0);
	int t_max = min(Rg_t, (int) mdd->levels.size() - 1);
	for (int t2 = t_min; t2 <= t_max; t2++)
	{
		int loc = instance.linearizeCoordinate(x, (Ri_y + (t2 - Ri_t) * sign));
		MDDNode* it = nullptr;
		for (MDDNode* n : mdd->levels[t2])
		{
			if (n->location == loc)
			{
				it = n;
				break;
			}
		}
		if (it == nullptr && t1 >= 0) // add constraints [t1, t2)
		{
			int loc1 = instance.linearizeCoordinate(x, (Ri_y + (t1 - Ri_t) * sign));
			int loc2 = instance.linearizeCoordinate(x, (Ri_y + (t2 - 1 - Ri_t) * sign));
			constraints.emplace_back(agent, loc1, loc2, t2 - 1, constraint_type::BARRIER);
			t1 = -1;
			continue;
		}
		else if (it != nullptr && t1 < 0)
		{
			t1 = t2;
		}
		if (it != nullptr && t2 == t_max)
		{
			int loc1 = instance.linearizeCoordinate(x, (Ri_y + (t1 - Ri_t) * sign));
			constraints.emplace_back(agent, loc1, loc, t2, constraint_type::BARRIER); // add constraints [t1, t2]
		}
	}
	return !constraints.empty();
}


// add a vertical modified barrier constraint
bool RectangleReasoning::addModifiedVerticalBarrierConstraint(int agent, const MDD* mdd, int y,
															  int Ri_x, int Rg_x, int Rg_t, list<Constraint>& constraints)
{
	int sign = Ri_x < Rg_x ? 1 : -1;
	int Ri_t = Rg_t - abs(Ri_x - Rg_x);
	int t1 = -1;
	int t_min = max(Ri_t, 0);
	int t_max = min(Rg_t, (int) mdd->levels.size() - 1);
	for (int t2 = t_min; t2 <= t_max; t2++)
	{
		int loc = instance.linearizeCoordinate((Ri_x + (t2 - Ri_t) * sign), y);
		MDDNode* it = nullptr;
		for (MDDNode* n : mdd->levels[t2])
		{
			if (n->location == loc)
			{
				it = n;
				break;
			}
		}
		if (it == nullptr && t1 >= 0) // add constraints [t1, t2)
		{
			int loc1 = instance.linearizeCoordinate((Ri_x + (t1 - Ri_t) * sign), y);
			int loc2 = instance.linearizeCoordinate((Ri_x + (t2 - 1 - Ri_t) * sign), y);
			constraints.emplace_back(agent, loc1, loc2, t2 - 1, constraint_type::BARRIER);
			t1 = -1;
			continue;
		}
		else if (it != nullptr && t1 < 0)
		{
			t1 = t2;
		}
		if (it != nullptr && t2 == t_max)
		{
			int loc1 = instance.linearizeCoordinate((Ri_x + (t1 - Ri_t) * sign), y);
			constraints.emplace_back(agent, loc1, loc, t2, constraint_type::BARRIER); // add constraints [t1, t2]
		}
	}
	return !constraints.empty();
}

bool RectangleReasoning::blocked(const Path& path, const list<Constraint>& constraints)
{
	for (auto constraint : constraints)
	{
		int a, x, y, t;
		constraint_type type;
		tie(a, x, y, t, type) = constraint;
		if (type == constraint_type::BARRIER)
		{
			int x1 = instance.getRowCoordinate(x), y1 = instance.getColCoordinate(x);
			int x2 = instance.getRowCoordinate(y), y2 = instance.getColCoordinate(y);
			if (x1 == x2)
			{
				if (y1 < y2)
				{
					for (int i = 0; i <= min(y2 - y1, t); i++)
					{
						if (traverse(path, instance.linearizeCoordinate(x1, y2 - i), t - i))
							return true;
					}
				}
				else
				{
					for (int i = 0; i <= min(y1 - y2, t); i++)
					{
						if (traverse(path, instance.linearizeCoordinate(x1, y2 + i), t - i))
							return true;
					}
				}
			}
			else // y1== y2
			{
				if (x1 < x2)
				{
					for (int i = 0; i <= min(x2 - x1, t); i++)
					{
						if (traverse(path, instance.linearizeCoordinate(x2 - i, y1), t - i))
							return true;
					}
				}
				else
				{
					for (int i = 0; i <= min(x1 - x2, t); i++)
					{
						if (traverse(path, instance.linearizeCoordinate(x2 + i, y1), t - i))
							return true;
					}
				}
			}
		}
		else if (type == constraint_type::VERTEX)
		{
			if (path[t].location == x)
				return true;
		}
	}
	return false;
}

bool RectangleReasoning::traverse(const Path& path, int loc, int t)
{
	if (t >= (int) path.size())
		return loc == path.back().location;
	else return t >= 0 && path[t].location == loc;
}


// add a pair of barrier constraints
void RectangleReasoning::addBarrierConstraints(int a1, int a2, const pair<int, int>& Rs, const pair<int, int>& Rg,
                                               const pair<int, int>& s1, const pair<int, int>& s2, int Rg_t,
                                               list<Constraint>& constraint1, list<Constraint>& constraint2)
{
    pair<int, int> R1, R2;
    assert(s1.first != s2.first);
    if ((s1.first - s2.first)*(s2.first - Rg.first) >= 0)
    {
        R1.first = Rg.first;
        R2.first = s2.first;
        R1.second = s1.second;
        R2.second = Rg.second;
    }
    else
    {
        R1.first = s1.first;
        R2.first = Rg.first;
        R1.second = Rg.second;
        R2.second = s2.second;
    }
    int r1 = instance.linearizeCoordinate(R1);
    int r2 = instance.linearizeCoordinate(R2);
    int rg = instance.linearizeCoordinate(Rg);
    constraint1.emplace_back(a1, r1, rg, Rg_t, constraint_type::BARRIER);
    constraint2.emplace_back(a2, r2, rg, Rg_t, constraint_type::BARRIER);
}



////////////////////////////////////////////////////////////////////////////
/// Tools for GR
//////////////////////////////////////////////////////////////////////////////
void RectangleReasoning::projectMDD2Map(vector<MDDNode*>& mapping, const MDD* mdd, MDDNode* multiple_visits)
{
    for (auto const& level : mdd->levels)
    {
        for (auto const& node : level)
        {
            if (mapping[node->location] != nullptr) // if the location has been visited before
                mapping[node->location] = multiple_visits;
            else
                mapping[node->location] = node;  // >=0 means the locations can only be visited at the timestep
        }
    }
}
void RectangleReasoning::findOverlapArea(int conflict_location, vector<bool>& overlap_area,
                                         const vector<MDDNode*>& visit_times1, const vector<MDDNode*>& visit_times2,
                                         set<pair<int, int> >& entry1, set<pair<int, int> >& entry2, list<int>& exit1, list<int>& exit2,
                                         pair<int, int>& Rs, pair<int, int>& Rg, const MDDNode* multiple_visits) const
{
    std::queue<int> open_list;
    open_list.push(conflict_location);
    overlap_area[conflict_location] = true;
    while (!open_list.empty())
    {
        int curr = open_list.front();
        open_list.pop();
        if (visit_times1[Rg.first]->level < visit_times1[curr]->level)
            Rg = make_pair(curr, curr); // Rg is the overlap location with the maximum timestep
        if (visit_times1[curr]->children.empty()) // goal loc of agent 1
        {
            exit1.emplace_back(curr);
        }
        else if (visit_times2[curr]->children.empty()) // goal loc of agent 2
        {
            exit2.emplace_back(curr);
        } // cannot be the start loc of either agent; cannot be the goal loc of both agents
        for (auto next : instance.getNeighbors(curr))
        {
            if (overlap_area[next])
            {
                continue;
            }
            else if (visit_times1[next] != multiple_visits && visit_times1[next] != nullptr &&
                     visit_times2[next] != multiple_visits && visit_times2[next] != nullptr &&
                     visit_times1[next]->level == visit_times2[next]->level)
            {
                open_list.push(next);
                overlap_area[next] = true;
                continue;
            }
            if (visit_times1[Rs.second]->level > visit_times1[curr]->level)
                Rs = make_pair(next, curr); // Rg is the overlap location with the minimum timestep
            if (visit_times1[next] != nullptr) // entry or exit of agent 1
            {
                for (auto child : visit_times1[curr]->children)
                {
                    if (child->location == next) // exit of agent 1
                    {
                        if (exit1.empty() || exit1.back() != curr)
                            exit1.emplace_back(curr);
                        break;
                    }
                }
                for (auto parent : visit_times1[curr]->parents)
                {
                    if (parent->location == next) // entry of agent 1
                    {
                        entry1.emplace(next, curr);
                        break;
                    }
                }
            }
            if (visit_times2[next] != nullptr) // entry or exit of agent 2
            {
                for (auto child : visit_times2[curr]->children)
                {
                    if (child->location == next) // exit of agent 2
                    {
                        if (exit2.empty() || exit2.back() != curr)
                            exit2.emplace_back(curr);
                        break;
                    }
                }
                for (auto parent : visit_times2[curr]->parents)
                {
                    if (parent->location == next) // entry of agent 2
                    {
                        entry2.emplace(next, curr);
                        break;
                    }
                }
            }
        }
    }
}
bool RectangleReasoning::scanPerimeter(int& stage, const vector<bool>& overlap_area,
                                       const pair<int, int>& Rs, const pair<int, int>& Rg,
                                       pair<int, int>& R1, pair<int, int>& R2,
                                       set<pair<int, int> >& entry1, set<pair<int, int> >& entry2) const
{
    int curr = Rs.first; //an external location
    while (!overlap_area[curr])
        curr = instance.walkCounterClockwise(Rs.second, curr);
    int prev = Rs.second;
    bool passRg = false;
    while (curr != Rs.second)
    {
        if (curr == Rg.first)
            passRg = true;
        int ex = instance.walkCounterClockwise(curr, prev);
        while (!overlap_area[ex]) // external location
        {
            pair<int, int> edge(ex, curr);
            auto found = entry1.find(edge);
            if (found != entry1.end()) // entry for agent 1
            {
                switch (stage)
                {
                    case 0: // inital stage
                        if (passRg)
                            stage = 3; // R1->Rs
                        else
                        	stage = 1; // Rs->R1
                        R1 = edge;
                        break;
                    case 1: // Rs->R1
                        if (passRg)
                            return false;
                        R1 = edge;
                        break;
                    case 2: //Rs->R2
						if (!passRg)
							return false;
                        stage = 3; //R1->Rs
                        R1 = edge;
                        break;
                    case 3: // R1->Rs
                        break;
                    case 4: // R2->Rs
                        return false;
                    default:
                        return false; // this should never happen
                }
                entry1.erase(found);
            }
            found = entry2.find(edge);
            if (found != entry2.end()) // entry for agent 2
            {
                switch (stage)
                {
                    case 0: // inital stage
                        if (passRg)
                            stage = 4; // R2->Rs
                        else
                            stage = 2; // Rs->R2
                        R2 = edge;
                        break;
                    case 1: // Rs->R1
						if (!passRg)
							return false;
                        stage = 4; // R2->Rs
                        R2 = edge;
                        break;
                    case 2: //Rs->R2
                        if (passRg)
                            return false;
                        R2 = edge;
                        break;
                    case 3: // R1->Rs
                        return false;
                    case 4: // R2->Rs
                        break;
                    default:
                        return false; // this should never happen
                }
                entry2.erase(found);
            }
            ex = instance.walkCounterClockwise(curr, ex); // scan the next edge
        }
        prev = curr;
        curr = ex;
    }
    return passRg; // TODO: temporal method
}
bool RectangleReasoning::checkHoles(set<pair<int, int> >& entry1, set<pair<int, int> >& entry2,
                                    const vector<bool>& overlap_area, const pair<int, int>& Rs) const
{
    while (!entry1.empty())
    {
        int prev = entry1.begin()->first;
        int curr = entry1.begin()->second;
        entry1.erase(entry1.begin());
        if (curr == Rs.second) // ignore entries at Rs
            continue;
        int origin = curr;
        do
        {
            int ex = instance.walkCounterClockwise(curr, prev);
            while (!overlap_area[ex]) // external location
            {
                pair<int, int> edge(ex, curr);
                auto found = entry1.find(edge);
                if (found != entry1.end()) // entry for agent 1
                {
                    entry1.erase(found);
                }
                if (entry2.find(edge) != entry2.end()) // entry for agent 2
                {
                    return false;
                }
                ex = instance.walkCounterClockwise(curr, ex); // scan the next edge
            }
            prev = curr;
            curr = ex;
        } while (curr != origin);
    }
    return true;
}
list<Constraint> RectangleReasoning::generateConstraint(const vector<bool>& overlap_area,
                                                        const vector<MDDNode*>& visit_times, int agent, const pair<int, int>& from, const pair<int, int>& to) const // generate constraints from from to to along the perimeter counter-clockwisely
{
    list<Constraint> constraints;
    constraints.emplace_back(agent, from.second, -1, visit_times[from.second]->level, constraint_type::VERTEX);
    if (from.second == to.second)
        return constraints;
    int prev = from.second;
    int curr;
    if (from.first == from.second)
    {
        for (auto neighbor : instance.getNeighbors(from.second))
        {
            if (!overlap_area[neighbor]) // external location
            {
                curr = neighbor;
                break;
            }
        }
    }
    else
    {
        curr = from.first;
    }
    do
    {
        curr = instance.walkCounterClockwise(from.second, curr);
    } while (!overlap_area[curr]); // internal location

    constraints.emplace_back(agent, curr, -1, visit_times[curr]->level, constraint_type::VERTEX);
    while (curr != to.second)
    {
        int ex = instance.walkCounterClockwise(curr, prev);
        while (!overlap_area[ex]) // external location
        {
            ex = instance.walkCounterClockwise(curr, ex); // scan the next edge
        }
        prev = curr;
        curr = ex;
        constraints.emplace_back(agent, curr, -1, visit_times[curr]->level, constraint_type::VERTEX);
    }
    return constraints;
}
void RectangleReasoning::printOverlapArea(const vector<Path*>& paths, int conflict_timestep,
                                          int a1, int a2, const MDD* mdd1, const MDD* mdd2)
{
    int conflict_location = paths[a1]->at(conflict_timestep).location;
    MDDNode multiple_visits(-1, nullptr);
    // project MDDs to the 2D space
    vector<MDDNode*> visit_times1(instance.map_size, nullptr);
    vector<MDDNode*> visit_times2(instance.map_size, nullptr);
    projectMDD2Map(visit_times1, mdd1, &multiple_visits);
    projectMDD2Map(visit_times2, mdd2, &multiple_visits);
    if (visit_times1[conflict_location] == &multiple_visits || visit_times2[conflict_location] == &multiple_visits)
        return;

    // find the purple area and its boundary nodes
    set<pair<int, int> > entry1, entry2; // entry edges
    list<int> exit1, exit2; // exit locations
    vector<bool> overlap_area(instance.map_size, false);
    pair<int, int> Rs(paths[a1]->at(conflict_timestep - 1).location, conflict_location), Rg(conflict_location, conflict_location);
    findOverlapArea(conflict_location, overlap_area, visit_times1, visit_times2, entry1, entry2, exit1, exit2, Rs, Rg, &multiple_visits);
    cout << "Overlap area: ";
    for (size_t i = 0; i < overlap_area.size(); i++)
    {
        if (overlap_area[i])
            cout << i << ",";
    }
    cout << endl;
    cout << "Entry for agent " << a1 << ": ";
    for (auto loc : entry1)
        cout << loc.first << ",";
    cout << endl;
    cout << "Entry for agent " << a2 << ": ";
    for (auto loc : entry2)
        cout << loc.first << ",";
    cout << endl;
}