#include "ReservationTable.h"


/*int ReservationTable::get_holding_time(int location)
{ 
	auto it = constraints.find(location);
	if (it != constraints.end())
	{
		for (auto constraint : it->second)
			insert_constraint(location, constraint.first, constraint.second);
	}
	
	if (RT.find(location) == RT.end()) 
	{
		return 0;
	}
	int t = std::get<1>(RT[location].back());
	if (t < INTERVAL_MAX)
		return INTERVAL_MAX;
	for (auto p =  RT[location].rbegin(); p != RT[location].rend(); ++p)
	{
		if (t == std::get<1>(*p))
			t = std::get<0>(*p);
		else
			break;
	}
	return t;
}*/


// build the constraint table and the conflict avoidance table
void ReservationTable::buildCAT(int agent, const vector<Path*>& paths)
{
	for (size_t ag = 0; ag < paths.size(); ag++)
	{
		if (ag == agent || paths[ag] == nullptr)
			continue;
		if (paths[ag]->size() == 1) // its start location is its goal location
		{
			cat[paths[ag]->front().location].emplace_back(0, MAX_TIMESTEP);
			continue;
		}
		int prev_location = paths[ag]->front().location;
		int prev_timestep = 0;
		for (size_t timestep = 0; timestep < paths[ag]->size(); timestep++)
		{
			int curr_location = paths[ag]->at(timestep).location;
			if (prev_location != curr_location)
			{
				cat[prev_location].emplace_back(prev_timestep, timestep); // add vertex conflict
				cat[getEdgeIndex(curr_location, prev_location)].emplace_back(timestep, timestep + 1); // add edge conflict
				prev_location = curr_location;
				prev_timestep = timestep;
			}
		}
		cat[paths[ag]->back().location].emplace_back(paths[ag]->size() - 1, MAX_TIMESTEP);
	}
}

int ReservationTable::getNumOfConflictsForStep(size_t curr_id, size_t next_id, size_t next_timestep) const
{
	int rst = 0;
	const auto& it = cat.find(next_id);
	if (it != cat.end())
	{
		for (const auto& constraint : it->second)
		{
			if (constraint.first <= (int) next_timestep && (int) next_timestep < constraint.second)
				rst++;
		}
	}
	const auto& it2 = cat.find(getEdgeIndex(curr_id, next_id));
	if (it2 != cat.end())
	{
		for (const auto& constraint : it2->second)
		{
			if (constraint.first <= (int) next_timestep && (int) next_timestep < constraint.second)
				rst++;
		}
	}
	return rst;
}

void ReservationTable::insert2RT(size_t location, size_t t_min, size_t t_max)
{
	assert(t_min >= 0 && t_min < t_max);
	if (sit.find(location) == sit.end())
	{
		assert(length_min <= length_max);
		int latest_timestep = min(length_max, MAX_TIMESTEP - 1) + 1;
		if (t_min > 0)
		{
			sit[location].emplace_back(0, t_min, 0);
		}
		if ((int) t_max < latest_timestep)
		{
			sit[location].emplace_back(t_max, latest_timestep, 0);
		}
		return;
	}
	for (auto it = sit[location].begin(); it != sit[location].end();)
	{
		if (t_min >= get<1>(*it))
			++it;
		else if (t_max <= get<0>(*it))
			break;
		else if (get<0>(*it) < t_min && get<1>(*it) <= t_max)
		{
			(*it) = make_tuple(get<0>(*it), t_min, 0);
			++it;
		}
		else if (t_min <= get<0>(*it) && t_max < get<1>(*it))
		{
			(*it) = make_tuple(t_max, get<1>(*it), 0);
			break;
		}
		else if (get<0>(*it) < t_min && t_max < get<1>(*it))
		{
			sit[location].insert(it, make_tuple(get<0>(*it), t_min, 0));
			(*it) = make_tuple(t_max, get<1>(*it), 0);
			break;
		}
		else // constraint_min <= get<0>(*it) && get<1> <= constraint_max
		{
			it = sit[location].erase(it);
		}
	}
}


void ReservationTable::insertSoftConstraint2RT(size_t location, size_t t_min, size_t t_max)
{
	if (sit.find(location) == sit.end())
	{
		if (t_min > 0)
		{
			sit[location].emplace_back(0, t_min, 0);
		}
		sit[location].emplace_back(t_min, t_max, 1);
		sit[location].emplace_back(t_max, min(length_max, MAX_TIMESTEP - 1) + 1, 0);
		return;
	}
	for (auto it = sit[location].begin(); it != sit[location].end(); it++)
	{
		if (t_min >= get<1>(*it))
			continue;
		else if (t_max <= get<0>(*it))
			break;

		int conflicts = get<2>(*it);

		if (get<0>(*it) < t_min && get<1>(*it) <= t_max)
		{
			sit[location].insert(it, make_tuple(get<0>(*it), t_min, conflicts));
			(*it) = make_tuple(t_min, get<1>(*it), conflicts + 1);
		}
		else if (t_min <= get<0>(*it) && t_max < get<1>(*it))
		{
			sit[location].insert(it, make_tuple(get<0>(*it), t_max, conflicts + 1));
			(*it) = make_tuple(t_max, get<1>(*it), conflicts);
		}
		else if (get<0>(*it) < t_min && t_max < get<1>(*it))
		{
			sit[location].insert(it, make_tuple(get<0>(*it), t_min, conflicts));
			sit[location].insert(it, make_tuple(t_min, t_max, conflicts + 1));
			(*it) = make_tuple(t_max, get<1>(*it), conflicts);
		}
		else // constraint_min <= get<0>(*it) && get<1> <= constraint_max
		{
			(*it) = make_tuple(get<0>(*it), get<1>(*it), conflicts + 1);
		}
	}
}


//merge successive safe intervals with the same number of conflicts.
/*void ReservationTable::mergeIntervals(list<Interval >& intervals) const
{
	if (intervals.empty())
		return;
	auto prev = intervals.begin();
	auto curr = prev;
	++curr;
	while (curr != intervals.end())
	{
		if (get<1>(*prev) == get<0>(*curr) && get<2>(*prev) == get<2>(*curr))
		{
			*prev = make_tuple(get<0>(*prev), get<1>(*curr), get<2>(*prev));
			curr = intervals.erase(curr);
		}
		else
		{
			prev = curr;
			++curr;
		}
	}
}*/ // we cannot merge intervals for goal locations separated by length_min


// update SIT at the given location
void ReservationTable::updateSIT(size_t location)
{
	if (sit.find(location) == sit.end())
	{
		// length constraints for the goal location
		if (location == goal_location) // we need to divide the same intervals into 2 parts [0, length_min) and [length_min, length_max + 1)
		{
			int latest_timestep = min(length_max, MAX_TIMESTEP - 1) + 1;
			if (length_min > length_max) // the location is blocked for the entire time horizon
			{
				sit[location].emplace_back(0, 0, 0);
				return;
			}
			if (0 < length_min)
			{
				sit[location].emplace_back(0, length_min, 0);
			}
			assert(length_min >= 0);
			sit[location].emplace_back(length_min, latest_timestep, 0);
		}

		// negative constraints
		const auto& it = ct.find(location);
		if (it != ct.end())
		{
			for (auto time_range : it->second)
				insert2RT(location, time_range.first, time_range.second);
			ct.erase(it);
		}

		// positive constraints
		if (location < map_size)
		{
			for (auto landmark : landmarks)
			{
				if (landmark.second != location)
				{
					insert2RT(location, landmark.first, landmark.first + 1);
				}
			}
		}

		// soft constraints
		const auto& it2 = cat.find(location);
		if (it2 != cat.end())
		{
			for (auto time_range : it2->second)
				insertSoftConstraint2RT(location, time_range.first, time_range.second);
			cat.erase(it2);
			// merge the intervals if possible
			auto prev = sit[location].begin();
			auto curr = prev;
			++curr;
			while (curr != sit[location].end())
			{
				if (get<1>(*prev) == get<0>(*curr) && get<2>(*prev) == get<2>(*curr) &&
					(location != goal_location || get<1>(*prev) != length_min))
				{
					*prev = make_tuple(get<0>(*prev), get<1>(*curr), get<2>(*prev));
					curr = sit[location].erase(curr);
				}
				else
				{
					prev = curr;
					++curr;
				}
			}
		}
	}
}

// [lower_bound, upper_bound)
list<Interval> ReservationTable::get_safe_intervals(size_t location, size_t lower_bound, size_t upper_bound)
{
	list<Interval> rst;
	if (lower_bound >= upper_bound)
		return rst;

	updateSIT(location);

	const auto& it = sit.find(location);

	if (it == sit.end())
	{
		rst.emplace_back(0, min(length_max, MAX_TIMESTEP - 1) + 1, 0);
		return rst;
	}

	for (auto interval : sit[location])
	{
		if (lower_bound >= get<1>(interval))
			continue;
		else if (upper_bound <= get<0>(interval))
			break;
		else
		{
			rst.emplace_back(interval);
		}

	}
	return rst;
}

// [lower_bound, upper_bound)
list<Interval> ReservationTable::get_safe_intervals(size_t from, size_t to, size_t lower_bound, size_t upper_bound)
{
	list<Interval> safe_vertex_intervals = get_safe_intervals(to, lower_bound, upper_bound);
	list<Interval> safe_edge_intervals = get_safe_intervals(getEdgeIndex(from, to), lower_bound, upper_bound);

	list<Interval> rst;
	auto it1 = safe_vertex_intervals.begin();
	auto it2 = safe_edge_intervals.begin();
	while (it1 != safe_vertex_intervals.end() && it2 != safe_edge_intervals.end())
	{
		auto t_min = max(get<0>(*it1), get<0>(*it2));
		auto t_max = min(get<1>(*it1), get<1>(*it2));
		if (t_min < t_max)
			rst.emplace_back(t_min, t_max, get<2>(*it1) + get<2>(*it2));
		if (t_max == get<1>(*it1))
			++it1;
		if (t_max == get<1>(*it2))
			++it2;
	}
	return rst;
}

Interval ReservationTable::get_first_safe_interval(size_t location)
{
	updateSIT(location);
	const auto& it = sit.find(location);
	if (it == sit.end())
		return Interval(0, min(length_max, MAX_TIMESTEP - 1) + 1, 0);
	else
		return it->second.front();
}

// find a safe interval with t_min as given
bool ReservationTable::find_safe_interval(Interval& interval, size_t location, size_t t_min)
{
	if (t_min >= min(length_max, MAX_TIMESTEP - 1) + 1)
		return false;
	updateSIT(location);
	const auto& it = sit.find(location);
	if (it == sit.end())
	{
		interval = Interval(t_min, min(length_max, MAX_TIMESTEP - 1) + 1, 0);
		return true;
	}
	for (auto i : it->second)
	{
		if ((int) get<0>(i) <= t_min && t_min < (int) get<1>(i))
		{
			interval = Interval(t_min, get<1>(i), get<2>(i));
			return true;
		}
		else if (t_min < (int) get<0>(i))
			break;
	}
	return false;
}


void ReservationTable::print() const
{
	for (const auto& entry : sit)
	{
		cout << "loc=" << entry.first << ":";
		for (const auto& interval : entry.second)
		{
			cout << "[" << get<0>(interval) << "," << get<1>(interval) << "],";
		}
	}
	cout << endl;
}