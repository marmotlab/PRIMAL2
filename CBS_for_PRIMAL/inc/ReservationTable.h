// This is used by SIPP
#pragma once

#include "ConstraintTable.h"

typedef tuple<size_t, size_t, size_t> Interval; // [t_min, t_max), num_of_collisions

//TODO:: merge ReservationTable with ConstraintTable
class ReservationTable: public ConstraintTable
{
public:
	double runtime;

	ReservationTable() = default;
	ReservationTable(size_t num_col, size_t map_size, int goal_location = -1) : ConstraintTable(num_col, map_size, goal_location) {}
	ReservationTable(const ConstraintTable& other) { copy(other); }


	list<Interval> get_safe_intervals(size_t location, size_t lower_bound, size_t upper_bound);
	list<Interval> get_safe_intervals(size_t from, size_t to, size_t lower_bound, size_t upper_bound);

	// int get_holding_time(int location);
	Interval get_first_safe_interval(size_t location);
	bool find_safe_interval(Interval& interval, size_t location, size_t t_min);

	void buildCAT(int agent, const vector<Path*>& paths); // build the conflict avoidance table

	void print() const;

private:
	// Safe Interval Table (SIT)
	unordered_map<size_t, list<Interval> > sit; // location/edge -> [t_min, t_max), num_of_collisions
	// Conflict Avoidance Table (CAT)
	unordered_map<size_t, list<pair<int, int> > > cat; // location/edge -> time range

	void insert2RT(size_t location, size_t t_min, size_t t_max);
	void insertSoftConstraint2RT(size_t location, size_t t_min, size_t t_max);
	// void mergeIntervals(list<Interval >& intervals) const;

	
	void updateSIT(size_t location); // update SIT at the given location

	int getNumOfConflictsForStep(size_t curr_id, size_t next_id, size_t next_timestep) const;
};