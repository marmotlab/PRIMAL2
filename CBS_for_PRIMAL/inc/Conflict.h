#pragma once

#include "common.h"


enum conflict_type { MUTEX, TARGET, CORRIDOR, RECTANGLE, STANDARD, TYPE_COUNT };

enum conflict_priority { CARDINAL, SEMI, NON, UNKNOWN, PRIORITY_COUNT };
// Pseudo-cardinal conflicts are semi-/non-caridnal conflicts between dependent agents. 
// We prioritize them over normal semi-/non-caridnal conflicts 

enum constraint_type { LEQLENGTH, GLENGTH, RANGE, BARRIER, VERTEX, EDGE, 
											POSITIVE_VERTEX, POSITIVE_EDGE, POSITIVE_BARRIER, POSITIVE_RANGE };

// enum conflict_selection {RANDOM, EARLIEST, CONFLICTS, MCONSTRAINTS, FCONSTRAINTS, WIDTH, SINGLETONS, AGENTID};

typedef std::tuple<int, int, int, int, constraint_type> Constraint;
// <agent, loc, -1, t, VERTEX>
// <agent, loc, -1, t, POSITIVE_VERTEX>
// <agent, from, to, t, EDGE> 
// <agent, B1, B2, t, BARRIER>
// <agent, loc, t1, t2, CORRIDOR> 
// <agent, loc, -1, t, LEQLENGTH>: path of agent_id should be of length at most t, and any other agent cannot be at loc at or after timestep t
// <agent, loc, -1, t, GLENGTH>: path of agent_id should be of length at least t + 1

std::ostream& operator<<(std::ostream& os, const Constraint& constraint);


class Conflict
{
public:
	int a1;
	int a2;
	list<Constraint> constraint1;
	list<Constraint> constraint2;
	conflict_type type;
	conflict_priority priority = conflict_priority::UNKNOWN;
	double secondary_priority = 0; // used as the tie-breaking criteria for conflict selection

	void vertexConflict(int _a1, int _a2, int v, int t)
	{
        constraint1.clear();
        constraint2.clear();
		this->a1 = _a1;
		this->a2 = _a2;
		this->constraint1.emplace_back(a1, v, -1, t, constraint_type::VERTEX);
		this->constraint2.emplace_back(a2, v, -1, t, constraint_type::VERTEX);
		type = conflict_type::STANDARD;
	}
		
	void edgeConflict(int _a1, int _a2, int v1, int v2, int t)
	{
        constraint1.clear();
        constraint2.clear();
		this->a1 = _a1;
		this->a2 = _a2;
		this->constraint1.emplace_back(a1, v1, v2, t, constraint_type::EDGE);
		this->constraint2.emplace_back(a2, v2, v1, t, constraint_type::EDGE);
		type = conflict_type::STANDARD;
	}

	void corridorConflict(int _a1, int _a2, const list<Constraint>& _constraint1, const list<Constraint>& _constraint2)
	{
		this->a1 = _a1;
		this->a2 = _a2;
		this->constraint1 = _constraint1;
		this->constraint2 = _constraint2;
		type = conflict_type::CORRIDOR;
	}

	bool rectangleConflict(int _a1, int _a2, const list<Constraint>& _constraint1, const list<Constraint>& _constraint2) // For RM
	{
		this->a1 = _a1;
		this->a2 = _a2;
		this->constraint1 = _constraint1;
		this->constraint2 = _constraint2;
		type = conflict_type::RECTANGLE;
		return true;
	}


	void targetConflict(int _a1, int _a2, int v, int t)
	{
        constraint1.clear();
        constraint2.clear();
		this->a1 = _a1;
		this->a2 = _a2;
		this->constraint1.emplace_back(a1, v, -1, t, constraint_type::LEQLENGTH);
		this->constraint2.emplace_back(a1, v, -1, t, constraint_type::GLENGTH);
		type = conflict_type::TARGET;
	}


	void mutexConflict(int _a1, int _a2)
	{
		constraint1.clear();
		constraint2.clear();
		this->a1 = _a1;
		this->a2 = _a2;
		type = conflict_type::MUTEX;
		priority = conflict_priority::CARDINAL;
		// TODO add constraints from mutex reasoning
	}
};

std::ostream& operator << (std::ostream& os, const Conflict& conflict);

bool operator < (const Conflict& conflict1, const Conflict& conflict2);

bool operator == (const Conflict& conflict1, const Conflict& conflict2);

bool operator != (const Conflict& conflict1, const Conflict& conflict2);
