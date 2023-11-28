#include "Conflict.h"
#include "RectangleReasoning.h"
#include "MDD.h"


std::ostream& operator<<(std::ostream& os, const Constraint& constraint)
{
	os << "<" << std::get<0>(constraint) << "," << std::get<1>(constraint) << "," <<
	   std::get<2>(constraint) << "," << std::get<3>(constraint) << ",";
	switch (get<4>(constraint))
	{
	case constraint_type::VERTEX:
		os << "V";
		break;
	case constraint_type::POSITIVE_VERTEX:
		os << "V+";
		break;
	case constraint_type::EDGE:
		os << "E";
		break;
	case constraint_type::POSITIVE_EDGE:
		os << "E+";
		break;
	case constraint_type::BARRIER:
		os << "B";
		break;
	case constraint_type::RANGE:
		os << "R";
		break;
	case constraint_type::GLENGTH:
		os << "G";
		break;
	case constraint_type::LEQLENGTH:
		os << "L";
		break;
	}
	os << ">";
	return os;
}


std::ostream& operator<<(std::ostream& os, const Conflict& conflict)
{
	switch (conflict.priority)
	{
	case conflict_priority::CARDINAL:
		os << "cardinal ";
		break;
	case conflict_priority::SEMI:
		os << "semi-cardinal ";
		break;
	case conflict_priority::NON:
		os << "non-cardinal ";
		break;
	case conflict_priority::PRIORITY_COUNT:
		break;
	}
	switch (conflict.type)
	{
	case conflict_type::STANDARD:
		os << "standard";
		break;
	case conflict_type::RECTANGLE:
		os << "rectangle";
		break;
	case conflict_type::CORRIDOR:
		os << "corridor";
		break;
	case conflict_type::TARGET:
		os << "target";
		break;
	case conflict_type::MUTEX:
		os << "mutex";
		break;
	case conflict_type::TYPE_COUNT:
		break;
	}
	os << " conflict:  " << conflict.a1 << " with ";
	for (auto con : conflict.constraint1)
		os << con << ",";
	os << " and " << conflict.a2 << " with ";
	for (auto con : conflict.constraint2)
		os << con << ",";
	return os;
}


// conflict selection
// First compare the cardinality: cardinal > semi-cardinal > non-cardinal (This step can be skipped by the user)
// Second compare the type: mutex > target > corridor > rectangle > vertex/edge
// Third compare the user-specified tie-breaking rule: RANDOM, EARLIEST, CONFLICTS, MCONSTRAINTS, FCONSTRAINTS, WIDTH, SINGLETONS
// Last break ties randomly
// For all the values below, smaller is better
bool operator<(const Conflict& conflict1, const Conflict& conflict2) // return true if conflict2 has higher priority
{
	if (conflict1.priority == conflict2.priority)
	{
		if (conflict1.type == conflict2.type)
		{
			if (conflict1.secondary_priority == conflict2.secondary_priority)
			{
				return random_tie_breaker();
			}
			return conflict1.secondary_priority > conflict2.secondary_priority;
		}
		return conflict1.type > conflict2.type;
	}
	return conflict1.priority > conflict2.priority;
}

bool operator == (const Conflict& conflict1, const Conflict& conflict2)
{
	return (conflict1.a1 == conflict2.a1 &&
				conflict1.a2 == conflict2.a2 &&
				conflict1.constraint1 == conflict2.constraint1 &&
				conflict1.constraint2 == conflict2.constraint2) ||
				(conflict1.a1 == conflict2.a2 &&
				conflict1.a2 == conflict2.a1 &&
				conflict1.constraint1 == conflict2.constraint2 &&
				conflict1.constraint2 == conflict2.constraint1);
}

bool operator != (const Conflict& conflict1, const Conflict& conflict2)
{
	return !(conflict1 == conflict2);
}