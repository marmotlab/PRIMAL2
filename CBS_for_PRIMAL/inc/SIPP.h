#pragma once

#include "SingleAgentSolver.h"
#include "ReservationTable.h"

class SIPPNode: public LLNode
{
public:
	// define a typedefs for handles to the heaps (allow up to quickly update a node in the heap)
	typedef boost::heap::pairing_heap<SIPPNode*, compare<SIPPNode::compare_node> >::handle_type open_handle_t;
	typedef boost::heap::pairing_heap<SIPPNode*, compare<SIPPNode::secondary_compare_node> >::handle_type focal_handle_t;
	open_handle_t open_handle;
	focal_handle_t focal_handle;

	Interval interval;

	SIPPNode() : LLNode() {}

	SIPPNode(int loc, int dir, int g_val, int h_val, SIPPNode* parent, int timestep, const Interval& interval,
			int num_of_conflicts = 0, bool in_openlist = false) :
			LLNode(loc, dir, g_val, h_val, parent, timestep, num_of_conflicts, in_openlist), interval(interval) {}

	SIPPNode(const SIPPNode& other)
	{
		location = other.location;
		g_val = other.g_val;
		h_val = other.h_val;
		parent = other.parent;
		timestep = other.timestep;
		in_openlist = other.in_openlist;
		open_handle = other.open_handle;
		focal_handle = other.focal_handle;
		num_of_conflicts = other.num_of_conflicts;
		interval = other.interval;
	}
	inline double getFVal() const { return g_val + h_val; }
	~SIPPNode() {}

	// The following is used by for generating the hash value of a nodes
	struct NodeHasher
	{
		std::size_t operator()(const SIPPNode* n) const
		{
			size_t loc_hash = std::hash<int>()(n->location);
			size_t timestep_hash = std::hash<int>()(get<0>(n->interval));
			return (loc_hash ^ (timestep_hash << 1));
		}
	};

	// The following is used for checking whether two nodes are equal
	// we say that two nodes, s1 and s2, are equal if
	// both are non-NULL and agree on the id and timestep
	struct eqnode
	{
		bool operator()(const SIPPNode* n1, const SIPPNode* n2) const
		{
			return (n1 == n2) ||
				   (n1 && n2 && n1->location == n2->location &&
					n1->wait_at_goal == n2->wait_at_goal &&
					get<0>(n1->interval) == get<0>(n2->interval)); //TODO: do we need to compare timestep here?
		}
	};
};

class SIPP: public SingleAgentSolver
{
public:
	// find path by SIPP
	// Returns a shortest path that satisfies the constraints of the given node while
	// minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
	// lowerbound is an underestimation of the length of the path in order to speed up the search.
	Path findPath(const CBSNode& node, const ConstraintTable& initial_constraints,
		const vector<Path*>& paths, int agent, int lowerbound, int direction);
	int getTravelTime(int end, const ConstraintTable& constraint_table, int upper_bound);

	string getName() const { return "SIPP"; }

	SIPP(const Instance& instance, int agent) :
			SingleAgentSolver(instance, agent) {}

private:
	// define typedefs and handles for heap
	typedef boost::heap::pairing_heap<SIPPNode*, boost::heap::compare<LLNode::compare_node> > heap_open_t;
	typedef boost::heap::pairing_heap<SIPPNode*, boost::heap::compare<LLNode::secondary_compare_node> > heap_focal_t;
	heap_open_t open_list;
	heap_focal_t focal_list;

	// define typedef for hash_map
	typedef boost::unordered_set<SIPPNode*, SIPPNode::NodeHasher, SIPPNode::eqnode> hashtable_t;
	hashtable_t allNodes_table;

	int min_f_val; // minimal f value in OPEN
	int lower_bound; // Threshold for FOCAL

	void generateChild(const Interval& interval, SIPPNode* curr, int next_location,
					   const ReservationTable& reservation_table, int lower_bound);

	// Updates the path datamember
	static void updatePath(const LLNode* goal, std::vector<PathEntry> &path);
	inline SIPPNode* popNode();
	inline void pushNode(SIPPNode* node);
	void updateFocalList();
	void releaseNodes();
};

