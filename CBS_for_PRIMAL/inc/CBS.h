#pragma once

#include "CBSHeuristic.h"
#include "RectangleReasoning.h"
#include "CorridorReasoning.h"
#include "MutexReasoning.h"

class CBS
{
public:
	bool randomRoot = false; // randomize the order of the agents in the root CT node

	/////////////////////////////////////////////////////////////////////////////////////
	// stats
	double runtime = 0;
	double runtime_generate_child = 0; // runtime of generating child nodes
	double runtime_build_CT = 0; // runtime of building constraint table
	double runtime_build_CAT = 0; // runtime of building conflict avoidance table
	double runtime_path_finding = 0; // runtime of finding paths for single agents
	double runtime_detect_conflicts = 0;
	double runtime_preprocessing = 0; // runtime of building heuristic table for the low level

	uint64_t num_corridor_conflicts = 0;
	uint64_t num_rectangle_conflicts = 0;
	uint64_t num_target_conflicts = 0;
	uint64_t num_mutex_conflicts = 0;
	uint64_t num_standard_conflicts = 0;

	uint64_t num_adopt_bypass = 0; // number of times when adopting bypasses

	uint64_t num_HL_expanded = 0;
	uint64_t num_HL_generated = 0;
	uint64_t num_LL_expanded = 0;
	uint64_t num_LL_generated = 0;


	CBSNode* dummy_start = nullptr;
	CBSNode* goal_node = nullptr;



	bool solution_found = false;
	int solution_cost = -2;
	double min_f_val;
	double focal_list_threshold;

	/////////////////////////////////////////////////////////////////////////////////////////
	// set params
	void setHeuristicType(heuristics_type h) {heuristic_helper.type = h; }
    void setClusterHeuristicType(cluster_heuristics_type ch) {heuristic_helper.ch_type = ch; }
	void setPrioritizeConflicts(bool p) {PC = p;	heuristic_helper.PC = p; }
	void setRectangleReasoning(rectangle_strategy r) {rectangle_helper.strategy = r; heuristic_helper.rectangle_reasoning = r; }
	void setCorridorReasoning(corridor_strategy c) {corridor_helper.setStrategy(c); heuristic_helper.corridor_reasoning = c; }
	void setTargetReasoning(bool t) {target_reasoning = t; heuristic_helper.target_reasoning = t; }
	void setMutexReasoning(bool m) {mutex_reasoning = m; heuristic_helper.mutex_reasoning = m; }
	void setDisjointSplitting(bool d) {disjoint_splitting = d; heuristic_helper.disjoint_splitting = d; }
	void setBypass(bool b) { bypass = b; } // 2-agent solver for heuristic calculation does not need bypass strategy.
	void setNodeLimit(int n) { node_limit = n; }
	void setSavingStats(bool s) { save_stats = s; heuristic_helper.save_stats = s; }

	////////////////////////////////////////////////////////////////////////////////////////////
	// Runs the algorithm until the problem is solved or time is exhausted 
	bool solve(double time_limit, int cost_lowerbound = 0, int cost_upperbound = MAX_COST);

	CBS(const Instance& instance, bool sipp, int screen);
	CBS(vector<SingleAgentSolver*>& search_engines,
		const vector<ConstraintTable>& constraints,
		vector<Path>& paths_found_initially, int screen);
	void clearSearchEngines();
	~CBS();

	// Save results
	void saveResults(const string &fileName, const string &instanceName) const;
	void saveStats(const string &fileName, const string &instanceName) const;
	void saveCT(const string &fileName) const; // write the CT to a file
    void savePaths(const string &fileName) const; // write the paths to a file
	list<list<pair<int,int>>> pathMatrix() const;

	void clear(); // used for rapid random  restart

private:
	bool target_reasoning; // using target reasoning
	bool disjoint_splitting; // disjoint splitting
	bool mutex_reasoning; // using mutex reasoning
	bool bypass; // using Bypass1
	bool PC; // prioritize conflicts
	bool save_stats;

	MDDTable mdd_helper;	
	RectangleReasoning rectangle_helper;
	CorridorReasoning corridor_helper;
	MutexReasoning mutex_helper;
	CBSHeuristic heuristic_helper;

	pairing_heap< CBSNode*, compare<CBSNode::compare_node> > open_list;
	pairing_heap< CBSNode*, compare<CBSNode::secondary_compare_node> > focal_list;
	list<CBSNode*> allNodes_table;


	string getSolverName() const;

	int screen;
	
	double time_limit;
	int node_limit = MAX_NODES;
	double focal_w = 1.0;
	int cost_upperbound = MAX_COST;


	vector<ConstraintTable> initial_constraints;
	clock_t start;

	int num_of_agents;


	vector<Path*> paths;
	vector<Path> paths_found_initially;  // contain initial paths found
	// vector<MDD*> mdds_initially;  // contain initial paths found
	vector < SingleAgentSolver* > search_engines;  // used to find (single) agents' paths and mdd


	// high level search
	// bool findPathForSingleAgent(CBSNode*  node, int ag, int lower_bound = 0);
	bool findPathForSingleAgent(CBSNode*  node, int ag, int lower_bound = 0, int direction = 0);
	bool generateChild(CBSNode* child, CBSNode* curr);
	bool generateRoot();

	//conflicts
	void findConflicts(CBSNode& curr);
	void findConflicts(CBSNode& curr, int a1, int a2);
	shared_ptr<Conflict> chooseConflict(const CBSNode &node) const;
	void classifyConflicts(CBSNode &parent);
	// void copyConflicts(const list<shared_ptr<Conflict>>& conflicts,
	// 	list<shared_ptr<Conflict>>& copy, int excluded_agent) const;
	static void copyConflicts(const list<shared_ptr<Conflict>>& conflicts,
		list<shared_ptr<Conflict>>& copy, const list<int>& excluded_agent);
	void removeLowPriorityConflicts(list<shared_ptr<Conflict>>& conflicts) const;
	//bool isCorridorConflict(std::shared_ptr<Conflict>& corridor, const std::shared_ptr<Conflict>& con, bool cardinal, ICBSNode* node);

	void computePriorityForConflict(Conflict& conflict, const CBSNode& node);

	//update information
	inline void updatePaths(CBSNode* curr);
	void updateFocalList();
	inline void releaseNodes();
	//inline void releaseMDDTable();
	// void copyConflictGraph(CBSNode& child, const CBSNode& parent);

	// print and save
	void printPaths() const;
	void printResults() const;
	static void printConflicts(const CBSNode &curr);

	bool validateSolution() const;
	inline int getAgentLocation(int agent_id, size_t timestep) const;
	inline void pushNode(CBSNode* node);
};
