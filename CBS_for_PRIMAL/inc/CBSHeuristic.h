#pragma once

#include "MDD.h"
#include "RectangleReasoning.h"
#include "CorridorReasoning.h"
#include "ConstraintPropagation.h"
#include "SMDDConstraintPropagation.h"

enum heuristics_type { ZERO, CG, DG, WDG, STRATEGY_COUNT };
enum cluster_heuristics_type { N, BP, CH, CHBPNS, CHBPNM,CHBPNRM, CHBP};
// N: No bypass and cluster heuristic applied.
// BP: Only apply bypass, but no cluster heuristic applied.
// CH: Only apply cluster heuristic, no bypass applied.
// CHBP: Our final algorithm cluster heuristic and bypass, with all optimization techniques.

// For ablation study:
// CHBPNS: Cluster heuristic and bypass, but do not apply solving the cluster.
// CHBPNM: Cluster heuristic and bypass, but do not apply memoization.
// BPMHNRM: Cluster heuristic and bypass, but do not apply reusable mutex propagation,

enum result_type{FNCNP,FBP,FC};
//FNCNP find no cluster no bypass;
//FBP find bypass.
//FHI find cluster.

typedef vector<tuple<int,int,int>> mutex_nodes;
typedef unordered_map<DoubleConstraintsHasher, int, DoubleConstraintsHasher::Hasher, DoubleConstraintsHasher::EqNode> HTable;
typedef unordered_map<MultiConstraintsHasher, int, MultiConstraintsHasher::Hasher, MultiConstraintsHasher::EqNode> CHTable;
typedef unordered_map<DoubleConstraintsHasher, pair<shared_ptr<mutex_nodes>,shared_ptr<mutex_nodes>>, DoubleConstraintsHasher::Hasher, DoubleConstraintsHasher::EqNode> MTable;

class CBSHeuristic
{
public:
	heuristics_type type;
    cluster_heuristics_type ch_type;
	rectangle_strategy rectangle_reasoning; // using rectangle reasoning
	corridor_strategy corridor_reasoning; // using corridor reasoning
	bool target_reasoning; // using target reasoning
	bool mutex_reasoning; // using mutex reasoning
	bool disjoint_splitting; // disjoint splitting
	bool PC; // prioritize conflicts
	bool save_stats;

	double runtime_build_dependency_graph = 0;
	double runtime_solve_MVC = 0;

	uint64_t num_merge_MDDs = 0;
	uint64_t num_solve_2agent_problems = 0;
	uint64_t num_memoization = 0; // number of times when memeorization helps
	 //stats
	list<tuple<int, int, const CBSNode*, uint64_t, int> > sub_instances; 	// <agent 1, agent 2, node, number of expanded CT nodes, h value> 


    // used for AAAI paper, need to be removed after acceptance.
    uint64_t num_cluster_solved = 0;
    uint64_t num_cluster_memorized = 0;
    uint64_t num_mutex_checking = 0;
    uint64_t num_mutex_memorized = 0;
    uint64_t num_second_heuristic_computed = 0;
    uint64_t num_second_heuristic_increased = 0;
    uint64_t total_second_heuristic_increased = 0;
    uint64_t num_node_found_bypass = 0;
    uint64_t num_bypass_found = 0;
    uint64_t num_cluster_detected = 0;
    uint64_t total_cluster_size = 0;
    uint64_t total_h_value= 0;
    uint64_t max_fh = 0;
    uint64_t total_fh_value= 0;
    uint64_t total_fh_computed= 0;
    uint64_t max_num_cluster= 0;
    uint64_t max_cluster_size= 0;
    uint64_t max_second_h= 0;
    uint64_t max_num_bypass= 0;
    double runtime_of_computing_second_heuristic = 0;


    // variable used to run different variations of algorithm
    bool applyMutexCache = false;
    bool applyHeuristicCache = false;
    bool applySolvingCluster = false;
    bool applyBypass = false;
    bool applyMutexHeuristic = false;

	CBSHeuristic(int num_of_agents,
               vector<Path*>& paths,
               vector<SingleAgentSolver*>& search_engines,
               const vector<ConstraintTable>& initial_constraints,
               MDDTable& mdd_helper) : num_of_agents(num_of_agents),
                                       paths(paths), search_engines(search_engines),
                                       initial_constraints(initial_constraints), mdd_helper(mdd_helper) {}

	void init()
	{
		if (type == heuristics_type::DG || type == heuristics_type::WDG)
		{
			lookupTable.resize(num_of_agents);
			for (int i = 0; i < num_of_agents; i++)
			{
				lookupTable[i].resize(num_of_agents);
			}
		}
        // for ablation study;
        switch (ch_type) {
            case cluster_heuristics_type::N: {
                applyMutexCache = false;
                applyHeuristicCache = false;
                applySolvingCluster = false;
                applyBypass = false;
                applyMutexHeuristic = false;
                break;
            }
            case cluster_heuristics_type::BP: {
                // do not use any heuristic;
                applyMutexCache = true;
                applyHeuristicCache = false;
                applySolvingCluster = false;
                applyBypass = true;
                applyMutexHeuristic = false;
                break;
            }
            case cluster_heuristics_type::CH: {
                // do not use bypass;
                applyMutexCache = true;
                applyHeuristicCache = true;
                applySolvingCluster = true;
                applyBypass = false;
                applyMutexHeuristic = true;
                break;
            }
            case cluster_heuristics_type::CHBPNM: {
                // no memoization
                applyMutexCache = false;
                applyHeuristicCache = false;
                applySolvingCluster = true;
                applyBypass = true;
                applyMutexHeuristic = true;
                break;
            }
            case cluster_heuristics_type::CHBPNRM: {
                // no reusable mutex propagation.
                applyMutexCache = false;
                applyHeuristicCache = true;
                applySolvingCluster = true;
                applyBypass = true;
                applyMutexHeuristic = true;
                break;
            }
            case cluster_heuristics_type::CHBPNS: {
                // do not solve the cluster.
                applyMutexCache = true;
                applyHeuristicCache = false;
                applySolvingCluster = false;
                applyBypass = true;
                applyMutexHeuristic = true;
                break;
            }
            case cluster_heuristics_type::CHBP: {
                // apply all optimization
                applyMutexCache = true;
                applyHeuristicCache = true;
                applySolvingCluster = true;
                applyBypass = true;
                applyMutexHeuristic = true;
                break;
            }
        }
        if(ch_type != cluster_heuristics_type::N){
            mdd_soft_copy.init(10000);
            map_size = search_engines.front()->instance.map_size;
            if(applyMutexCache){
                cp_helper.init(search_engines.front()->instance.map_size);
                mutexLookupTable.resize(num_of_agents);
                CHLookupTable.resize(num_of_agents);
                for (int i = 0; i < num_of_agents; i++)
                {
                    mutexLookupTable[i].resize(num_of_agents);
                    CHLookupTable[i].resize(num_of_agents);
                }
            }else{
                smdd_cp_helper.init(search_engines.front()->instance.map_size);
                CHLookupTable.resize(num_of_agents);
                for (int i = 0; i < num_of_agents; i++)
                {
                    CHLookupTable[i].resize(num_of_agents);
                }
            }
        }
	}

	bool computeInformedHeuristics(CBSNode& curr, double time_limit); // this function is called when poping a CT node for the first time
	void computeQuickHeuristics(CBSNode& curr) const; // this function is called when generating a CT node
	void copyConflictGraph(CBSNode& child, const CBSNode& parent) const;
	void clear() { lookupTable.clear(); }

private:
	int screen = 0;
	int num_of_agents;
	int DP_node_threshold = 8; // run dynamic programming (=brute-force search) only when #nodes <= th
	// int DP_product_threshold = 1024;  // run dynamic programming (=brute-force search) only when #\product range <= th
	vector<vector<HTable> > lookupTable;

	double time_limit;
	int node_limit = 10;  // terminate the sub CBS solver if the number of its expanded nodes exceeds the node limit.
	double start_time;

	vector<Path*>& paths;
	const vector<SingleAgentSolver*>& search_engines;
	const vector<ConstraintTable>& initial_constraints;
	MDDTable& mdd_helper;


    vector<vector<CHTable>> CHLookupTable; // cluster heuristic look up table
    vector<vector<MTable> > mutexLookupTable; // mutex lookup table
    SoftMDD mdd_soft_copy; // for quick copy of MDD
    SMDDConstraintPropagation smdd_cp_helper; // for mutex propagation.
    ConstraintPropagation cp_helper; // for mutex propagation.
    size_t map_size;

	vector<int> buildConflictGraph(const CBSNode& curr) const;
	void buildCardinalConflictGraph(CBSNode& curr, vector<int>& CG, vector<bool>& excluded_agents, int& num_of_CGedges);
	bool buildDependenceGraph(CBSNode& node, vector<int>& CG, vector<bool>& excluded_agents, int& num_of_CGedges);
	bool buildWeightedDependencyGraph(CBSNode& curr, vector<int>& CG, vector<bool>& excluded_agents);

	bool dependent(int a1, int a2, CBSNode& node); // return true if the two agents are dependent
	int solve2Agents(int a1, int a2, const CBSNode& node, bool cardinal);
	static bool SyncMDDs(const MDD &mdd1, const MDD& mdd2); 	// Match and prune MDD according to another MDD.

	int minimumVertexCover(const vector<int>& CG);
	int minimumVertexCover(const vector<int>& CG, int old_mvc, int cols, int num_of_edges);
	bool KVertexCover(const vector<int>& CG, int num_of_CGnodes, int num_of_CGedges, int k, int cols);
	static int greedyMatching(const vector<int>& CG, int cols);
	int minimumWeightedVertexCover(const vector<int>& CG);
	int weightedVertexCover(const vector<int>& CG);
	int DPForWMVC(vector<int>& x, int i, int sum, const vector<int>& CG, const vector<int>& range, int& best_so_far);
	// int ILPForWMVC(const vector<int>& CG, const vector<int>& node_max_value) const;
	int MVConAllConflicts(CBSNode& curr);



    //Functions used in CHBP
    int computeClusterHeuristicAndBypass(CBSNode& curr, vector<bool>& excluded_agents, double time_limit);
    void filterCluster(CBSNode& node, int& cluster_heuristic, vector<bool>& excluded_agents);
    int getMaxConflictAgent(vector<bool>& processed_agents, vector<bool>& excluded_agents, vector<bool>& SG);
    void removeEdgesFromSG(int agent_id, vector<bool>& SG);
    result_type findClusterOrBypass(CBSNode& curr, int a_m, Path& bypass_path, vector<bool>& excluded_agents,vector<bool>& SG);
    bool findMutexBetweenMDDs( MDD* main_mdd, MDD* secondary_mdd, vector<tuple<int, int, int>>& main_mutex,   vector<tuple<int, int, int>>& secondary_mutex);
    bool mutexCheckingWithCache(CBSNode& node, int main_agent, int check_agent, const Path& main_path,bool& found_incompatible_nodes, bool& path_not_existed  );
    bool mutexChecking(CBSNode& node, int check_agent, const Path& main_path,bool& found_incompatible_nodes, bool& path_not_existed  );
    int solveClusterAgents(const unordered_set<int>& cluster_set,  const CBSNode& node, bool cardinal);
    void computeClusterHeuristic(CBSNode& node, const unordered_set<int>& cluster_set );
    void labelConflict2MDD(SoftMDD& main_mdd, int curr_agent);
    void getPathAndConflictAgentFromMDD(SoftMDD& main_mdd, Path& main_path, unordered_set<int>& CA);
    void copyMutexGraph(CBSNode& child, const CBSNode& parent) const;
    void copyCluster(CBSNode &child, const CBSNode &parent) const;

};