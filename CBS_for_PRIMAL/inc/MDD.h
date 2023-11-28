#pragma once

#include "SingleAgentSolver.h"


class MDDNode
{
public:
	MDDNode(int currloc, MDDNode* parent)
	{
		location = currloc; 
		if (parent == nullptr)
			level = 0;
		else
		{
			level = parent->level + 1;
			parents.push_back(parent);
		}
	}
	int location;
	int level;
	int cost=0; // minimum cost of path traversing this MDD node

	bool operator==(const MDDNode& node) const
	{
		return (this->location == node.location) && (this->level == node.level);
	}


	list<MDDNode*> children;
	list<MDDNode*> parents;

    int position = 0;
};

class MDD
{
private:
	const SingleAgentSolver* solver;

public:
	vector<list<MDDNode*>> levels;

	bool buildMDD(const ConstraintTable& ct,
				  int num_of_levels, const SingleAgentSolver* solver);
	// bool buildMDD(const std::vector <std::list< std::pair<int, int> > >& constraints, int numOfLevels,
	// 	int start_location, const int* moves_offset, const std::vector<int>& my_heuristic, int map_size, int num_col);
	void printNodes() const;
	MDDNode* find(int location, int level) const;
	void deleteNode(MDDNode* node);
	void clear();
	// bool isConstrained(int curr_id, int next_id, int next_timestep, const std::vector< std::list< std::pair<int, int> > >& cons) const;

	void increaseBy(const ConstraintTable& ct, int dLevel, SingleAgentSolver* solver);
	MDDNode* goalAt(int level);
    size_t get_total_size() const{
        size_t size = 0;
        for(auto& level: levels){
            size += level.size();
        }
        return size;
    }

	MDD() = default;
	MDD(const MDD& cpy);
	~MDD();
};

std::ostream& operator<<(std::ostream& os, const MDD& mdd);

class SyncMDDNode
{
public:
	SyncMDDNode(int currloc, SyncMDDNode* parent)
	{
		location = currloc;
		if (parent != nullptr)
		{
			//level = parent->level + 1;
			parents.push_back(parent);
		}
		//parent = NULL;
	}
	int location;
	//int level;

	bool operator==(const SyncMDDNode& node) const
	{
		return (this->location == node.location);
	}


	list<SyncMDDNode*> children;
	list<SyncMDDNode*> parents;
	list<const MDDNode*> coexistingNodesFromOtherMdds;

};


class SyncMDD
{
public:
	vector<list<SyncMDDNode*>> levels;

	SyncMDDNode* find(int location, int level) const;
	void deleteNode(SyncMDDNode* node, int level);
	void clear();

	explicit SyncMDD(const MDD& cpy);
	~SyncMDD();
};

class MDDTable
{
public:
	double accumulated_runtime = 0;  // runtime of building MDDs
	uint64_t num_released_mdds = 0; // number of released MDDs ( to save memory)

	MDDTable(const vector<ConstraintTable>& initial_constraints,
			 const vector<SingleAgentSolver*>& search_engines) :
			initial_constraints(initial_constraints), search_engines(search_engines) {}

	void init(int number_of_agents)
	{
		lookupTable.resize(number_of_agents);
	}
	~MDDTable() { clear(); }


	MDD* getMDD(CBSNode& node, int agent, size_t mdd_levels);
	void findSingletons(CBSNode& node, int agent, Path& path);
	double getAverageWidth(CBSNode& node, int agent, size_t mdd_levels);
	void clear();
private:
	int max_num_of_mdds = 10000;

	vector<unordered_map<ConstraintsHasher, MDD*,
			ConstraintsHasher::Hasher, ConstraintsHasher::EqNode>> lookupTable;

	const vector<ConstraintTable>& initial_constraints;
	const vector<SingleAgentSolver*>& search_engines;
	void releaseMDDMemory(int id);
};

unordered_map<int, MDDNode*> collectMDDlevel(MDD* mdd, int i);

class SoftMDDNode
{
public:
    int location;
    int level;
    SoftMDDNode(int currloc, int curlevel)
    {
        location = currloc;
        level = curlevel;
    }
    SoftMDDNode() = default;
    bool operator==(const SoftMDDNode& node) const
    {
        return (this->location == node.location);
    }

    vector<SoftMDDNode*> children;
    vector<SoftMDDNode*> parents;

    int num_of_conflict = 0;
    int min_total_conflict =0;

    vector<int> node_conflict_agent; // for node

    SoftMDDNode* pre_node = nullptr;
};




class SoftMDD
{
    // quick soft copy of MDD, use vector-based implementation.
    // completely get ride of list;
private:
    vector<SoftMDDNode> node_pool;
    int np_counter;
    int increase_size;

    SoftMDDNode* allocate_node(int location, int level){
        np_counter++;
//        if(np_counter == node_pool.size()){
//            node_pool.resize(node_pool.size() + increase_size);
//        }
        node_pool[np_counter].children.clear(); node_pool[np_counter].parents.clear();
        node_pool[np_counter].level = level; node_pool[np_counter].location = location;
        node_pool[np_counter].node_conflict_agent.clear();
        node_pool[np_counter].pre_node = nullptr; node_pool[np_counter].num_of_conflict  = 0;
        return &node_pool[np_counter];
    }

    void reset(){
        np_counter = 0;
    }


public:
    vector<vector<SoftMDDNode*>> levels;
    bool is_labeled = false;

    vector<boost::unordered_map<size_t, vector<int>>> edge_conflict_agent;
//    vector<vector<int>> edge_conflict_agent; // for parent edge.
    SoftMDDNode* find(int location, int level);

//    void print();
    void deleteNode(SoftMDDNode* node);
    void deleteEdge(SoftMDDNode* from, SoftMDDNode* to);
    void deleteNode(SoftMDDNode* node, const vector<PathEntry>& main_path, bool& path_not_existed);
    void deleteEdge(SoftMDDNode* from, SoftMDDNode* to, const vector<PathEntry>& main_path, bool& path_not_existed);

    void clear();
    void fastCopyMDD(const MDD& cpy);
    explicit SoftMDD(int pool_size){
        increase_size = pool_size;
        node_pool =vector<SoftMDDNode>(pool_size);
        np_counter = 0;
    }

    void resize(size_t MDD_size){
        if(MDD_size > node_pool.size()){
            node_pool.clear();
            node_pool.resize((MDD_size / increase_size +1) * increase_size);
        }
    }

    void init(int pool_size){
        increase_size = 10000;
        node_pool = vector<SoftMDDNode>(pool_size);
    }

    SoftMDD(){
        increase_size =0;
        np_counter =0;
    }
    ~SoftMDD(){
        clear();
    }
};
