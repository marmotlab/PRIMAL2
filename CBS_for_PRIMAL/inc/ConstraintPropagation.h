// #include <vector>

#ifndef CONS_PROP_H
#define CONS_PROP_H

#include <boost/unordered_set.hpp>
#include "MDD.h"

typedef std::pair<MDDNode*, MDDNode*> node_pair;
typedef std::pair<node_pair, node_pair> edge_pair;

bool is_edge_mutex(edge_pair ep);

class MDDCache{
public:
    MDDCache(){}

    explicit MDDCache(unsigned id_count):cached_node(id_count),last_seen(id_count, 0), current_timestamp(1){}

    bool is_set(unsigned id)const{
        return last_seen[id] == current_timestamp;
    }

    void set(unsigned id, MDDNode* mdd_ptr){
        last_seen[id] = current_timestamp;
        cached_node[id] = mdd_ptr;
    }

    MDDNode* get(unsigned id){
//        if(is_set(id)){
        return cached_node[id];
//        }
//        return nullptr;
    }


    void collectMDDlevel(MDD* mdd, int i)
    {
        reset_all();
        for (auto& it_0 : mdd->levels[i])
        {
            int loc = it_0->location;
            set(loc,it_0);
        }
    }

    void reset_all(){
        ++current_timestamp;
        if(current_timestamp == 0){
            std::fill(last_seen.begin(), last_seen.end(), 0);
            current_timestamp = 1;
        }
    }

private:
    std::vector<MDDNode*> cached_node;
    std::vector<unsigned>last_seen;
    unsigned current_timestamp;
};


class ConstraintPropagation{
protected:
  // TODO ownership?
  // std::vector<MDD*> mdds;
    MDD* mdd0;
    MDD* mdd1;

	// check whether two nodes could be mutexed
	// return true if they are mutexed
	bool should_be_fwd_mutexed(MDDNode*, MDDNode*);

	bool should_be_fwd_mutexed(MDDNode* node_a, MDDNode* node_a_to,
							   MDDNode* node_b, MDDNode* node_b_to);

	bool should_be_bwd_mutexed(MDDNode*, MDDNode*);
	bool should_be_bwd_mutexed(MDDNode* node_a, MDDNode* node_a_to,
							   MDDNode* node_b, MDDNode* node_b_to);


	void add_bwd_node_mutex(MDDNode* node_a, MDDNode* node_b);

	void add_fwd_node_mutex(MDDNode* node_a, MDDNode* node_b);
	void add_fwd_edge_mutex(MDDNode* node_a, MDDNode* node_a_to,
							MDDNode* node_b, MDDNode* node_b_to);

	// boost::unordered_set<node_pair> node_cons;

	/*
	  A Mutex is in the form of <<node*, node*>, <node*, node*>>
	  if second node* in each pair are nullptr, then it is a node mutex

	 */

	// void fwd_mutex_prop_generalized_helper(MDD* mdd_0, MDD* mdd_1, int level);

public:
    ConstraintPropagation(){};
	ConstraintPropagation(MDD* mdd0, MDD* mdd1) :
			mdd0(mdd0), mdd1(mdd1) {}

	boost::unordered_set<edge_pair> fwd_mutexes;
	boost::unordered_set<edge_pair> bwd_mutexes;

    MDDCache mdd_cache1;
    MDDCache mdd_cache2;
    int map_size;

    void init(int _map_size){
        mdd_cache1 = MDDCache(_map_size);
        mdd_cache2 = MDDCache(_map_size);
        map_size = _map_size;
    }
    void set_MDD(MDD* _mdd0, MDD* _mdd1) {
        mdd0 = _mdd0;
        mdd1 = _mdd1;
        fwd_mutexes.clear();
        bwd_mutexes.clear();
    }

  void init_mutex();
  void fwd_mutex_prop();
  void get_mutex_node(vector<tuple<int, int, int>> &main_mutex,
                        vector<tuple<int, int, int>> &secondary_mutex);
  // void fwd_mutex_prop_generalized();

  void bwd_mutex_prop();

  bool has_mutex(edge_pair);
  bool has_mutex(MDDNode*, MDDNode*);

  bool has_fwd_mutex(edge_pair);
  bool has_fwd_mutex(MDDNode*, MDDNode*);

  // MDD 0 of level_0 and MDD 1 of level_1 mutexed at goal
  bool mutexed(int level_0, int level_1);
  bool feasible(int level_0, int level_1);
  int _feasible(int level_0, int level_1);

  // virtual bool semi_cardinal(int level, int loc);

  std::pair<std::vector<Constraint>, std::vector<Constraint>> generate_constraints(int, int);

  ~ConstraintPropagation(){};
};

#endif
