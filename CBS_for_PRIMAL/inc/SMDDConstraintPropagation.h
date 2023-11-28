#pragma once
#include <boost/unordered_set.hpp>
#include "MDD.h"

typedef std::pair<SoftMDDNode*, SoftMDDNode*> smdd_node_pair;
typedef std::pair<MDDNode*, MDDNode*> mdd_node_pair;
typedef std::pair<smdd_node_pair, mdd_node_pair> smdd_edge_pair;

inline bool is_smdd_edge_mutex(smdd_edge_pair ep)
{
    return ep.first.second != nullptr;
}


class SoftMDDCache{
public:
    SoftMDDCache(){}

    explicit SoftMDDCache(unsigned id_count):cached_node(id_count),last_seen(id_count, 0), current_timestamp(1){}

    bool is_set(unsigned id)const{
        return last_seen[id] == current_timestamp;
    }

    void set(unsigned id, SoftMDDNode* mdd_ptr){
        last_seen[id] = current_timestamp;
        cached_node[id] = mdd_ptr;
    }

    SoftMDDNode* get(unsigned id){
//        if(is_set(id)){
        return cached_node[id];
//        }
//        return nullptr;
    }


    void collectMDDlevel(SoftMDD* mdd, int i)
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
    std::vector<SoftMDDNode*> cached_node;
    std::vector<unsigned>last_seen;
    unsigned current_timestamp;
};


class normalMDDCache{
public:
    normalMDDCache(){}

    explicit normalMDDCache(unsigned id_count):cached_node(id_count),last_seen(id_count, 0), current_timestamp(1){}

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


class SMDDConstraintPropagation{
protected:
    // TODO ownership?
    // std::vector<MDD*> mdds;
    SoftMDD* mdd0;
    MDD* mdd1;




    void add_fwd_node_mutex(SoftMDDNode* node_a, MDDNode* node_b);
    void add_fwd_edge_mutex(SoftMDDNode* node_a, SoftMDDNode* node_a_to,
                            MDDNode* node_b, MDDNode* node_b_to);

    bool should_be_fwd_mutexed(SoftMDDNode*, MDDNode*);



    // boost::unordered_set<node_pair> node_cons;

    /*
      A Mutex is in the form of <<node*, node*>, <node*, node*>>
      if second node* in each pair are nullptr, then it is a node mutex

     */

    // void fwd_mutex_prop_generalized_helper(MDD* mdd_0, MDD* mdd_1, int level);

public:
    SMDDConstraintPropagation(){};
    SMDDConstraintPropagation(SoftMDD* mdd0, MDD* mdd1) :
            mdd0(mdd0), mdd1(mdd1) { }

    boost::unordered_set<smdd_edge_pair> fwd_mutexes;
    boost::unordered_set<smdd_edge_pair> bwd_mutexes;

    SoftMDDCache smdd_cache1;


    normalMDDCache mdd_cache1;
    normalMDDCache mdd_cache2;

    int map_size;
    void init(int _map_size){
        smdd_cache1 = SoftMDDCache(_map_size);
        mdd_cache1 = normalMDDCache(_map_size);
        mdd_cache2 = normalMDDCache(_map_size);
        map_size = _map_size;
    }
    void set_MDD(SoftMDD* _mdd0, MDD* _mdd1) {
        mdd0 = _mdd0;
        mdd1 = _mdd1;
        fwd_mutexes.clear();
        bwd_mutexes.clear();
    }

    bool has_fwd_mutex(smdd_edge_pair);
    bool has_fwd_mutex(SoftMDDNode*, MDDNode*);

    void init_mutex();
    void fwd_mutex_prop();

    ~SMDDConstraintPropagation(){};
};
