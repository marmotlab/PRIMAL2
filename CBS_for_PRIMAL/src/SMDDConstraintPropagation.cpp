#include "SMDDConstraintPropagation.h"



bool SMDDConstraintPropagation::has_fwd_mutex(smdd_edge_pair e)
{
    return fwd_mutexes.find({ e.first, e.second }) != fwd_mutexes.end() ;
//    ||
//           fwd_mutexes.find({ e.second, e.first }) != fwd_mutexes.end();
}

bool SMDDConstraintPropagation::has_fwd_mutex(SoftMDDNode* a, MDDNode* b)
{
    return has_fwd_mutex({{ a, nullptr }, { b, nullptr }});
}



void SMDDConstraintPropagation::add_fwd_edge_mutex(SoftMDDNode* node_a, SoftMDDNode* node_a_to,
                                                   MDDNode* node_b, MDDNode* node_b_to)
{
    if (has_fwd_mutex({{ node_a, node_a_to }, { node_b, node_b_to }}))
        return;
    fwd_mutexes.insert({{ node_a, node_a_to }, { node_b, node_b_to }});
}

void SMDDConstraintPropagation::add_fwd_node_mutex(SoftMDDNode* node_a, MDDNode* node_b)
{
    // TODO check
    if (has_fwd_mutex({{ node_a, nullptr }, { node_b, nullptr }}))
        return;
    fwd_mutexes.insert({{ node_a, nullptr }, { node_b, nullptr }});
}


bool SMDDConstraintPropagation::should_be_fwd_mutexed(SoftMDDNode* node_a, MDDNode* node_b)
{
    for (auto node_a_from: node_a->parents)
    {
        for (auto node_b_from: node_b->parents)
        {
            // either if node mutex or edge mutex
            if (has_fwd_mutex( node_a_from,node_b_from))
            {
                continue;
            }
            if (has_fwd_mutex({{ node_a_from, node_a }, { node_b_from, node_b }}))
            {
                continue;
            }
            return false;
        }
    }
    return true;
}

void SMDDConstraintPropagation::init_mutex()
{
//    MDDCache mdd_cache1(map_size);
//    MDDCache mdd_cache2(map_size);

    int num_level = std::min(mdd0->levels.size(), mdd1->levels.size());
    // node mutex
    for (int i = 0; i < num_level; i++)
    {
        // COMMENT unordered map can be changed to vector for efficiency
        smdd_cache1.collectMDDlevel(mdd0,i);
        for (MDDNode* it_1 : mdd1->levels[i])
        {
            if (smdd_cache1.is_set(it_1->location))
            {
                add_fwd_node_mutex(smdd_cache1.get(it_1->location), it_1);
            }
        }
    }
    // edge mutex

    normalMDDCache* loc2mddThisLvl = &mdd_cache1;
    normalMDDCache* loc2mddNextLvl = &mdd_cache2;
    loc2mddNextLvl->collectMDDlevel(mdd1,0);
    for (int i = 0; i < num_level - 1; i++)
    {
        normalMDDCache* tmp = loc2mddThisLvl;
        loc2mddThisLvl->collectMDDlevel(mdd1, i + 1);
        loc2mddThisLvl = loc2mddNextLvl;
        loc2mddNextLvl = tmp;
        for (auto& node_0 : mdd0->levels[i])
        {
            int loc_0 = node_0->location;

            if (!loc2mddNextLvl->is_set(loc_0))
            {
                continue;
            }
            MDDNode* node_1_to = loc2mddNextLvl->get(loc_0);

            for (auto node_0_to:node_0->children)
            {
                int loc_1 = node_0_to->location;
                if (!loc2mddThisLvl->is_set(loc_1))
                {
                    continue;
                }

                MDDNode* node_1 = loc2mddThisLvl->get(loc_1);
                for (auto ptr:node_1->children)
                {
                    if (ptr == node_1_to)
                    {
                        add_fwd_edge_mutex(node_0, node_0_to, node_1, node_1_to);
                    }
                }
            }
        }
    }
//    print_all_mutex();
//    bool a =0;
}

void SMDDConstraintPropagation::fwd_mutex_prop()
{
    std::vector<boost::unordered_set<smdd_edge_pair> > to_check(max(mdd0->levels.size(), mdd1->levels.size()));

    for (const auto & mutex: fwd_mutexes)
    {
        int l = mutex.first.first->level;
        to_check[l].insert(mutex);
    }
    // std::deque<edge_pair> open(fwd_mutexes.begin(), fwd_mutexes.end());

    for (int i = 0; i < to_check.size(); i ++)
    {
        for (auto & mutex: to_check[i])
        {
            if (is_smdd_edge_mutex(mutex))
            {
                auto node_to_1 = mutex.first.second;
                auto node_to_2 = mutex.second.second;

                if (has_fwd_mutex(node_to_1, node_to_2))
                {
                    continue;
                }

                if (!should_be_fwd_mutexed(node_to_1, node_to_2))
                {
                    continue;
                }
                auto new_mutex = std::make_pair(std::make_pair(node_to_1, nullptr),
                                                std::make_pair(node_to_2, nullptr));

                fwd_mutexes.insert(new_mutex);
                assert(i + 1 == node_to_1->level);
                to_check[i + 1].insert(new_mutex);
            }
            else
            {
                // Node mutex
                auto node_a = mutex.first.first;
                auto node_b = mutex.second.first;

                // Check their child

                for (auto node_a_ch: node_a->children)
                {
                    for (auto node_b_ch: node_b->children)
                    {
                        if (has_fwd_mutex(node_a_ch, node_b_ch))
                        {
                            continue;
                        }

                        if (!should_be_fwd_mutexed(node_a_ch, node_b_ch))
                        {
                            continue;
                        }
                        auto new_mutex = std::make_pair(std::make_pair(node_a_ch, nullptr),
                                                        std::make_pair(node_b_ch, nullptr));

                        fwd_mutexes.insert(new_mutex);
                        assert(i + 1 == node_a_ch->level);
                        to_check[i + 1].insert(new_mutex);
                    }
                }
            }
        }
    }
}