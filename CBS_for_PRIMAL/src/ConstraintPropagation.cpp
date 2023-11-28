#include <algorithm>
#include <deque>
#include <stack>
#include "ConstraintPropagation.h"

#include <utility>
#include <iostream>

// is a mutex edge mutex.
bool is_edge_mutex(edge_pair ep)
{
	return ep.first.second != nullptr;
}

bool ConstraintPropagation::should_be_bwd_mutexed(MDDNode* node_a, MDDNode* node_b)
{
	for (auto node_a_to: node_a->children)
	{
		for (auto node_b_to: node_b->children)
		{
			// either if node mutex or edge mutex
			if (has_mutex(node_b_to, node_a_to))
			{
				continue;
			}
			if (has_mutex({{ node_a, node_a_to }, { node_b, node_b_to }}))
			{
				continue;
			}
			return false;
		}
	}
	return true;
}

bool ConstraintPropagation::should_be_fwd_mutexed(MDDNode* node_a, MDDNode* node_b)
{
	for (auto node_a_from: node_a->parents)
	{
		for (auto node_b_from: node_b->parents)
		{
			// either if node mutex or edge mutex
			if (has_fwd_mutex(node_b_from, node_a_from))
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

bool ConstraintPropagation::has_mutex(edge_pair e)
{
	return (bwd_mutexes.find({ e.first, e.second }) != bwd_mutexes.end() ||
		   bwd_mutexes.find({ e.second, e.first }) != bwd_mutexes.end()) ||
		   has_fwd_mutex(e);
}

bool ConstraintPropagation::has_mutex(MDDNode* a, MDDNode* b)
{
	return has_mutex({{ a, nullptr }, { b, nullptr }});
}

bool ConstraintPropagation::has_fwd_mutex(edge_pair e)
{
	return fwd_mutexes.find({ e.first, e.second }) != fwd_mutexes.end() ||
	       fwd_mutexes.find({ e.second, e.first }) != fwd_mutexes.end();
}

bool ConstraintPropagation::has_fwd_mutex(MDDNode* a, MDDNode* b)
{
	return has_fwd_mutex({{ a, nullptr }, { b, nullptr }});
}

void ConstraintPropagation::add_bwd_node_mutex(MDDNode* node_a, MDDNode* node_b)
{
	// TODO check
	if (has_mutex({{ node_a, nullptr }, { node_b, nullptr }}))
		return;
	bwd_mutexes.insert({{ node_a, nullptr }, { node_b, nullptr }});
}

void ConstraintPropagation::add_fwd_edge_mutex(MDDNode* node_a, MDDNode* node_a_to,
											   MDDNode* node_b, MDDNode* node_b_to)
{
	if (has_fwd_mutex({{ node_a, node_a_to }, { node_b, node_b_to }}))
		return;
	fwd_mutexes.insert({{ node_a, node_a_to }, { node_b, node_b_to }});
}

void ConstraintPropagation::add_fwd_node_mutex(MDDNode* node_a, MDDNode* node_b)
{
	// TODO check
	if (has_fwd_mutex({{ node_a, nullptr }, { node_b, nullptr }}))
		return;
	fwd_mutexes.insert({{ node_a, nullptr }, { node_b, nullptr }});
}

//
//void ConstraintPropagation::init_mutex()
//{
//	int num_level = std::min(mdd0->levels.size(), mdd1->levels.size());
//	// node mutex
//	for (int i = 0; i < num_level; i++)
//	{
//		// COMMENT unordered map can be changed to vector for efficiency
//		auto loc2mdd = collectMDDlevel(mdd0, i);
//		for (MDDNode* it_1 : mdd1->levels[i])
//		{
//			if (loc2mdd.find(it_1->location) != loc2mdd.end())
//			{
//				add_fwd_node_mutex(loc2mdd[it_1->location], it_1);
//			}
//		}
//	}
//	// edge mutex
//
//	unordered_map<int, MDDNode*> loc2mddThisLvl;
//	unordered_map<int, MDDNode*> loc2mddNextLvl = collectMDDlevel(mdd1, 0);
//
//	for (int i = 0; i < num_level - 1; i++)
//	{
//		loc2mddThisLvl = loc2mddNextLvl;
//		loc2mddNextLvl = collectMDDlevel(mdd1, i + 1);
//		for (auto& node_0 : mdd0->levels[i])
//		{
//			int loc_0 = node_0->location;
//			if (loc2mddNextLvl.find(loc_0) == loc2mddNextLvl.end())
//			{
//				continue;
//			}
//			MDDNode* node_1_to = loc2mddNextLvl[loc_0];
//
//			for (auto node_0_to:node_0->children)
//			{
//				int loc_1 = node_0_to->location;
//				if (loc2mddThisLvl.find(loc_1) == loc2mddThisLvl.end())
//				{
//					continue;
//				}
//
//				MDDNode* node_1 = loc2mddThisLvl[loc_1];
//				for (auto ptr:node_1->children)
//				{
//					if (ptr == node_1_to)
//					{
//						add_fwd_edge_mutex(node_0, node_0_to, node_1, node_1_to);
//					}
//				}
//			}
//		}
//	}
//}


void ConstraintPropagation::init_mutex()
{

    int num_level = std::min(mdd0->levels.size(), mdd1->levels.size());
    // node mutex
    for (int i = 0; i < num_level; i++)
    {
        mdd_cache1.collectMDDlevel(mdd0,i);
        for (MDDNode* it_1 : mdd1->levels[i])
        {
            if (mdd_cache1.is_set(it_1->location))
            {
                add_fwd_node_mutex(mdd_cache1.get(it_1->location), it_1);
            }
        }
    }
    // edge mutex

    MDDCache* loc2mddThisLvl = &mdd_cache1;
    MDDCache* loc2mddNextLvl = &mdd_cache2;
    loc2mddNextLvl->collectMDDlevel(mdd1,0);
    for (int i = 0; i < num_level - 1; i++)
    {
        MDDCache* tmp = loc2mddThisLvl;
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


void ConstraintPropagation::fwd_mutex_prop()
{
	std::vector<boost::unordered_set<edge_pair> > to_check(max(mdd0->levels.size(), mdd1->levels.size()));

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
			if (is_edge_mutex(mutex))
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

void ConstraintPropagation::bwd_mutex_prop()
{
	std::deque<edge_pair> open(fwd_mutexes.begin(), fwd_mutexes.end());

	while (!open.empty())
	{
		auto mutex = open.front();

		open.pop_front();

		if (is_edge_mutex(mutex))
		{
			auto node_from_1 = mutex.first.first;
			auto node_from_2 = mutex.second.first;

			if (has_mutex(node_from_1, node_from_2))
			{
				continue;
			}

			if (!should_be_bwd_mutexed(node_from_1, node_from_2))
			{
				continue;
			}
			auto new_mutex = std::make_pair(std::make_pair(node_from_1, nullptr),
											std::make_pair(node_from_2, nullptr));

			bwd_mutexes.insert(new_mutex);
			open.push_back(new_mutex);
		}
		else
		{
			// Node mutex
			auto node_a = mutex.first.first;
			auto node_b = mutex.second.first;

			// Check their child
			for (auto node_a_pa: node_a->parents)
			{
				for (auto node_b_pa: node_b->parents)
				{

					if (has_mutex(node_a_pa, node_b_pa))
					{
						continue;
					}

					if (!should_be_bwd_mutexed(node_a_pa, node_b_pa))
					{
						continue;
					}
					auto new_mutex = std::make_pair(std::make_pair(node_a_pa, nullptr),
													std::make_pair(node_b_pa, nullptr));

					bwd_mutexes.insert(new_mutex);
					open.push_back(new_mutex);
				}
			}
		}
	}
}

bool ConstraintPropagation::mutexed(int level_0, int level_1){
  // level_0 < mdd_s->levels, the index of the level
  MDD* mdd_s = mdd0;
  MDD* mdd_l = mdd1;
  if (level_0 > level_1){
    std::swap(level_0, level_1);
    mdd_s = mdd1;
    mdd_l = mdd0;
  }

  if (level_0 > mdd_s->levels.size()){
    std::cout << "ERROR!" << std::endl;
  }
  if (level_1 > mdd_l->levels.size()){
    std::cout << "ERROR!" << std::endl;
  }

  auto goal_ptr_i = mdd_s->goalAt(level_0);

  std::stack<MDDNode*> dfs_stack;

  for (auto& it:mdd_l->levels[level_0]){
    if (it->cost <= level_1 && !has_fwd_mutex(goal_ptr_i, it)){
      return false;
    }
  }
  return true;
}


int ConstraintPropagation::_feasible(int level_0, int level_1){
  MDD* mdd_s = mdd0;
  MDD* mdd_l = mdd1;
  if (level_0 > level_1){
    std::swap(level_0, level_1);
    mdd_s = mdd1;
    mdd_l = mdd0;
  }

  if (level_0 > mdd_s->levels.size()){
    std::cout << "ERROR!" << std::endl;
  }
  if (level_1 > mdd_l->levels.size()){
    std::cout << "ERROR!" << std::endl;
  }

  auto goal_ptr_i = mdd_s->goalAt(level_0);

  std::stack<MDDNode*> dfs_stack;

  for (auto& it:mdd_l->levels[level_0]){
    if (it->cost <= level_1 && !has_fwd_mutex(goal_ptr_i, it)){
      //return false;
      dfs_stack.push(it);
    }
  }
  if (dfs_stack.empty()){
    return -1;
  }

  // Using dfs to see is there any path lead to goal


  MDDNode* goal_ptr_j = mdd_l->goalAt(level_1);

  int not_allowed_loc = goal_ptr_i->location;

  boost::unordered_set<MDDNode*> closed;

  while (!dfs_stack.empty()){
    auto ptr = dfs_stack.top();
    dfs_stack.pop();

    if (ptr == goal_ptr_j){
      // find a single path
      return 1;
    }

    if (closed.find(ptr) != closed.end()){
      continue;
    }
    closed.insert(ptr);

    for (auto child_ptr: ptr->children){
      if (closed.find(child_ptr) != closed.end()){
        continue;
      }

      if (child_ptr->location == not_allowed_loc){
        continue;
      }

      dfs_stack.push(child_ptr);
    }
  }
  return -2;
}

bool ConstraintPropagation::feasible(int level_0, int level_1)
{
	return _feasible(level_0, level_1) < 0;
}

std::pair<std::vector<Constraint>, std::vector<Constraint>> ConstraintPropagation::generate_constraints(int level_0, int level_1){
  MDD* mdd_s = mdd0;
  MDD* mdd_l = mdd1;
  bool reversed = false;
  if (level_0 > level_1){
    std::swap(level_0, level_1);
    mdd_s = mdd1;
    mdd_l = mdd0;
    reversed = true;
  }

  // level_0 <= level_1

  auto goal_ptr_i = mdd_s->goalAt(level_0);

  std::vector<MDDNode*> mutexed;
  std::vector<MDDNode*> non_mutexed;
  for (auto& it:mdd_l->levels[level_0]){
    if (it->cost <= level_1){
      if (! has_fwd_mutex(goal_ptr_i, it)){
        non_mutexed.push_back(it);
        // return {{}, {}};
      }else{
        mutexed.push_back(it);
      }
    }
  }

  if (!non_mutexed.empty()){
    // AC
    int l = level_0;
    // std::vector<std::pair<int, int>> cons_vec_0;
    // std::vector<std::pair<int, int>> cons_vec_1;
    boost::unordered_set<std::pair<int, int>> cons_set_1;
    boost::unordered_set<MDDNode*> level_i({goal_ptr_i});
    // boost::unordered_set<MDDNode*> level_j(non_mutexed.begin(), non_mutexed.end());
    boost::unordered_set<MDDNode*> level_j;

    for (auto& it:mdd_l->levels[level_0]){
      if (it->cost <= level_1){
        level_j.insert(it);
      }
    }

    for (l = level_0; l >= 0; l--){
      for (auto ptr_i:level_i){
        bool non_all_mutexed=false;
        for (auto ptr_j:level_j){
          if (!has_fwd_mutex(ptr_i, ptr_j)){
            non_all_mutexed=true;
            break;
          }
        }
      }
      for (auto ptr_j:level_j){
        bool non_all_mutexed=false;
        for (auto ptr_i:level_i){
          if (!has_fwd_mutex(ptr_i, ptr_j)){
            non_all_mutexed=true;
            break;
          }
        }
        if (!non_all_mutexed){
          cons_set_1.insert({l, ptr_j->location});
        }
      }

      // go to prev levels
      boost::unordered_set<MDDNode*> level_i_prev;
      boost::unordered_set<MDDNode*> level_j_prev;
      for (auto ptr_i:level_i){
        for (auto parent_ptr:ptr_i->parents){
          level_i_prev.insert(parent_ptr);
        }
      }
      for (auto ptr_j:level_j){
        for (auto parent_ptr:ptr_j->parents){
          level_j_prev.insert(parent_ptr);
        }
      }
      level_i = level_i_prev;
      level_j = level_j_prev;
    }

    // for level_j, we still need to consider consflict after i reach goal;
    MDDNode* goal_ptr_j = mdd_l->goalAt(level_1);

    int not_allowed_loc = goal_ptr_i->location;

    boost::unordered_set<MDDNode*> closed;
    std::deque<MDDNode*> dfs_stack(non_mutexed.begin(), non_mutexed.end());

    while (!dfs_stack.empty()){
      auto ptr = dfs_stack.front();
      dfs_stack.pop_front();

      if (ptr == goal_ptr_j){
        // find a single path
        std::cout << "ERROR: Non mutexed pair of MDDs" << std::endl;
        return {{}, {}};
      }

      if (closed.find(ptr) != closed.end()){
        continue;
      }
      closed.insert(ptr);

      for (auto child_ptr: ptr->children){
        if (closed.find(child_ptr) != closed.end()){
          continue;
        }
        if (child_ptr->location == not_allowed_loc){
          cons_set_1.insert({child_ptr->level, child_ptr->location});
          continue;
        }
        dfs_stack.push_front(child_ptr);
      }
    }

    Constraint length_con(0, goal_ptr_i->location, -1, level_0, constraint_type::GLENGTH);
    // std::pair<int, int> length_con = {-1, level_0};

    std::vector<Constraint>cons_vec_1;
    for (auto& it:cons_set_1){
      cons_vec_1.push_back(Constraint(1, it.second, -1, it.first, constraint_type::VERTEX));
    }
    // std::vector<std::pair<int, int>> cons_vec_1(cons_set_1.begin(), cons_set_1.end());
    // cons_vec_1.push_back(Constraint(0, goal_ptr_i->location,  -1, level_0 - 1, constraint_type::LEQLENGTH));


    if (reversed){
      return {cons_vec_1, {length_con}};
    }
    return {{length_con}, cons_vec_1};
  }


  // goal nodes are mutexed
  boost::unordered_set<MDDNode*> cons_0, cons_1;
  boost::unordered_set<MDDNode*> blue_0, blue_1;

  for (int lvl = 0; lvl <= level_0; lvl ++){
    std::vector<MDDNode*> nodes_i, nodes_j;
    for (auto& it:mdd_s->levels[lvl]){
      if (it->cost <= level_0){
        nodes_i.push_back(it);
      }
    }
    for (auto& it:mdd_l->levels[lvl]){
      if (it->cost <= level_1){
        nodes_j.push_back(it);
      }
    }
    for (auto it_i:nodes_i){
      bool all_mutexed = true;
      for (auto it_j:nodes_j){
        if (!has_fwd_mutex(it_i, it_j)){
          all_mutexed = false;
          break;
        }
      }
      if (all_mutexed){
        blue_0.insert(it_i);
        bool has_non_blue_parent = false;
        for (auto ptr:it_i -> parents){
          if (blue_0.find(ptr) == blue_0.end()){
            has_non_blue_parent = true;
            break;
          }
        }

        if (has_non_blue_parent){
          cons_0.insert(it_i);
        }
      }
    }
    for (auto it_j:nodes_j){
      bool all_mutexed = true;
      for (auto it_i:nodes_i){
        if (!has_fwd_mutex(it_i, it_j)){
          all_mutexed = false;
          break;
        }
      }

      if (all_mutexed){

        blue_1.insert(it_j);
        bool has_non_blue_parent = false;
        for (auto ptr:it_j -> parents){
          if (blue_1.find(ptr) == blue_1.end()){
            has_non_blue_parent = true;
            break;
          }
        }

        if (has_non_blue_parent){
          cons_1.insert(it_j);
        }
      }
    }
  }

	std::vector<Constraint> cons_vec_0;
	std::vector<Constraint> cons_vec_1;

	for (auto& it:cons_0)
		cons_vec_0.push_back({0, it->location, -1, it->level, constraint_type::VERTEX});
	for (auto& it:cons_1)
		cons_vec_1.push_back({1, it->location, -1, it->level, constraint_type::VERTEX});
	if (reversed)
		return {cons_vec_1, cons_vec_0};
	return {cons_vec_0, cons_vec_1};
}

void ConstraintPropagation::get_mutex_node(vector<tuple<int, int, int>> &main_mutex,
                                           vector<tuple<int, int, int>> &secondary_mutex){
//   tried not add dominated mutex node, but not much improvement
//    mdd0->resetDominance();
//    mdd1->resetDominance();
    int mdd_size = min(mdd1->levels.size(),mdd0->levels.size());
    for (int i = 1; i < mdd_size; i++) {

        for (auto &curr_ptr_i: mdd0->levels[i]) {
//            if(curr_ptr_i->is_dominate) continue;
            bool is_mutex_node = true;
            for (auto &it: mdd1->levels[i]) {
                if (!has_fwd_mutex(curr_ptr_i, it)) {
                    is_mutex_node = false;
                    break;
                }
            }
            if (is_mutex_node) {
//                curr_ptr_i->is_dominate = true;
//                for(auto& np : curr_ptr_i->children) mdd0->labelFwdDominance(np);
                main_mutex.emplace_back(make_tuple(i, curr_ptr_i->location, -1));
//                    std::cout<<"Found mutex"<< i << ", "<<curr_ptr_i->location << std::endl;
            }

            if(mdd1->levels[i].size() == 1 && mdd1->levels[i].front()->parents.size() == 1){
                if(curr_ptr_i ->location == mdd1->levels[i].front()->parents.front()->location){
                    for(auto& parent_ptr_i: curr_ptr_i->parents){
                        if(parent_ptr_i->location == mdd1->levels[i].front()->location){
                            main_mutex.emplace_back(make_tuple(i-1, parent_ptr_i->location, curr_ptr_i->location));
                        }
                    }
                }
            }
        }


        for (auto &curr_ptr_i:  mdd1->levels[i]) {
//            if(curr_ptr_i->is_dominate) continue;
            bool is_mutex_node = true;
            for (auto &it: mdd0->levels[i]) {
                if (!has_fwd_mutex(curr_ptr_i, it)) {
                    is_mutex_node = false;
                    break;
                }
            }
            if (is_mutex_node) {
//                curr_ptr_i->is_dominate = true;
//                for(auto& np : curr_ptr_i->children) mdd1->labelFwdDominance(np);
                secondary_mutex.emplace_back(make_tuple(i, curr_ptr_i->location, -1));
//                    std::cout<<"Found mutex"<< i << ", "<<curr_ptr_i->location << std::endl;
            }
            if(mdd0->levels[i].size() == 1 && mdd0->levels[i].front()->parents.size() == 1){
                // singleton edges
                if(curr_ptr_i ->location == mdd0->levels[i].front()->parents.front()->location){
                    for(auto& parent_ptr_i: curr_ptr_i->parents){
                        if(parent_ptr_i->location == mdd0->levels[i].front()->location){
                            secondary_mutex.emplace_back(make_tuple(i-1, parent_ptr_i->location, curr_ptr_i->location));
                        }
                    }
                }
            }
        }
    }

}
