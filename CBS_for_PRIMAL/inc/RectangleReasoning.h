#pragma once

#include "MDD.h"

// no rectangle, rectangle, rectangle with MDDs, generalized rectangle, disjoint rectangle
enum rectangle_strategy { NR, R, RM, GR, DR };

class RectangleReasoning
{
public:
	rectangle_strategy strategy = rectangle_strategy::NR;
	double accumulated_runtime = 0;

	explicit RectangleReasoning(const Instance& instance) : instance(instance) {}

	shared_ptr<Conflict> run(const vector<Path*>& paths, int timestep,
							 int a1, int a2, const MDD* mdd1, const MDD* mdd2);

	void printOverlapArea(const vector<Path*>& paths, int timestep,
		int a1, int a2, const MDD* mdd1, const MDD* mdd2);

	string getName() const
    {
	    switch (strategy)
        {
            case rectangle_strategy::NR:
                return "NR";
            case rectangle_strategy::R:
                return "R";
            case rectangle_strategy::RM:
                return "RM";
            case rectangle_strategy::GR:
                return "GR";
            case rectangle_strategy::DR:
                return "DR";
            default:
                break;
        }
        return "";
    }
private:
	const Instance& instance;

    shared_ptr<Conflict> findRectangleConflict(const vector<Path*>& paths, int timestep, int a1, int a2); // R: find rectangle only for entire paths
	shared_ptr<Conflict> findRectangleConflict(const vector<Path*>& paths, int timestep,
                                               int a1, int a2, const MDD* mdd1, const MDD* mdd2);  // RM: find rectangle for path segments
	shared_ptr<Conflict> findGenerealizedRectangleConflict(const vector<Path*>& paths, int timestep,
                                                           int a1, int a2, const MDD* mdd1, const MDD* mdd2);

	//Identify rectangle conflicts
	static bool isRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
		    const pair<int, int>& g1, const pair<int, int>& g2, int g1_t, int g2_t);// for CR and R
	static bool isRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
	        const pair<int, int>& g1, const pair<int, int>& g2) ;// for RM

	//Classify rectangle conflicts
	static int classifyRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
		    const pair<int, int>& g1, const pair<int, int>& g2);// for CR and R
	static int classifyRectangleConflict(const pair<int, int>& s1, const pair<int, int>& s2,
	        const pair<int, int>& g1, const pair<int, int>& g2, const pair<int, int>& Rg);// for RM

	//Add barrier constraints
    void addBarrierConstraints(int a1, int a2, const pair<int, int>& Rs, const pair<int, int>& Rg,
            const pair<int, int>& s1, const pair<int, int>& s2, int Rg_t,
            list<Constraint>& constraint1, list<Constraint>& constraint2); // for CR and R
    bool addModifiedBarrierConstraints(int a1, int a2, const pair<int, int>& Rs, const pair<int, int>& Rg,
                                       const pair<int, int>& s1, const pair<int, int>& s2, int Rg_t,
                                       const MDD* mdd1, const MDD* mdd2,
                                       list<Constraint>& constraint1, list<Constraint>& constraint2); // for RM
    // add a horizontal modified barrier constraint
    bool addModifiedHorizontalBarrierConstraint(int agent, const MDD* mdd, int x,
                                                int Ri_y, int Rg_y, int Rg_t, list<Constraint>& constraints);
    // add a vertical modified barrier constraint
    bool addModifiedVerticalBarrierConstraint(int agent, const MDD* mdd, int y,
                                              int Ri_x, int Rg_x, int Rg_t, list<Constraint>& constraints);

	 //Compute rectangle corners
	static pair<int, int> getRg(const pair<int, int>& s1, const pair<int, int>& g1, const pair<int, int>& g2);
	static pair<int, int> getRs(const pair<int, int>& s1, const pair<int, int>& s2, const pair<int, int>& g1);

	// tools to validate constraints
	static bool blocked(int location, const list<Constraint>& constraints);
	bool blocked(const Path& path, const list<Constraint>& constraints);
	static bool blocked(const MDD& mdd, const list<Constraint>& constraints);
	static bool traverse(const Path& path, int loc, int t);


    //Compute start and goal candidates for RM
    list<int> getStartCandidates(const Path& path, int timestep);
    list<int> getGoalCandidates(const Path& path, int timestep);


    ////////////////////////////////////////////////////////////////////////////
    /// Tools for GR
    //////////////////////////////////////////////////////////////////////////////
	static void projectMDD2Map(vector<MDDNode*>& mapping, const MDD* mdd, MDDNode* multiple_visits);
	void findOverlapArea(int conflict_location, vector<bool>& overlap_area,
		const vector<MDDNode*>& visit_times1, const vector<MDDNode*>& visit_times2,
		set<pair<int, int> >& entry1, set<pair<int, int> >& entry2, list<int>& exit1, list<int>& exit2, 
		pair<int, int>& Rs, pair<int, int>& Rg, const MDDNode* multiple_visits) const;
	bool scanPerimeter(int& stage, const vector<bool>& overlap_area, 
		const pair<int, int>& Rs, const pair<int, int>& Rg,
		pair<int, int>& R1, pair<int, int>& R2,
		set<pair<int, int> >& entry1, set<pair<int, int> >& entry2) const;
	bool checkHoles(set<pair<int, int> >& entry1, set<pair<int, int> >& entry2,
		const vector<bool>& overlap_area, const pair<int, int>& Rs) const;
	list<Constraint> generateConstraint(const vector<bool>& overlap_area,
		const vector<MDDNode*>& visit_times, int agent, const pair<int, int>& from, const pair<int, int>& to) const;
};

