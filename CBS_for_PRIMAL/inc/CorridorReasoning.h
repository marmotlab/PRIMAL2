#pragma once

#include "ReservationTable.h"
#include "Instance.h"
#include "SingleAgentSolver.h"

enum corridor_strategy { NC, C, PC, STC, GC, DC };

class CorridorReasoning
{
public:
	double accumulated_runtime = 0;
	int num_pesudo_corridors = 0;
	CorridorReasoning(const vector<SingleAgentSolver*>& search_engines,
					  const vector<ConstraintTable>& initial_constraints) :
			search_engines(search_engines), initial_constraints(initial_constraints) {}

	shared_ptr<Conflict> run(const shared_ptr<Conflict>& conflict,
		const vector<Path*>& paths, const CBSNode& node);
	
	void setStrategy(corridor_strategy s)
	{
		strategy = s;
		/*if (s != corridor_strategy::NC)
		{
			lookupTable.resize(search_engines.size());
			for (size_t i = 0; i < search_engines.size(); i++)
			{
				lookupTable[i].resize(search_engines.size());
			}
		}*/
	}
	corridor_strategy getStrategy() const {return strategy; }
    string getName() const
    {
        switch (strategy)
        {
            case corridor_strategy::NC:
                return "NC";
            case corridor_strategy::C:
                return "C";
            case corridor_strategy::PC:
                return "PC";
            case corridor_strategy::STC:
                return "STC";
            case corridor_strategy::GC:
                return "GC";
            case corridor_strategy::DC:
                return "DC";
            default:
                break;
        }
        return "";
    }
private:
	corridor_strategy strategy;
	const vector<SingleAgentSolver*>& search_engines;
	const vector<ConstraintTable>& initial_constraints;
	vector<unordered_map<ConstraintsHasher,
					list<tuple<int, int, int, int> >, // location, "from location", timestep, timestep for bypass
					ConstraintsHasher::Hasher, 
					ConstraintsHasher::EqNode> > lookupTable;

	shared_ptr<Conflict> findCorridorConflict(const shared_ptr<Conflict>& conflict,
		const vector<Path*>& paths, const CBSNode& node);
	shared_ptr<Conflict> findCorridorTargetConflict(const shared_ptr<Conflict>& conflict,
		const vector<Path*>& paths, const CBSNode& node);
	shared_ptr<Conflict> findPseudoCorridorConflict(const shared_ptr<Conflict>& conflict,
		const vector<Path*>& paths, const CBSNode& node);

	vector<int> findCorridor(int loc1, int loc2); // return the nodes in the corridor

	int getEnteringTime(const std::vector<PathEntry>& path, const std::vector<PathEntry>& path2, int t);
	static int getCorridorLength(const std::vector<PathEntry>& path, int t_start, int loc_end, std::pair<int, int>& edge);
	pair<int, int> getTimeRanges(int a1, int a2, int endpoint1, int endpoint2,
		int from1, int from2, int lowerbound1, int lowerbound2, int corridor_length, const CBSNode& node);
	static bool blocked(const Path& path, const Constraint& constraint);

};


