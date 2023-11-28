#pragma once

#include"common.h"


// Currently only works for undirected unweighted 4-neighbor grids
class Instance 
{
public:
	int num_of_cols = 0;
	int num_of_rows = 0;
	int map_size = 0;
	vector<int> start_directions;

	Instance() = default;
	Instance(const string& map_fname, const string& agent_fname, 
		int num_of_agents = 0, const string& agent_indices = "",
		int num_of_rows = 0, int num_of_cols = 0, int num_of_obstacles = 0, int warehouse_width = 0);
	Instance(list<bool> map, list<int> startL, list<int> startD, list<int> goalL, int cols, int rows, int agents);

	void printAgents() const;
	void printMap() const;


	inline bool isObstacle(int loc) const { return my_map[loc]; }
	inline bool validMove(int curr, int next) const
	{
		if (next < 0 || next >= map_size)
			return false;
		if (my_map[next])
			return false;
		return getManhattanDistance(curr, next) < 2;
	}

	list<int> getNeighbors(int curr) const;
	list<pair<int,int>> getNeighbors(int curr, int direction) const;
    inline int linearizeCoordinate(int row, int col) const { return ( this->num_of_cols * row + col); }
	inline int linearizeCoordinate(const pair<int, int>& cell) const { return linearizeCoordinate(cell.first, cell.second); }
	inline int getRowCoordinate(int id) const { return id / this->num_of_cols; }
	inline int getColCoordinate(int id) const { return id % this->num_of_cols; }
	inline pair<int, int> getCoordinate(int id) const { return make_pair(id / this->num_of_cols, id % this->num_of_cols); }
	inline int getCols() const { return num_of_cols; }

	inline int getManhattanDistance(int loc1, int loc2) const
	{
		int loc1_x = getRowCoordinate(loc1);
		int loc1_y = getColCoordinate(loc1);
		int loc2_x = getRowCoordinate(loc2);
		int loc2_y = getColCoordinate(loc2);
		return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y);
	}

	static inline int getManhattanDistance(const pair<int, int>& loc1, const pair<int, int>& loc2)
	{
		return abs(loc1.first - loc2.first) + abs(loc1.second - loc2.second);
	}
	int walkCounterClockwise(int from, int to) const
	{
		assert(validMove(from, to));
		int dir = turnLeft(to - from);
		while (!validMove(from, from + dir))
			dir = turnLeft(dir);
		return from + dir;
	}
	inline int turnLeft(int dir) const
	{
		if (dir ==  1) 
			return -num_of_cols;
		else if (dir == -num_of_cols)
			return - 1;
		else if (dir == -1)
			return num_of_cols;
		else if (dir == num_of_cols)
			return 1;
		else
			return 0;
	 }
	inline int turnRight(int dir) const
	{
		if (dir == 1)
			return num_of_cols;
		else if (dir == num_of_cols)
			return -1;
		else if (dir == -1)
			return -num_of_cols;
		else if (dir == -num_of_cols)
			return 1;
		else
			return 0;
	}
	int getDegree(int loc) const // return the number of neighbors
	{
		assert(loc >= 0 && loc < map_size && !my_map[loc]);
		int degree = 0;
		if (0 < loc - num_of_cols && !my_map[loc - num_of_cols])
			degree++;
		if (loc + num_of_cols < map_size && !my_map[loc + num_of_cols])
			degree++;
		if (loc % num_of_cols > 0 && !my_map[loc - 1])
			degree++;
		if (loc % num_of_cols < num_of_cols - 1 && !my_map[loc + 1])
			degree++;
		return degree;
	}

	int getDefaultNumberOfAgents() const { return num_of_agents; }

private:
	  // int moves_offset[MOVE_COUNT];
	  vector<bool> my_map;
	  string map_fname;
	  string agent_fname;
	  string agent_indices;
	  int num_of_agents = 0;
	  vector<int> start_locations;
	  vector<int> goal_locations;

	bool loadMap();
	
	void saveMap() const;

	bool loadAgents();
	void saveAgents() const;

	  void generateConnectedRandomGrid(int rows, int cols, int obstacles); // initialize new [rows x cols] map with random obstacles
	  void generateRandomAgents(int warehouse_width);
	  bool addObstacle(int obstacle); // add this obstacle only if the map is still connected
	  bool isConnected(int start, int goal) const; // run BFS to find a path between start and goal, return true if a path exists.

	int randomWalk(int loc, int steps) const;

	// Class  SingleAgentSolver can access private members of Node
	friend class SingleAgentSolver;
};

