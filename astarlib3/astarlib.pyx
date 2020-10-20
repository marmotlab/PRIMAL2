#!python
# cython: language_level=3
# distutils: language = c
# distutils: sources = astarlib.c

# The MIT License
#
# Copyright (c) 2019 Herbert Shin  https://github.com/initbar/astarlib
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
astarlib
--------

This module implements A* pathfinding algorithm for graph and 2d search space.
"""

from collections import deque
from libc.math cimport pow
from libc.math cimport sqrt
import numpy as np

__all__ = (
    "NEIGHBOR_LINEAR_SQUARE",
    "NEIGHBOR_STAR_SQUARE",
    "PathNotFoundException",
    "aStar",
)


#
# Exceptions
#

class PathNotFoundException(ValueError):
    """A* path to destination not found"""
    pass


#
# Heuristics
#

# +---+---+---+
# | X | X | X |
# +---+---+---+
# | X |   | X |
# +---+---+---+
# | X | X | X |
# +---+---+---+
NEIGHBOR_STAR_SQUARE = (  # delta table for linear and diagonal neighbors
    (-1, 1),  (0, 1),  (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1),
)

# +---+---+---+
# |   | X |   |
# +---+---+---+
# | X |   | X |
# +---+---+---+
# |   | X |   |
# +---+---+---+
NEIGHBOR_LINEAR_SQUARE = (  # delta table for linear neighbors
    (0, 1), (-1, 0), (1, 0), (0, -1),
)

cpdef float euclidean_distance(int x1, int y1, int x2, int y2):
    """Euclidean distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    |   |   | E |
    +---+---/---+
    |   | / |   |
    +---/---+---+
    | S |   |   |  distance S->E = 2
    +---+---+---+
    """
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y2, 2))

cpdef unsigned int chebyshev_distance(int x1, int y1, int x2, int y2):
    """Chebyshev distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    | E | E | E |
    +---\-|-/---+
    | E - S - E |
    +---/-|-\---+
    | E | E | E |  distance S->E = 1
    +---+---+---+
    """
    return max(abs(x2 - x1), abs(y2 - y1))

cpdef unsigned int manhattan_distance(int x1, int y1, int x2, int y2):
    """Manhattan distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    | E | E | E |
    +---\-|-/---+
    | E - S - E |
    +---/-|-\---+
    | E | E | E |  distance S->E = {1, 2}
    +---+---+---+
    """
    return abs(x1 - x2) + abs(y1 - y2)


#
# A*
#

cdef class pNode:

    cdef public int x, y
    cdef public unsigned int g, h

    # unlike conventional graphs or trees, the `parent` and `child`
    # relationship here is used in the context of a linked-list.
    cdef public object parent, child

    def __cinit__(self, x, y, g=0, h=0, parent=None, child=None):
        """represents point-node (pNode).
        :type x: int or float
        :param x: value on Cartesian x-axis.
        :type y: int or float
        :param y: value on Cartesian y-ayis.
        :type parent: pNode
        :param parent: parent pNode.
        :type child: pNode
        :param child: child pNode.
        """
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.parent = parent
        self.child = child

    @property
    def point(self):
        return (self.x, self.y)  # (x, y)

    @property
    def f(self):
        return self.g + self.h  # f(n) := g(n) + h(n)

cdef resolve_child_pnodes_to_points(pnode):
    """resolve linked-list pNodes into sequence of points in head to tail direction.
    :type pnode: pNode
    :param pnode: pNode instance.

    This function returns (a.point .. z.point):

    +----------+     +---+            +----------+
    | pNode: a | --> | b | --> .. --> | pNode: z |
    +----------+     +---+            +----------+
       (head)                            (tail)
    """
    head = pnode
    if head.child is None:
        return (head.point,)
    path = [head.point]
    while head.child is not None:
        path.append(head.child.point)
        head = head.child
    return tuple(path)

cdef resolve_parent_pnodes_to_points(pnode, reverse=False):
    """resolve linked-list pNodes into sequence of points in tail to head direction.
    :type pnode: pNode
    :param pnode: pNode instance.

    :type reverse: bool
    :param reverse: reverse the resolution order.

    This function returns (a.point .. z.point) or (z.point .. a.point)
    depending on the `reverse` state:

    +----------+     +---+            +----------+
    | pNode: a | <-- | b | <-- .. <-- | pNode: z |
    +----------+     +---+            +----------+
       (head)                            (tail)
    """
    tail = pnode
    if tail.parent is None:
        return (tail.point,)
    # utilize double-ended queue to pre-inject the resolved points in sequential
    # or reversed direction. Otherwise, we need to unnecessarily iterate through
    # the sequence twice by calling [::-1] on the result.
    path = deque([tail.point])
    while tail.parent is not None:
        if reverse is True:
            path.appendleft(tail.parent.point)  # (a.point -> z.point)
        else:
            path.append(tail.parent.point)  # (z.point -> a.point)
        tail = tail.parent
    return tuple(path)


class aStar(object):

    __slots__ = "_buffer", "_height", "_width"

    def __init__(self, array=[[]]):
        self._height = len(array)
        self._width = len(array[-1])
        self._buffer = array

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def find_path(self, start=(0, 0), end=(0, 0), h=manhattan_distance):
        """find A* path from `start` to `end`.
        :type start: tuple
        :param start: starting position in (x, y).

        :type end: tuple
        :param end: destination position in (x, y).

        :type h: function
        :param h: h function to calculate H(n).
        """

        if start == end:
            # if the starting point is already resting on the end point,
            # do not spend time trying to find the most optimal path.
            return (), 0  # path, cost
        opened = {start: pNode(x=start[0], y=start[1], h=h(start[0], start[1], end[0], end[1]))}
        closed = {}
        while opened:
            # pick a node with the lowest F(n) score; F(n) = G(n) + H(n).
            #
            # |<------ G(n) -------->|<------------ H(n) ------------>|
            # |----------------------|------------------------------->|
            # origin              position                         goal
            current_node = min(opened.values(), key=lambda pnode: pnode.f)
            if current_node.point == end:
                # since `current_node` is at the destination, recursively
                # resolve and reverse the order of its' parents to construct
                # the correct traversal sequence. Otherwise, the path order
                # will be backwards from goal -> origin.
                path = resolve_parent_pnodes_to_points(pnode=current_node, reverse=True)
                cost = len(path)
                return path, cost
            # close the `current_node` since it is now evaluated and begin
            # evaluating its' neighbors' F(n) scores. Neighbors' F(n)
            # scores will decide where to advance next.
            closed[current_node.point] = opened.pop(current_node.point)
            neighbors = self.neighbors_at(x=current_node.x, y=current_node.y)
            for neighbor in neighbors:  # neighbor := (x, y)
                if neighbor in closed:
                    continue
                elif self._buffer[neighbor[0]][neighbor[1]] == -1:
                    # ignore this neighbor (an obstacle).
                    closed[neighbor] = None
                    continue
                # since this node can be traversed, generate a neighbor
                # pNode instance in relation to the `current_node`.
                #
                # +--------------+  G(i)  +----------+  H(i)  +------+
                # | current_node |--------| neighbor |--------| goal |
                # +--------------+        +----------+        +------+
                #        |
                #        | G(i-1)
                #        |
                #    +--------+
                #    | origin |
                #    +--------+
                if neighbor not in opened:
                    opened[neighbor] = pNode(x=neighbor[0], y=neighbor[1],
                                             g=sum([
                                                 current_node.g,  # G(i-1)
                                                 h(current_node.x,  # G(i)
                                                   current_node.y,
                                                   neighbor[0],
                                                   neighbor[1])
                                             ]),
                                             h=h(neighbor[0],  # H(i)
                                                 neighbor[1],
                                                 end[0],
                                                 end[1]))
                neighbor_pnode = opened[neighbor]
                # lower or equal neighbor's F(n) score means we're heading
                # towards the right direction to the goal. Attach this
                # neighbor as the next linked node to the `current_node`.
                if neighbor_pnode.f <= current_node.f:
                    neighbor_pnode.parent = current_node
                    current_node.child = neighbor_pnode
                    continue
                closed[neighbor] = opened.pop(neighbor)
        raise PathNotFoundException

    def getAstarDistanceMap(self, end=(0, 0), h=manhattan_distance):
        """build A* distance map for goal `end`.
        :type end: tuple
        :param end: destination position in (x, y).

        :type h: function
        :param h: h function to calculate H(n).
        """

        opened = {end: pNode(x=end[0], y=end[1], h=0)}
        closed = {}

        while opened:
            current_node = min(opened.values(), key=lambda pnode: pnode.f)

            # close the `current_node` since it is now evaluated and begin
            # evaluating its' neighbors' F(n) scores. Neighbors' F(n)
            # scores will decide where to advance next.
            closed[current_node.point] = opened.pop(current_node.point)
            neighbors = self.neighbors_at(x=current_node.x, y=current_node.y)
            for neighbor in neighbors:  # neighbor := (x, y)
                if neighbor in closed:
                    continue

                if neighbor not in opened:
                    opened[neighbor] = pNode(x=neighbor[0], y=neighbor[1],
                                             g=current_node.g + 1, h=0)

                neighbor_pnode = opened[neighbor]
                # lower or equal neighbor's F(n) score means we're heading
                # towards the right direction to the goal. Attach this
                # neighbor as the next linked node to the `current_node`.
                if neighbor_pnode.f <= current_node.f:
                    neighbor_pnode.parent = current_node
                    current_node.child = neighbor_pnode
                    continue

        Astar_map = - np.ones(( len(self._buffer), len(self._buffer[0]) ))
        for (i, j) in closed.keys():
            if self._buffer[i][j] == -1:
                Astar_map[i][j] = -1
            else:
                Astar_map[i][j] = closed[(i,j)].g
        return Astar_map

    def neighbors_at(self, x, y, delta=NEIGHBOR_LINEAR_SQUARE):
        """get indexes of neighboring elements of (x, y)"""
        if not delta:
            yield None
        # traverse through immediately adjacent elements and ignore non-existent
        # neighbors. For example, no valid neighbors exist adjacent to (-2, -2),
        # but (0, -1) is adjacent to a valid neighbor (0, 0).
        cdef int nx, ny
        for dx, dy in delta:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self._buffer[nx][ny] != -1:
                yield (nx, ny)
