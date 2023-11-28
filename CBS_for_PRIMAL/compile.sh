#!/bin/bash

mkdir build

# build exec for cpp

cd build
cmake ../
make -j


# build exec for python

# cd build
# cmake ../ -DPYTHON=true
# make -j
