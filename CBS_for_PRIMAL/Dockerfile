FROM ubuntu:jammy
RUN apt-get update
RUN ["apt-get", "install", "--yes", "--no-install-recommends","make","cmake","build-essential","libboost-all-dev","pip","python3-dev","python3-pybind11","python3-numpy"]
COPY ./. /cbs_for_primal/codes/ 
WORKDIR /cbs_for_primal/codes/ 
RUN rm -rdf /cbs_for_primal/codes/build 

