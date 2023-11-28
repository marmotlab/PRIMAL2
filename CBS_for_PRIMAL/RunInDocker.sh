#!/usr/bin/env bash
echo "Only run this script in the root of your code base."
echo "Create Dockerfile using apt.txt"

# Specify which docker to use.
content="FROM ubuntu:jammy\nRUN apt-get update\n"

# Read packages to install.
pkgs='RUN ["apt-get", "install", "--yes", "--no-install-recommends"'
while read -r line;
do
   pkgs="${pkgs},\"$line\"" ;
done < apt.txt
pkgs="${pkgs}]\n"
content="${content}${pkgs}"

# Copy codes to target dir and set codes dir to be the working directory.
# Then run compile.sh to compile codes.
content="${content}COPY ./. /cbs_for_primal/codes/ \n"
content="${content}WORKDIR /cbs_for_primal/codes/ \n"
content="${content}RUN rm -rdf /cbs_for_primal/codes/build \n"
# content="${content}RUN chmod u+x compile.sh \n"
# content="${content}RUN ./compile.sh \n"
echo -e $content > Dockerfile

echo "Remove container and images if exist... ..."
out=$(docker container stop cbs_for_primal_test 2>&1 ; docker container rm cbs_for_primal_test 2>&1 ; docker rmi cbs_for_primal_image 2>&1)

echo "Build image and run the container... ..."
docker build --no-cache -t cbs_for_primal_image ./
docker container run -it --name cbs_for_primal_test cbs_for_primal_image
