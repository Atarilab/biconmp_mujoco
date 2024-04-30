#!/bin/bash

ORANGE=$'\e[0;33m'
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

# do one update
sudo apt update

###### BICONVEX_MPC DEPENDENCIES ######

# Check if project directory exists
if [ -d "/home/atari_ws/project" ]; then
    echo "project found"

    # check if user pulled the repo correctly with non-empty extern directory
    if [ -z "$(ls -A /home/atari_ws/project/extern)" ]; then
        echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
        exit 10

    else
        echo "${ORANGE}Building project...${NC}"
        
        # check if project is built before
        if [ -d "/home/atari_ws/project/build" ]; then

            echo "build dir in project found"
            echo "deleting build file..."

            # Delete any previous build files in project
            if [ -d "/home/atari_ws/project/build" ]; then
                cd /home/atari_ws/project
                sudo rm -rf build
            fi
        fi
        
        source /home/atari/.bashrc
        cd /home/atari_ws/project
        mkdir build && cd build

        cmake .. -DCMAKE_BUILD_TYPE=Release #-Dpybind11_DIR=/home/atari_ws/project/extern/build
        make install -j12
            
        echo "${GREEN}Succesfully build project${NC}"
        
    fi