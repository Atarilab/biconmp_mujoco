#!/bin/bash

ORANGE=$'\e[0;33m'
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

# do one update
sudo apt update

###### BICONVEX_MPC DEPENDENCIES ######

# Check if biconvex_mpc directory exists
if [ -d "/home/atari_ws/biconvex_mpc" ]; then
    echo "biconvex_mpc found"

    # check if user pulled the repo correctly with non-empty extern directory
    if [ -z "$(ls -A /home/atari_ws/biconvex_mpc/extern)" ]; then
        echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
        exit 10

    else
        echo "${ORANGE}Building biconvex_mpc...${NC}"
        
        # check if biconvex_mpc is built before
        if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then

            echo "build dir in biconvex_mpc found"
            echo "deleting build file..."

            # Delete any previous build files in biconvex_mpc
            if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then
                cd /home/atari_ws/biconvex_mpc
                sudo rm -rf build
            fi
        fi
        
        source /home/atari/.bashrc
        cd /home/atari_ws/biconvex_mpc
        mkdir build && cd build

        cmake .. -DCMAKE_BUILD_TYPE=Release #-Dpybind11_DIR=/home/atari_ws/biconvex_mpc/extern/build
        make install -j16
            
        echo "${GREEN}Succesfully build biconvex_mpc${NC}"
        
    fi
fi