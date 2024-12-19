#!/bin/bash
cd setup

ls

./download_idenprof.sh
./download_fairface.sh
./download_pata.sh
./download_phase.sh
./download_miap.sh

# mv data-scratch data
# mv data ..
# cd ..