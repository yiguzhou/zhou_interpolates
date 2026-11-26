#!/bin/bash

# submit run_once.py serially and manually define input data

#cd /path/to/project/directory/code/

cd /home/yzhou/gitrepo/clean/interpolate/code/

#:::::::::::::::Running surface map experiments::::::::::::#
source batch_run_config 'neuromaps'

for p in 5 10 25 50; do
    for m in ${maptags[@]}; do

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 1 -s 'fibonacci'

        python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'dsk33'

        #python run_once.py -g 'reduced-midthickness' -d 'neuromaps' -p ${p} -m ${m} -n 100 -s 'lobes'
    
    done
done