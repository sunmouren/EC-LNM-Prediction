#!/bin/bash

for i in $(seq 3 -1 0); do
    echo "Number $i"
    /path/envs/tf211/bin/python3 run.py $i 
    sleep 5
done