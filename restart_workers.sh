#!/bin/bash

for i in {101..200}; do
    docker restart allora-worker-$i
    sleep 3
done
