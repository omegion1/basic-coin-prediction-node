#!/bin/bash

for i in {1..100}; do
    cp "./config_${i}.json" ./config.json
    ./init.config
    
    # Ki?m tra xem t?p env_file có t?n t?i không tru?c khi sao chép
    if [[ -f "./worker-data/env_file" ]]; then
        cp ./worker-data/env_file "./worker-data/env_file_$((i + 100))"
    else
        echo "T?p env_file không t?n t?i."
    fi
done
