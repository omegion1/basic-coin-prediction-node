#!/bin/bash
# -*- coding: utf-8 -*-
for i in {1..100}; do
    cp "./config_${i}.json" ./config.json
    ./init.config
    cp ./worker-data/env_file "./worker-data/env_file_${i}"
done
