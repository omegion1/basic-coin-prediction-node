# -*- coding: utf-8 -*-

import json

# Ð?nh nghia file g?c config.json
config_template = {
    "wallet": {
        "addressKeyName": "test1",
        "addressRestoreMnemonic": "...", 
        "alloraHomeDir": "",
        "gas": "auto",
        "gasAdjustment": 1.5,
        "nodeRpc": "",
        "maxRetries": 10,
        "delay": 4,
        "submitTx": True
    },
    "worker": [
        {
            "topicId": 7,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8013/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 8,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8013/inference/{Token}",
                "Token": "BNB"
            }
        },
        {
            "topicId": 9,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8013/inference/{Token}",
                "Token": "ARB"
            }
        }
    ]
}

# Hàm d? t?o các file config
def generate_configs(num_files):
    for i in range(1, num_files + 1):
        # T?o b?n sao c?a template
        config_data = config_template.copy()
        
        # C?p nh?t "addressKeyName" cho t?ng file
        config_data['wallet']['addressKeyName'] = f'test{i}'
        
        # Tên file
        file_name = f'config_{i}.json'
        
        # Ghi n?i dung ra file json
        with open(file_name, 'w') as file:
            json.dump(config_data, file, indent=4)
        
        print(f'{file_name} created successfully.')

# G?i hàm d? t?o các file (ví d? t?o 10 file)
generate_configs(100)
