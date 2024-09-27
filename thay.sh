#!/bin/bash

# Đường dẫn đến file vi.txt
vi_file="/root/worker3-20m/mnemonic_info.txt"

# Thư mục chứa các file JSON cần thay đổi
json_dir="/root/worker3-20m"

# Đường dẫn đến file Python cần thay đổi
py_file="/root/worker3-20m/model.py"  # Thay đổi đường dẫn tới file model.py tại đây

# Kiểm tra sự tồn tại của file vi.txt
if [ ! -f "$vi_file" ]; then
  echo "File vi.txt không tồn tại."
  exit 1
fi

# Kiểm tra sự tồn tại của file Python
if [ ! -f "$py_file" ]; then
  echo "File model.py không tồn tại."
  exit 1
fi

# Đếm số dòng trong file vi.txt
line_count=$(wc -l < "$vi_file")

# Đọc từng dòng trong file vi.txt
line_number=1
while IFS= read -r new_value_address && [ $line_number -le $line_count ]; do
  # Tạo tên file JSON theo thứ tự config_1.json, config_2.json, ...
  json_file="$json_dir/config_$line_number.json"

  # Kiểm tra sự tồn tại của file JSON
  if [ ! -f "$json_file" ]; then
    echo "File $json_file không tồn tại."
    exit 1
  fi

  # Thay thế giá trị "addressRestoreMnemonic" trong file JSON
  sed -i 's/"addressRestoreMnemonic": ".*"/"addressRestoreMnemonic": "'"$new_value_address"'"/g' "$json_file"
  echo "Đã thay đổi giá trị 'addressRestoreMnemonic' trong file $json_file."

  # Thay thế maxRetries và delay
  sed -i 's/"maxRetries": 3/"maxRetries": 10/g' "$json_file"
  sed -i 's/"delay": 5/"delay": 4/g' "$json_file"

  # Thay thế nodeRpc
  sed -i 's|"nodeRpc": "https://allora-rpc.testnet.allora.network/"|"nodeRpc": "http://65.108.204.245:32657/"|g' "$json_file"
  echo "Đã thay đổi giá trị 'nodeRpc' trong file $json_file."

  # Tăng số thứ tự của file JSON
  ((line_number++))
done < "$vi_file"

echo "Hoàn thành việc thay đổi các file JSON."
