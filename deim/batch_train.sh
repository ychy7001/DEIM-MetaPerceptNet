#!/bin/bash

# 设置变量
dataset="breast_bm_b-mode"
base_dir=$(pwd) # 获取当前目录的绝对路径

# 循环执行训练和重命名
find "${base_dir}/myConfigs/" -maxdepth 1 -name "${dataset}*.yml" -print0 | while IFS= read -r -d $'\0' yml_file; do
  echo "开始处理: $yml_file"

  # 1. 训练模型
  torchrun --nproc_per_node=1 "${base_dir}/functions/train.py" -c "$yml_file"
  if [ $? -ne 0 ]; then
    echo "训练失败，文件: $yml_file"
    exit 1  # 退出脚本，并返回错误码 1
  fi

  # 2. 重命名实验文件夹 (在训练完成后执行)
  find "${base_dir}/runs" -type d -name ".ipynb_checkpoints" -print0 | xargs -0 rm -rf
  python "${base_dir}/functions/exp_rename.py"

  echo "完成处理: $yml_file"
  echo "------------------------------------"
done

echo "所有文件处理完毕."
exit 0 # 正常退出，返回错误码 0
