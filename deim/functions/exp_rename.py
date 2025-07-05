import os
import json
import csv
import shutil

# 假设 runs 目录位于与 functions 目录同级的目录中
RELATIVE_PATH_TO_RUNS = '../runs'  # 从 functions 目录到 runs 目录的相对路径

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 base_dir 的绝对路径
base_dir = os.path.join(script_dir, RELATIVE_PATH_TO_RUNS)
base_dir = os.path.abspath(base_dir)

print(f"Base directory: {base_dir}")


def convert_log_to_csv(log_file, csv_file):
    """
    将 log.txt 文件转换为类似 YOLO results.csv 格式的文件。
    """
    with open(log_file, 'r') as f_in, open(csv_file, 'w', newline='') as f_out:
        reader = (json.loads(line) for line in f_in)
        writer = csv.writer(f_out)

        # 写入表头
        header = ['epoch', 'train_lr', 'train/box_loss', 'train/fgl_loss', 'train/giou_loss', 'train/mal_loss',
                  'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'n_parameters']
        writer.writerow(header)

        for data in reader:
            epoch = data['epoch']
            train_lr = data['train_lr']
            train_loss_bbox = data['train_loss_bbox']
            train_loss_fgl = data['train_loss_fgl']
            train_loss_giou = data['train_loss_giou']
            train_loss_mal = data['train_loss_mal']
            test_coco_eval_bbox = data['test_coco_eval_bbox']
            mAP50 = test_coco_eval_bbox[1]
            mAP50_95 = test_coco_eval_bbox[0]
            n_parameters = data['n_parameters']

            row = [epoch, train_lr, train_loss_bbox, train_loss_fgl, train_loss_giou, train_loss_mal, mAP50,
                   mAP50_95, n_parameters]
            writer.writerow(row)


def get_best_map(log_file):
    """
    从 log.txt 文件中提取最佳 mAP50-95(B) 值。
    """
    best_map = 0.0
    try:
        with open(log_file, 'r') as f_log:
            for line in f_log:
                try:
                    data = json.loads(line)
                    test_coco_eval_bbox = data.get('test_coco_eval_bbox')  # 获取 test_coco_eval_bbox 列表

                    if test_coco_eval_bbox is None or not isinstance(test_coco_eval_bbox, list) or len(
                            test_coco_eval_bbox) < 2:
                        print(
                            f"Warning: 'test_coco_eval_bbox' not found or invalid in line: {line.strip()}")
                        continue

                    current_map = test_coco_eval_bbox[0]  # 使用 test_coco_eval_bbox[0] (mAP50-95(B))
                    if current_map > best_map:
                        best_map = current_map
                except (KeyError, ValueError, TypeError) as e:
                    print(
                        f"Warning: Invalid JSON, missing key, or incorrect type in line: {line.strip()} - {e}")
                    continue  # Skip to the next line
    except FileNotFoundError as e:
        print(f"Error reading {log_file}: {e}")
        return None  # Indicate error

    return best_map


def process_runs_directory(base_dir):
    """
    处理指定 base_dir 目录下的所有 log.txt 文件。  现在会进入每个二级子目录查找 log.txt
    """
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue  # 跳过非目录文件

        # 移除 .ipynb_checkpoints 目录
        checkpoints_dir = os.path.join(model_path, '.ipynb_checkpoints')
        if os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir):
            try:
                shutil.rmtree(checkpoints_dir)
                print(f"Removed directory: {checkpoints_dir}")
            except OSError as e:
                print(f"Error removing directory {checkpoints_dir}: {e}")

        for run_name in os.listdir(model_path):
            run_path = os.path.join(model_path, run_name)
            if not os.path.isdir(run_path):
                continue  # 跳过非目录文件

            # 新增的循环，进入 run_path 的下一级目录
            for sub_dir_name in os.listdir(run_path):
                sub_dir_path = os.path.join(run_path, sub_dir_name)
                if not os.path.isdir(sub_dir_path):
                    continue

                log_file = os.path.join(sub_dir_path, 'log.txt')
                csv_file = os.path.join(sub_dir_path, 'results.csv')

                if not os.path.exists(log_file):
                    print(f"Warning: No 'log.txt' file found in {sub_dir_path}")
                    continue

                if os.path.exists(csv_file):
                    # print(f"Skipping {sub_dir_path} because 'results.csv' already exists.")
                    continue

                # 1. 转换为 results.csv
                convert_log_to_csv(log_file, csv_file)
                print(f"Converted {log_file} to {csv_file}")

                # 2. 读取最佳 mAP50-95 并重命名 运行文件夹
                best_map = get_best_map(log_file)

                if best_map is None:
                    print(f"Skipping {sub_dir_path} due to error reading log file.")
                    continue  # Skip to the next run

                if best_map > 0.0:
                    new_sub_dir_name = f"{int(best_map * 1000)}"  # 提取前三位整数 (mAP50-95)
                    # new_sub_dir_name = f"{best_map:.3f}" # 提取到小数点后三位
                    new_sub_dir_path = os.path.join(run_path, new_sub_dir_name)

                    if os.path.exists(new_sub_dir_path):
                        # 目标文件夹已存在，比较 mAP50-95
                        existing_log_file = os.path.join(new_sub_dir_path, 'log.txt')
                        existing_best_map = get_best_map(existing_log_file)

                        if existing_best_map is None:
                            print(
                                f"Warning: Could not read log file in existing directory {new_sub_dir_path}.  Keeping existing directory.")
                        elif best_map >= existing_best_map:  # 新的 mAP50-95 更高或相等
                            # 删除旧的并重命名
                            try:
                                shutil.rmtree(new_sub_dir_path)  # 删除整个文件夹
                                os.rename(sub_dir_path, new_sub_dir_path)  # 重命名的是 sub_dir_path
                                print(
                                    f"Renamed {sub_dir_path} to {new_sub_dir_path} (replaced existing, best_map was higher or equal)")
                                # 更新 sub_dir_path
                                sub_dir_path = new_sub_dir_path
                            except OSError as e:
                                print(
                                    f"Error renaming {sub_dir_path} to {new_sub_dir_path} or removing existing directory: {e}")
                        else:
                            # 旧的 mAP50-95 更高，删除当前文件夹
                            try:
                                shutil.rmtree(sub_dir_path)  # 删除整个文件夹
                                print(f"Deleted {sub_dir_path} (existing mAP50-95 was higher)")
                            except OSError as e:
                                print(f"Error removing {sub_dir_path}: {e}")


                    else:
                        # 目标文件夹不存在，直接重命名
                        try:
                            os.rename(sub_dir_path, new_sub_dir_path)  # 重命名的是 sub_dir_path
                            print(f"Renamed {sub_dir_path} to {new_sub_dir_path}")
                            # 更新 sub_dir_path
                            sub_dir_path = new_sub_dir_path
                        except OSError as e:
                            print(f"Error renaming {sub_dir_path} to {new_sub_dir_path}: {e}")
                else:
                    print(f"Warning: No valid mAP50-95(B) found in {log_file}")

                # 3. 删除 checkpoint 文件
                for filename in os.listdir(sub_dir_path):
                    if filename.startswith('checkpoint') and filename.endswith('.pth'):
                        filepath = os.path.join(sub_dir_path, filename)
                        os.remove(filepath)
                        print(f"Deleted {filepath}")
                    elif filename == 'last.pth':
                        filepath = os.path.join(sub_dir_path, filename)
                        os.remove(filepath)
                        print(f"Deleted {filepath}")

    print("All directories processed!")


# 添加这行来启动处理流程
process_runs_directory(base_dir)