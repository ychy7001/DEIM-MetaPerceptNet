import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig # YAMLConfig
import json  # 导入 json 模块

import warnings
warnings.filterwarnings('ignore')

# --- 第一段代码的参数 ---
# 原训练数据
dataset_name = "BUS-BRA"
# 测试数据
image_path = os.path.join('/mnt/d/VSCode/Datasets/coco/breast_bm_b-mode', "images/test")
coco_label_path = os.path.join('/mnt/d/VSCode/Datasets/coco/breast_bm_b-mode', "labels/test")

# save path
runs_path = os.path.join("/mnt/d/VSCode/Ckpts/deim/runs", dataset_name)
results_path = os.path.join("/mnt/d/VSCode/Ckpts/deim/results", dataset_name) # Add results_path

# default
title = "DEIM"
img_size = 640
# 设置最大插值阈值数量
max_interpolated_thresholds = 500

# --- 总表保存路径 ---
overall_results_path = os.path.join(results_path, "overall_results.csv")
os.makedirs(results_path, exist_ok=True)

# --- 读取已存在的总表 ---
try:
    overall_df = pd.read_csv(overall_results_path)
    print(f"已加载已存在的总表：{overall_results_path}")
except FileNotFoundError:
    overall_df = pd.DataFrame()  # 创建一个空的 DataFrame
    print(f"未找到已存在的总表，创建一个新的总表。")

# --- 遍历 runs 目录 ---
for folder_name in os.listdir(runs_path):
    folder_path = os.path.join(runs_path, folder_name)
    if os.path.isdir(folder_path):
        # 遍历3位数字的子文件夹
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if subfolder_name.isdigit():  # 确保是数字文件夹
                # 遍历 stg (stage)
                for stg in ["stg1", "stg2"]:
                    # --- 模型路径修改：指定 runs_path 下的权重文件 ---
                    model_ckpt_path = os.path.join(runs_path, folder_name, subfolder_name, f"best_{stg}.pth")
                    config_path = os.path.join("./myConfigs", f"{dataset_name}_{folder_name}.yml") # 配置文件路径
                    if not os.path.exists(model_ckpt_path):
                        print(f"Model checkpoint not found: {model_ckpt_path}, skipping...")
                        continue
                    if not os.path.exists(config_path):
                        print(f"Config file not found: {config_path}, skipping...")
                        continue

                    # --- 检查模型是否已测过 ---
                    model_name = folder_name  # 提取模型名称, 已经重命名，直接使用folder_name
                    model_name_for_check = folder_name # 去除数据集名，例如 11n-RSCD

                    if not overall_df.empty:
                        found_model = False
                        for index, row in overall_df.iterrows():
                            csv_model = str(row['Model']).strip() # Explicitly convert to string and strip whitespace
                            check_model = str(model_name_for_check).strip()
                            csv_subfolder = str(row['Subfolder']).strip()
                            check_subfolder = str(subfolder_name).strip()
                            csv_stg = str(row['Stage']).strip()
                            check_stg = str(stg).strip()

                            if csv_model == check_model and csv_subfolder == check_subfolder and csv_stg == check_stg:
                                found_model = True
                                break # Exit loop once found

                        if found_model:
                            print(f"模型 {model_name}/{subfolder_name}/{stg} 已存在于总表中，跳过。")
                            continue
                        else:
                            print(f"模型 {model_name}/{subfolder_name}/{stg} 不存在于总表中，正在处理...")
                    else:
                        print(f"总表为空，正在处理模型 {model_name}/{subfolder_name}/{stg}...")

                    # --- 第二段代码的参数（部分） ---
                    # Extract result folder based on model path
                    base_model_path = os.path.normpath(model_ckpt_path)
                    relative_path = "/".join(base_model_path.split(os.sep)[2:-2]) # Extract "yolo11n/671"

                    # 修改 output_folder 的构建方式
                    output_folder = os.path.join(results_path, folder_name, subfolder_name, stg)
                    os.makedirs(output_folder, exist_ok=True)

                    # Result Paths
                    roc_curve_path = os.path.join(output_folder, "roc_curve.png")
                    pr_curve_path = os.path.join(output_folder, "pr_curve.png")
                    pr_csv_path = os.path.join(output_folder, "pr_curve_data.csv")

                    # New paths for saving raw data
                    raw_scores_path = os.path.join(output_folder, "raw_scores.csv")
                    interpolated_thresholds_path = os.path.join(output_folder, "interpolated_thresholds.txt")
                    original_pr_path = os.path.join(output_folder, "original_pr.csv")
                    
                    # --- DEIM 模型加载 ---
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # 1. 加载配置文件
                    cfg = YAMLConfig(config_path, resume=model_ckpt_path)  # 使用 resume 加载权重
                    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
                    checkpoint = torch.load(model_ckpt_path, map_location='cpu')
                    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
                    cfg.model.load_state_dict(state)

                    class Model(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.model = cfg.model.deploy()
                            self.postprocessor = cfg.postprocessor.deploy()
                        def forward(self, images, orig_target_sizes):
                            outputs = self.model(images)
                            outputs = self.postprocessor(outputs, orig_target_sizes)
                            return outputs
                    model = Model().to(device)
                    model.eval() # 设置为评估模式

                    # Initialize variables
                    image_files = [f for f in glob.glob(os.path.join(image_path, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_scores = []
                    all_labels = []
                    all_filenames = [] # 用于存储文件名

                    try: # ADDED: Handle KeyboardInterrupt
                        # Process images
                        for image_file in tqdm(image_files, desc=f"Processing images for {folder_name}/{subfolder_name}/{stg}"):
                            image_id = os.path.splitext(os.path.basename(image_file))[0]
                            label_file = os.path.join(coco_label_path, f"{image_id}.json") # 修改标签文件后缀为 .json
                            if not os.path.exists(label_file):
                                print(f"Label file {label_file} not found, skipping...")
                                continue

                            # Load true label from JSON
                            try:
                                with open(label_file, 'r') as f:
                                    label_data = json.load(f)
                                    # 假设label在shapes[0].label中
                                    actual_label = int(label_data['shapes'][0]['label'])  # 获取标签
                            except (KeyError, IndexError, json.JSONDecodeError) as e:
                                print(f"Error reading label from {label_file}: {e}, skipping...")
                                continue

                            # --- DEIM 模型推理 ---
                            # 1. 图像预处理
                            im_pil = Image.open(image_file).convert('RGB')
                            w, h = im_pil.size
                            orig_size = torch.tensor([[w, h]]).to(device) # orig_size
                            transforms = T.Compose([
                                T.Resize((img_size, img_size)),
                                T.ToTensor(),
                            ])
                            im_data = transforms(im_pil).unsqueeze(0).to(device)

                            # 2. 模型推理
                            with torch.no_grad():  # 禁用梯度计算
                                labels, boxes, scores = model(im_data, orig_size) # 需要修改

                            # 3. 提取恶性肿瘤的预测分数 (请根据你的类别ID修改)
                            malignant_scores = []
                            if labels is not None and scores is not None:  # 检查 labels 和 scores 是否为 None
                                for i in range(len(labels[0])):
                                    if labels[0][i] == 1:  # 1 是恶性肿瘤的类别ID
                                        malignant_scores.append(scores[0][i].item())
                                image_score = max(malignant_scores) if malignant_scores else 0.0
                            else:
                                image_score = 0.0  # 如果 labels 或 scores 为 None，则设置 image_score 为 0.0

                            # Save scores and labels
                            all_scores.append(image_score)
                            all_labels.append(actual_label)
                            all_filenames.append(os.path.basename(image_file)) # 保存文件名

                        # Metrics calculation
                        precision, recall_vals, thresholds = precision_recall_curve(all_labels, all_scores)
                        pr_auc = auc(recall_vals, precision)

                        # --- 保存原始分数和标签 ---
                        raw_data = {'filename': all_filenames, 'score': all_scores, 'label': all_labels} # 添加 filename
                        raw_df = pd.DataFrame(raw_data)
                        raw_df.to_csv(raw_scores_path, index=False)
                        print(f"Raw scores and labels saved to {raw_scores_path}")

                        # --- 修正 original_pr_df 的创建 ---
                        min_len = min(len(precision), len(recall_vals), len(thresholds))
                        original_pr_data = {
                            'precision': precision[:min_len],
                            'recall': recall_vals[:min_len],
                            'threshold': thresholds[:min_len]
                        }
                        original_pr_df = pd.DataFrame(original_pr_data)
                        original_pr_df.to_csv(original_pr_path, index=False)
                        print(f"Original precision, recall, thresholds saved to {original_pr_path}")
                        
                        # --- 阈值插值 ---
                        min_threshold = np.min(thresholds)
                        max_threshold = np.max(thresholds)
                        threshold_range = max_threshold - min_threshold
                        num_thresholds = len(thresholds)
                        avg_threshold_diff = threshold_range / (num_thresholds - 1) if num_thresholds > 1 else 0

                        # 确定插值阈值
                        interpolation_interval = avg_threshold_diff / 2  # 插值间隔设为平均差值的一半
                        interpolated_thresholds = np.arange(min_threshold, max_threshold, interpolation_interval)
                        interpolated_thresholds = np.unique(np.concatenate((thresholds, interpolated_thresholds))) # 去重

                        # --- 限制插值阈值的数量 ---
                        if len(interpolated_thresholds) > max_interpolated_thresholds:
                            # 如果超过最大数量，则重新计算插值间隔，并生成固定数量的阈值
                            interpolated_thresholds = np.linspace(min_threshold, max_threshold, max_interpolated_thresholds)

                        # --- 保存插值阈值 ---
                        np.savetxt(interpolated_thresholds_path, interpolated_thresholds, delimiter=",")
                        print(f"Interpolated thresholds saved to {interpolated_thresholds_path}")

                        # Prepare PR Curve Data with TN, FP, FN, TP
                        pr_data = []
                        for threshold in interpolated_thresholds:
                            predicted_labels = [1 if score >= threshold else 0 for score in all_scores]
                            tn, fp, fn, tp = confusion_matrix(all_labels, predicted_labels).ravel()

                            # 重新计算 precision 和 recall
                            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                            
                            f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

                            # 将结果格式化为与总表一致
                            pr_data.append({
                                "Model": model_name_for_check, # 模型名称
                                "Subfolder": subfolder_name, # 子文件夹名称
                                "Stage": stg, # 模型阶段
                                "F1-Score": round(f1, 3),
                                "Precision": round(precision_val, 3),
                                "Recall/Sens": round(recall_val, 3),
                                "Spec": round(tn / (tn + fp), 3) if (tn + fp) > 0 else 0, # 计算 specificity
                                "TN": tn,
                                "FP": fp,
                                "FN": fn,
                                "TP": tp,
                                "Threshold": round(threshold, 3)
                            })

                        # Save PR Curve Data to CSV
                        pr_df = pd.DataFrame(pr_data)

                        # 按照总表的顺序调整列的顺序
                        pr_df = pr_df[["Model", "Subfolder", "Stage", "F1-Score", "Precision", "Recall/Sens", "Spec", "TN", "FP", "FN", "TP", "Threshold"]] # 删除 PRCAUC 和 ROCAUC
                        pr_df.to_csv(pr_csv_path, index=False)
                        print(f"PR curve data saved to {pr_csv_path}")

                        # Save PR Curve Plot
                        plt.figure()
                        plt.plot(recall_vals, precision, color="darkblue", lw=2, label=f"PR curve (area = {pr_auc:.3f})")
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.title(title)
                        plt.legend(loc="lower right")
                        plt.savefig(pr_curve_path)
                        plt.close()
                        print(f"PR curve saved to {pr_curve_path}")

                        # Save ROC Curve Plot
                        fpr, tpr, _ = roc_curve(all_labels, all_scores)
                        roc_auc = auc(fpr, tpr)
                        plt.figure()
                        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
                        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(title)
                        plt.legend(loc="lower right")
                        plt.savefig(roc_curve_path)
                        plt.close()
                        print(f"ROC curve saved to {roc_curve_path}")

                        # --- 提取最佳 F1-score 及其对应指标 ---
                        best_f1_index_int = pr_df['F1-Score'].sort_values(ascending=False).index[0] # 使用 sort_values 获取索引
                        best_threshold = pr_df.iloc[best_f1_index_int]['Threshold'] # 通过整数索引获取 Threshold 值
                        best_f1 = pr_df.iloc[best_f1_index_int]['F1-Score']
                        best_tn = int(pr_df.iloc[best_f1_index_int]['TN']) # 通过整数索引获取 TN 值
                        best_fp = int(pr_df.iloc[best_f1_index_int]['FP'])
                        best_fn = int(pr_df.iloc[best_f1_index_int]['FN'])
                        best_tp = int(pr_df.iloc[best_f1_index_int]['TP'])
                        best_precision = pr_df.iloc[best_f1_index_int]['Precision']

                        # --- 计算 Sensitivity 和 Specificity ---
                        sens = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0  # Sensitivity (True Positive Rate)
                        spec = best_tn / (best_tn + best_fp) if (best_tn + best_fp) > 0 else 0  # Specificity (True Negative Rate)

                        # --- 格式化为保留 3 位小数的字符串 ---
                        sens_str = f"{sens:.3f}"
                        spec_str = f"{spec:.3f}"
                        roc_auc_str = f"{roc_auc:.3f}"
                        pr_auc_str = f"{pr_auc:.3f}"

                        # --- 创建包含当前模型结果的字典 ---
                        model_name = folder_name  # 提取模型名称
                        current_result = {
                            "Model": model_name_for_check,  # 保存去除数据集名称的模型名称
                            "Subfolder": subfolder_name,
                            "Stage": stg,  # 模型阶段
                            "Best F1-Score": best_f1,
                            "Precision": best_precision,
                            "Recall/Sens": sens_str,
                            "Spec": spec_str,
                            "PRCAUC": pr_auc_str,
                            "ROCAUC": roc_auc_str,
                            "TN": best_tn,
                            "FP": best_fp,
                            "FN": best_fn,
                            "TP": best_tp,
                            "Best Threshold": best_threshold,
                        }

                        # --- 将当前模型的结果添加到总表中 ---
                        current_df = pd.DataFrame([current_result])
                        # 定义目标列顺序
                        target_order = ["Model", "Subfolder", "Stage", "Best F1-Score", "Precision", "Recall/Sens", "Spec", "PRCAUC", "ROCAUC", "TN", "FP", "FN", "TP", "Best Threshold"]
                        current_df = current_df[target_order]  # 重新排列列

                        overall_df = pd.concat([overall_df, current_df], ignore_index=True)

                        # --- 按 'Model' 列排序 ---
                        overall_df = overall_df.sort_values(by="Model", ignore_index=True)

                        # --- 实时保存总表到 CSV 文件 ---
                        overall_df.to_csv(overall_results_path, index=False)

                        print(f"Current result for {folder_name}/{subfolder_name}/{stg} saved to {overall_results_path}")
                    
                    except KeyboardInterrupt:
                        print("\nScript interrupted by user.  Exiting gracefully...")
                        break # Exit the outer loop too