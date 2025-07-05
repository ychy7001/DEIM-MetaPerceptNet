import os
import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# path
base_dir = "/mnt/d/VsCode/Ckpts/yolov11/results/breast_bm_b-mode"

def calculate_patient_level_metrics(csv_path, threshold):
    """
    计算以病人为单位的 Recall, Specificity, Precision。

    Args:
        csv_path (str): raw_scores.csv 文件的路径。
        threshold (float): 用于判断图像是否为恶性的阈值。

    Returns:
        dict: 包含 Recall, Specificity, Precision 的字典
    """

    # 1. 读取 CSV 文件
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None

    # 2. 提取病人ID
    def extract_patient_id(filename):
        try:
            if '_' in filename:
                return filename.split('_')[0]  # 处理下划线命名方式
            elif '-' in filename:
                parts = filename.split('-')
                if len(parts) >= 2:
                    return parts[0] + '-' + parts[1]  # 处理横线命名方式
                else:
                    return None  # 文件名格式不正确，返回 None
            else:
                return None
        except:
            return None  # 处理其他可能出现的错误

    df['patient_id'] = df['filename'].apply(extract_patient_id)

    # 3. 移除 patient_id 为 None 的行
    df = df.dropna(subset=['patient_id'])

    # 4. 将标签转换为数值类型（如果不是）
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # 5. 计算以病人为单位的预测标签
    def patient_level_prediction(group):
        # 如果组内有任何图像的 score 大于阈值，则认为病人为恶性
        return int((group['score'] >= threshold).any())

    patient_predictions = df.groupby('patient_id').apply(patient_level_prediction)
    patient_predictions.name = 'predicted_label'  # 命名 Series

    # 6. 计算以病人为单位的实际标签
    def patient_level_label(group):
        # 病人只要有一张图是恶性，那么病人就是恶性
        return int((group['label'] == 1).any())

    patient_labels = df.groupby('patient_id').apply(patient_level_label)
    patient_labels.name = 'actual_label'  # 命名 Series

    # 7. 合并预测标签和实际标签
    patient_results = pd.concat([patient_labels, patient_predictions], axis=1)
    patient_results.reset_index(inplace=True)

    # 8. 计算 TP, TN, FP, FN
    tp = patient_results[(patient_results['actual_label'] == 1) & (patient_results['predicted_label'] == 1)].shape[0]
    tn = patient_results[(patient_results['actual_label'] == 0) & (patient_results['predicted_label'] == 0)].shape[0]
    fp = patient_results[(patient_results['actual_label'] == 0) & (patient_results['predicted_label'] == 1)].shape[0]
    fn = patient_results[(patient_results['actual_label'] == 1) & (patient_results['predicted_label'] == 0)].shape[0]

    # 9. 计算 Recall, Specificity, Precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {"recall": recall, "specificity": specificity, "precision": precision}


# --- 使用示例 ---
if __name__ == "__main__":
    best_results = []

    # 1. 自动查找包含 raw_scores.csv 的 Model, Subfolder, 和 Stage
    model_subfolder_stage_data = []
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            for subfolder in os.listdir(model_path):
                subfolder_path = os.path.join(model_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for stage in os.listdir(subfolder_path):  # 遍历 stg1 和 stg2
                        stage_path = os.path.join(subfolder_path, stage)
                        if os.path.isdir(stage_path) and "raw_scores.csv" in os.listdir(stage_path):
                            model_subfolder_stage_data.append({'Model': model, 'Subfolder': subfolder, 'Stage': stage})

    model_subfolder_stage_df = pd.DataFrame(model_subfolder_stage_data) # 从自动检索的结果创建 DataFrame

    # 使用 tqdm 显示文件处理进度
    for index, row in tqdm(model_subfolder_stage_df.iterrows(), total=len(model_subfolder_stage_df), desc="Processing files"):
        model = row['Model']
        subfolder = row['Subfolder']
        stage = row['Stage']

        # 2. 构建 raw_scores.csv 文件的路径
        csv_file = os.path.join(base_dir, model, subfolder, stage, "raw_scores.csv")

        # 确定输出目录
        output_dir = os.path.dirname(csv_file)

        # 0. 检查是否已经计算过
        output_csv = os.path.join(output_dir, "patient_level_metrics.csv")
        #if os.path.exists(output_csv): # 注释掉，因为我们希望即使存在也重新计算
        #    print(f"Skipping {csv_file}: patient_level_metrics.csv already exists")
        #    continue

        # 1. 从 pr_curve_data.csv 文件中读取阈值
        pr_curve_data_path = os.path.join(output_dir, "pr_curve_data.csv")
        try:
            pr_curve_df = pd.read_csv(pr_curve_data_path)
            min_threshold = pr_curve_df["Threshold"].min()
            max_threshold = pr_curve_df["Threshold"].max()
            num_thresholds = min(30, len(pr_curve_df["Threshold"])) # 确保不超过实际阈值数量
            thresholds = np.linspace(min_threshold, max_threshold, num_thresholds).tolist()
        except FileNotFoundError:
            print(f"Error: pr_curve_data.csv not found at {pr_curve_data_path}")
            continue  # 跳过当前文件夹


        # 2. 针对每个阈值计算指标并保存
        results = []
        # 使用 tqdm 显示阈值处理进度
        for threshold in tqdm(thresholds, desc=f"Processing thresholds for {os.path.basename(csv_file)}"):
            metrics = calculate_patient_level_metrics(csv_file, threshold)
            if metrics:
                results.append({
                    "threshold": threshold,
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "precision": metrics["precision"]
                })

        # 3. 将结果保存到 CSV 文件 (一次性写入)
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")

            # 4. 寻找最佳表现
            results_df['Weighted_metric'] = results_df['recall'] * 2 + results_df['specificity']
            best_row = results_df.loc[results_df['Weighted_metric'].idxmax()]
            best_results.append({
                'Model': model, # 添加 Model
                'Subfolder': subfolder, # 添加 Subfolder
                'Stage': stage,  # 添加 Stage
                'Threshold': best_row['threshold'],
                'Recall': best_row['recall'],
                'Specificity': best_row['specificity'],
                'Precision': best_row['precision'],
                'Weighted_metric': best_row['Weighted_metric']
            })
        else:
            print(f"No results to save for {csv_file}")

    # 5. 将最佳结果保存到 breast_bm_b-mode 目录下
    best_results_csv = os.path.join(base_dir, "best_patient_level_metrics.csv")
    if os.path.exists(best_results_csv):
        # 如果文件存在，则读取现有数据
        try:
            existing_df = pd.read_csv(best_results_csv)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()  # 如果文件为空，则创建一个空DataFrame
    else:
        existing_df = pd.DataFrame()

    # 创建一个包含新结果的 DataFrame
    new_results_df = pd.DataFrame(best_results)

    # 合并新结果和现有 DataFrame
    merged_df = pd.concat([existing_df, new_results_df], ignore_index=True)

    # 删除重复的 Model/Subfolder/Stage 组合，保留 Weighted_metric 最高的
    merged_df = merged_df.sort_values('Weighted_metric', ascending=False).drop_duplicates(['Model', 'Subfolder', 'Stage']).sort_index()

    best_results_df = merged_df


    # 强制转换 'Model', 'Subfolder', 和 'Stage' 列为字符串类型
    best_results_df['Model'] = best_results_df['Model'].astype(str)
    best_results_df['Subfolder'] = best_results_df['Subfolder'].astype(str)
    best_results_df['Stage'] = best_results_df['Stage'].astype(str)

    if not best_results_df.empty:
        # 在保存之前，检查 Model/Subfolder/Stage 是否仍然存在
        def is_valid_path(row):
            csv_path = os.path.join(base_dir, row['Model'], row['Subfolder'], row['Stage'], "raw_scores.csv")
            return os.path.exists(csv_path)

        valid_rows = best_results_df.apply(is_valid_path, axis=1)
        best_results_df = best_results_df[valid_rows]

        # 排序
        best_results_df = best_results_df.sort_values(
            by=['Model', 'Subfolder', 'Stage', 'Weighted_metric'],
            ascending=[True, True, True, False]
        )
        # 保存到 CSV 文件
        best_results_df.to_csv(best_results_csv, index=False)
        print(f"Best results saved to: {best_results_csv}")
    else:
        print("No best results to save.")