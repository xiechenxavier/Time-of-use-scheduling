import numpy as np
import os
import json

# 参数设定
machine_counts = [9, 10, 11, 12]
job_ratios = [10, 11, 12]
mu_range = (20, 90)  # 平均处理时间范围
consumption_range = (0.05, 0.55)  # 功耗范围
d_q_values = [0.01, 0.04]  # 方差 D[Q_{i,j}] 的值

# 创建数据主目录
main_dir = 'E:/ToU重做3/dict_experiment_data/expected_value/Big_expected/'  # 更新为你存储的路径
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

max_m = 10

def Generate_exp_data(max_m, main_dir):
    # 生成功耗数据，用于最大实例的 m=12

    power_consumptions = np.sort(np.random.uniform(consumption_range[0], consumption_range[1], max_m))

    # 遍历每个 (m, n) 实例
    for m in machine_counts:
        for ratio in job_ratios:
            n = m * ratio
            
            # 为当前实例生成 mu_{i,j} 数据
            mu_values = np.random.randint(mu_range[0], mu_range[1] + 1, (m, n))
            
            # 按照从小到大的顺序，从功耗数据中随机选择 m 个功耗值
            selected_consumptions = np.random.choice(power_consumptions, m, replace=False)
            
            # 将 mu_{i,j} 和功耗数据保存为字典格式
            instance_dict = {
                "mu_{i,j}": mu_values.tolist(),
                "Power Consumptions": selected_consumptions.tolist()
            }
            
            # 保存到 JSON 文件
            instance_file = os.path.join(main_dir, f'{m}_{n}.json')
            with open(instance_file, 'w') as f:
                json.dump(instance_dict, f, indent=4)

# 生成数据
# Generate_exp_data(max_m, main_dir)

def Generate_standard_dev(d_q,main_dir):
    # 创建每个 D[Q_{i,j}] 的文件夹
    d_q_dir = os.path.join(main_dir, f'D_{d_q:.2f}')
    if not os.path.exists(d_q_dir):
        os.makedirs(d_q_dir)
    
    # 再次遍历所有 (m, n) 实例
    for m in machine_counts:
        for ratio in job_ratios:
            n = m * ratio
            # 重新读取之前生成的 mu_{i,j} 数据
            instance_file = os.path.join(main_dir, f'{m}_{n}.json')
            with open(instance_file, 'r') as f:
                instance_data = json.load(f)
            
            mu_values = np.array(instance_data["mu_{i,j}"])
            
            # 计算 sigma_{i,j}，并转换为整数
            sigma_values = np.rint(mu_values * np.sqrt(d_q)).astype(int)
            
            # 将 mu_{i,j} 和 sigma_{i,j} 保存为字典格式
            instance_dict_dq = {
                "mu_{i,j}": mu_values.tolist(),
                "sigma_{i,j}": sigma_values.tolist(),
                "Power Consumptions": [round(x, 2) for x in instance_data["Power Consumptions"]]
            }
            
            # 保存到每个 D[Q_{i,j}] 文件夹中
            d_q_instance_file = os.path.join(d_q_dir, f'{m}_{n}.json')
            with open(d_q_instance_file, 'w') as f:
                json.dump(instance_dict_dq, f, indent=4)

# 生成数据
Generate_standard_dev(d_q_values[0],main_dir)