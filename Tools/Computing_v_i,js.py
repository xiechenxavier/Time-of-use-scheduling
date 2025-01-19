import json
import os
import numpy as np

def calculate_and_store_v(folder_path, alpha):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            # Extract the instance size (m, n) from the file name
            file_path = os.path.join(folder_path, file_name)
            
            # Load the existing mu_{i,j} and sigma_{i,j} data from the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            mu_values = np.array(data["mu_{i,j}"])
            sigma_values = np.array(data["sigma_{i,j}"])
            m, n = mu_values.shape  # Get m and n from the shape of mu_{i,j}
            
            # Step 1: Calculate beta
            beta = 1 - np.power(1 - alpha, 1/n)
            
            # Step 2: Calculate v_{i,j}_1
            v_ij_1 = sigma_values * np.sqrt((1 - beta) / beta)
            v_ij_1 = np.rint(v_ij_1).astype(int)
            # Step 3: Calculate v_{i,j}_2
            v_ij_2 = sigma_values * np.sqrt((4 / (9 * beta)) - 1)
            v_ij_2 = np.rint(v_ij_2).astype(int)
            
            # Step 4: Store (v_{i,j}_1, v_{i,j}_2) in the original JSON file
            data["v_{i,j}_1"] = v_ij_1.tolist()  # Convert arrays to lists for JSON compatibility
            data["v_{i,j}_2"] = v_ij_2.tolist()
            
            # Write the updated data back to the JSON file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

# Example usage:
folder_path = "E:/ToU重做3/dict_experiment_data/D_0.04/alpha=0.2"  # Replace with your actual folder path
alpha_value = 0.2  # Example alpha value
calculate_and_store_v(folder_path, alpha_value)

def update_power_consumption(source_folder, target_folder):
    # 获取源文件夹中所有的json文件
    files = [f for f in os.listdir(source_folder) if f.endswith('.json')]
    
    for filename in files:
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)
        
        # 检查目标文件是否存在
        if not os.path.exists(target_file):
            print(f"目标文件不存在：{target_file}")
            continue
        
        # 读取源文件的内容到dict1
        with open(source_file, 'r', encoding='utf-8') as f:
            dict1 = json.load(f)
        
        # 读取目标文件的内容到dict2
        with open(target_file, 'r', encoding='utf-8') as f:
            dict2 = json.load(f)
        
        # 更新dict2中的'Power Consumptions'
        power_consumption = dict1.get('Power Consumptions', 0)
        dict2['Power Consumptions'] = power_consumption
        for i in range(len(dict2['Power Consumptions'])):
            dict2['Power Consumptions'][i] = round(dict2['Power Consumptions'][i], 2)
        
        # 将dict2写回目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(dict2, f, ensure_ascii=False, indent=4)

        print(f"已更新文件：{target_file}")

# 示例用法
source_folder = 'E:/ToU重做3/dict_experiment_data/expected_value'  # 替换为实际的源文件夹路径
target_folder = 'E:/ToU重做3/dict_experiment_data/D_0.04/alpha=0.2'  # 替换为实际的目标文件夹路径

update_power_consumption(source_folder, target_folder)