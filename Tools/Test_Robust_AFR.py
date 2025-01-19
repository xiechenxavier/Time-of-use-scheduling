# 我们需要一个函数计算所有文件数据中存储解的鲁棒性
import ast
import json
import os

def verify_solution_robustness(p_sample, sol_list):
    passed_count = 0

    for solution in sol_list:
        t_dict = solution['t']

        # 创建一个字典，存储累加后的 t 值
        t_accumulated = {}
        for key, value in t_dict.items():
            # 解析键为 (i, j, k)，支持多种格式
            if isinstance(key, str):
                i, j, k = ast.literal_eval(key)
            else:
                i, j, k = key

            if (i, j) not in t_accumulated:
                t_accumulated[(i, j)] = 0
            t_accumulated[(i, j)] += value

        # 检查 t_accumulated 中存在的 (i, j)，并与 p_sample 比较
        feasible = True
        for (i, j), t_sum in t_accumulated.items():
            if t_sum > 0:  # 仅检查 t_accumulated 中值不为 0 的项
                if p_sample[i][j] > t_sum:
                    print(f"Failed at (i, j) = ({i}, {j}), p_sample[{i}][{j}] = {p_sample[i][j]}, t_accumulated[{i}, {j}] = {t_sum}")
                    feasible = False
                    break
        
        # 如果 solution 通过所有验证，计入通过数量
        if feasible:
            passed_count += 1

    # 计算可行率
    feasibility_rate = passed_count / len(sol_list) if sol_list else 0
    return feasibility_rate

# def verify_solution_robustness(p_sample, sol_list):
#     passed_count = 0
#     for solution in sol_list:
#         t_dict = solution['t']

#         # 创建一个字典，存储累加后的t值，累积每个(i, j)对应的sum_{k} t_{i,j,k}
#         t_accumulated = {}
#         for key, value in t_dict.items():
#             # 将字符串形式的元组转换为实际元组
#             i, j, k = ast.literal_eval(key)
#             if (i, j) not in t_accumulated:
#                 t_accumulated[(i, j)] = 0
#             t_accumulated[(i, j)] += value

#         # 检查p_sample中的每个p_{i, j}是否小于或等于t_accumulated中的累积t值
#         feasible = True
#         for (i, j), t_sum in t_accumulated.items():
#             if t_sum > 0:  # 仅检查 t_accumulated 中值不为 0 的项
#                 if p_sample[i][j] > t_sum:
#                     print(f"Failed at (i, j) = ({i}, {j}), p_sample[{i}][{j}] = {p_sample[i][j]}, t_accumulated[{i}, {j}] = {t_sum}")
#                     feasible = False
#                     break
        
        
#         # 如果solution通过了所有p_samples的验证，则累加通过数量
#         if feasible:
#             passed_count += 1

#     # 计算一组p_sample的可行率
#     feasibility_rate = passed_count / len(sol_list)
        
#     return feasibility_rate

        
# Extract solutions from a JSON file
def extract_sols_from_file(file_path,sol_name):
    # Read the JSON file content
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[sol_name]

# Extract p_sample from a JSON file
def extract_p_sample_from_file(file_path):
    # Read the JSON file content
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["samples"]

# Average_feasibility_rate 
def compute_average_feasibility_rate(sol_list, p_samples):
    total_feasibility_rate = 0
    for p_sample in p_samples:
        total_feasibility_rate += verify_solution_robustness(p_sample, sol_list)
    return total_feasibility_rate / len(p_samples)

#     P2_feasibility_rate1 = compute_average_feasibility_rate(sol_list1, p_samples)
#     P2_feasibility_rate2 = compute_average_feasibility_rate(sol_list2, p_samples)
#     P2_feasibility_rate3 = compute_average_feasibility_rate(sol_list3, p_samples)
#     return P2_feasibility_rate1, P2_feasibility_rate2, P2_feasibility_rate3

# function to return results for avg vij rates and time for three alphas
def compute_results(file_path1, file_path2, file_path3, p_samples,
                                  P2_sols = "P2_solutions",sol_name1="rate_vij1_avg", 
                                  sol_name2="rate_vij2_avg", sol_name3="P2_time"):
    # Read the JSON file content
    with open(file_path1, 'r') as f:
        data1 = json.load(f)
    with open(file_path2, 'r') as f:
        data2 = json.load(f)
    with open(file_path3, 'r') as f:
        data3 = json.load(f)
    sol_list1, sol_list2, sol_list3 = data1[P2_sols], data2[P2_sols], data3[P2_sols]
    P2_feasibility_rate1 = compute_average_feasibility_rate(sol_list1, p_samples)
    P2_feasibility_rate2 = compute_average_feasibility_rate(sol_list2, p_samples)
    P2_feasibility_rate3 = compute_average_feasibility_rate(sol_list3, p_samples)
    avg_rate_vij1_1,avg_rate_vij2_1 = data1[sol_name1], data1[sol_name2]
    avg_rate_vij1_2,avg_rate_vij2_2 = data2[sol_name1], data2[sol_name2]
    avg_rate_vij1_3,avg_rate_vij2_3 = data3[sol_name1], data3[sol_name2]
    time_P2_1,time_P2_2, time_P2_3 = data1[sol_name3], data2[sol_name3], data3[sol_name3]
    data = { "P2_feasibility_rate1": P2_feasibility_rate1,
            "P2_feasibility_rate2": P2_feasibility_rate2,
            "P2_feasibility_rate3": P2_feasibility_rate3,
            "avg_rate_vij1_1": avg_rate_vij1_1,
            "avg_rate_vij2_1": avg_rate_vij2_1,
            "avg_rate_vij1_2": avg_rate_vij1_2,
            "avg_rate_vij2_2": avg_rate_vij2_2,
            "avg_rate_vij1_3": avg_rate_vij1_3,
            "avg_rate_vij2_3": avg_rate_vij2_3,
            "time_P2_1": time_P2_1,
            "time_P2_2": time_P2_2,
            "time_P2_3": time_P2_3
            }
    return data
    

# P2_sols = "P2_solutions"
# result_path1 = "E:/ToU重做3/dict_experiment_data/D_0.01/results/PolicyA/alpha=0.1/5_30_result.json"
# result_path2 = "E:/ToU重做3/dict_experiment_data/D_0.01/results/PolicyA/alpha=0.2/5_30_result.json"
# result_path3 = "E:/ToU重做3/dict_experiment_data/D_0.01/results/PolicyA/alpha=0.3/5_30_result.json"
# sample_path = "E:/ToU重做3/dict_experiment_data/D_0.01/Generate_samples/5_30_samples.json"
# p_samples = extract_p_sample_from_file(sample_path)
# # import numpy as np
# # print(np.array(p_samples).shape)
# data = compute_results(result_path1, result_path2, result_path3, p_samples)
# print(data)


def batch_process(base_path, scenarios, policies, result_filename_pattern1,result_filename_pattern2,result_filename_pattern3
                  ,sample_filename_pattern, output_filename_pattern):
    for scenario in scenarios:
        for policy in policies:
            # 设置路径
            result_path1 = os.path.join(base_path, result_filename_pattern1.format(scenario=scenario, policy=policy))
            result_path2 = os.path.join(base_path, result_filename_pattern2.format(scenario=scenario, policy=policy))
            result_path3 = os.path.join(base_path, result_filename_pattern3.format(scenario=scenario, policy=policy))
            sample_path = os.path.join(base_path, sample_filename_pattern.format(scenario=scenario))
            output_filename = os.path.join(base_path, output_filename_pattern.format(scenario=scenario, policy=policy))

            # 检查文件是否存在
            if not os.path.exists(result_path1) or not os.path.exists(sample_path):
                print(f"Warning: Missing files for scenario {scenario} and policy {policy}")
                continue
            
            if not os.path.exists(result_path2) or not os.path.exists(sample_path):
                print(f"Warning: Missing files for scenario {scenario} and policy {policy}")
                continue

            if not os.path.exists(result_path3) or not os.path.exists(sample_path):
                print(f"Warning: Missing files for scenario {scenario} and policy {policy}")
                continue

            # 提取数据
            p_samples = extract_p_sample_from_file(sample_path)

            # 计算可行率
            data = compute_results(result_path1, result_path2, result_path3, p_samples)

            # 保存结果
            # with open(output_filename, 'w') as json_file:
            #     json.dump(data, json_file, indent=4)
            print(data)

if __name__ == "__main__":
    base_path = "E:/ToU重做3/dict_experiment_data/D_0.04"
    # scenarios = ["3_12", "3_15", "3_18","5_20", "5_25", "5_30","6_24", "6_30", "6_36"]  # 你可以在这里添加更多场景
    # scenarios = ["7_49", "7_56", "7_63","8_56", "8_64", "8_72","9_63", "9_72", "9_81"]
    scenarios = ["6_24"]
    policies = ["PolicyB"]  # 你可以在这里添加更多策略

    result_filename_pattern1 = "results/{policy}/alpha=0.1/{scenario}_result.json"
    result_filename_pattern2 = "results/{policy}/alpha=0.2/{scenario}_result.json"
    result_filename_pattern3 = "results/{policy}/alpha=0.3/{scenario}_result.json"
    sample_filename_pattern = "Generate_samples/{scenario}_samples.json"
    # output_filename_pattern = "Table_data/{policy}/50samples/{scenario}.json"
    output_filename_pattern = "Table_data/{policy}/newtable/{scenario}.json"

    batch_process(base_path, scenarios, policies, result_filename_pattern1, result_filename_pattern2, 
                  result_filename_pattern3, sample_filename_pattern, output_filename_pattern)
