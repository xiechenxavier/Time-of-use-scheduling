from gurobipy import Model, GRB, quicksum
import numpy as np
import json
import os
import time
import math


def extract_data_from_file(file_path, nb_K):
    # Step 1: Extract m and n from the filename
    file_name = os.path.basename(file_path)  # Get the file name (m_n.json)
    m, n = map(int, file_name.replace(".json", "").split("_"))  # Extract m and n from the file name
    
    # Step 2: Read the JSON file content
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Step 3: Extract relevant arrays from the dictionary
    p = data.get("mu_{i,j}", [])  # Extract mu_{i,j}
    q = data.get("Power Consumptions", [])  # Extract q[i] (Power Consumption)
    v_ij_1 = data.get("v_{i,j}_1", [])  # Extract v_{i,j}_1
    v_ij_2 = data.get("v_{i,j}_2", [])  # Extract v_{i,j}_2

    K = nb_K
    if K == 5:
        T = [60,360,360,180,480]
        R = [0.0045,0.0033,0.0045,0.0033,0.0045]
    elif K == 3:
        T = [390,960,90]
        R = [0.0033, 0.0045, 0.0033]
    else:
        print("Invalid number of time intervals")
    # 
    S = [sum(T[:k]) for k in range(K)] # 时间间隔的开始时间
    B = 1440
    
    # Return the extracted values
    return m, n, p, q, v_ij_1, v_ij_2, T, R, S, B, K

# 读取数据
file_path = "E:/ToU重做3/dict_experiment_data/D_0.01/8_56.json"
m, n, p, q, v_ij_1, v_ij_2, T, R, S, B, K = extract_data_from_file(file_path,5)
T_min = min(T)
T_max = max(T)
R_min = min(R)
R_max = max(R)
K_low = [k for k in range(K) if T[k] <= T_min]
K_high = [k for k in range(K) if T[k] >= T_max]
Delta = math.ceil(sum([T[k] for k in K_high])/sum([T[k] for k in K_low]))
t_indices = [(i, j, k) for i in range(m) for j in range(n) for k in range(K)]
p_v_ij_2 = [[p[i][j] + min(v_ij_1[i][j],v_ij_2[i][j]) for j in range(n)] for i in range(m)]


def Configure_model(model, y, t, w, z, C_max, p):
    model.addConstrs((quicksum(y[i, j] for i in range(m)) == 1 for j in range(n)), name="AssignJob")
    model.addConstrs((quicksum(t[i, j, k] for k in range(K)) == p[i][j] * y[i, j] for i in range(m) for j in range(n)), name="ProcTime")
    # model.addConstrs((quicksum(t[i, j, k] for j in range(n)) <= T[k] for i in range(m) for k in range(K)), name="machineCapacity")
    model.addConstrs((quicksum(t[i, j, k] for i in range(m)) <= T[k] * w[j, k] for j in range(n) for k in range(K)), name="Link_w_t")
    model.addConstrs((quicksum(t[i,j,k] for i in range(m)) >= T[k]*(w[j, k - 1] + w[j, k + 1] - 1) for j in range(n) for k in range(1, K-1)), name="Continuity") # to change
    # model.addConstrs((w[j, k] >= (1 / T[k]) * quicksum(t[i, j, k] for i in range(m)) for j in range(n) for k in range(K)), name="Link_w_t2")
    model.addConstrs((quicksum(w[j, kp] for kp in range(K) if kp > k + 1) <= (K - k - 1) * (1 - w[j, k] + w[j, k + 1]) for j in range(n) for k in range(K-2)), name="ProcessingOrder")
    model.addConstrs((z[i, k] >= (1 / T[k]) * quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="Link_z_t")
    model.addConstrs((C_max >= z[i, k] * S[k] + quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="makespan")

max_val = 1e10

def optimize_constrained_TEC(model,epsilon2,drop_constraint_name,C_max,TEC):
    # model.clear()
    model.setObjective(TEC, GRB.MINIMIZE)
    if drop_constraint_name is not None: # 如果已经有了约束，先删除再添加
        model.remove(model.getConstrByName(drop_constraint_name))
        model.addConstr(C_max <= epsilon2, "const_Cmax")
    elif epsilon2 < max_val-1:
        model.addConstr(C_max <= epsilon2, "const_Cmax")
    else:
        pass
    model.update()

def optimize_constrained_Cmax(model,epsilon,drop_constraint_name2,C_max,TEC):
    model.setObjective(C_max, GRB.MINIMIZE)
    if drop_constraint_name2 is not None: # 如果已经有了约束，先删除再添加
        model.remove(model.getConstrByName(drop_constraint_name2))
        model.addConstr(TEC <= epsilon, "const_TEC")
    elif epsilon < max_val-1:
        model.addConstr(TEC <= epsilon, "const_TEC")
    else:
        pass
    model.update()

def optimize_constrained_Cmax2(model,epsilon,C_max,TEC):
    model.setObjective(C_max, GRB.MINIMIZE)
    model.addConstr(TEC <= epsilon, "const_TEC")
    model.update()

def optimize_constrained_TEC2(model,epsilon2,C_max,TEC):
    model.setObjective(TEC, GRB.MINIMIZE)
    model.addConstr(C_max <= epsilon2, "const_Cmax")
    model.update()

def store_solution(y,t,w,z):
    var_values = {
        'y': {},  # To store y_{i,j} variables
        'w': {},  # To store w_{j,k} variables
        't': {},  # To store t_{i,j,k} variables
        'z': {},  # To store z_{i,k} variables
        }
    # Store y_{i,j} variables
    for i in range(m):
        for j in range(n):
            y_value = y[i, j].X
            if abs(y_value) > 1e-6:
                var_values['y'][(i, j)] = y_value

    # Store w_{j,k} variables
    for j in range(n):
        for k in range(K):
            w_value = w[j, k].X
            if abs(w_value) > 1e-6:
                var_values['w'][(j, k)] = w_value

    # Store t_{i,j,k} variables
    for i in range(m):
        for j in range(n):
            for k in range(K):
                t_value = t[i, j, k].X
                if abs(t_value) > 1e-6:
                    var_values['t'][(i, j, k)] = t_value

    # Store z_{i,k} variables
    for i in range(m):
        for k in range(K):
            z_value = z[i, k].X
            if abs(z_value) > 1e-6:
                var_values['z'][(i, k)] = z_value
        # solution_list.append(var_values)
    return var_values

def add_valid_inequalities(model,y,t,w,C_max,TEC,p,obj):
    valid_inequalities = []
    if obj == "TEC":
        # 添加TEC的有效不等式
        TEC_vli1 = model.addConstrs( quicksum(t[i,j,k] for i in range(m) ) 
                            >= quicksum(p[i][j]*y[i,j] * (w[j,k] - quicksum(w[j,k_prime] for k_prime in range(k,K)) )
                                        - quicksum(t[i,j,k_prime] for k_prime in range(0,k))
                                        for i in range(m))
                            for j in range(n) for k in range(1,K-1))
        TEC_vli2 = model.addConstr(TEC <= quicksum(q[i] * R_max * t[i, j, k] for (i, j, k) in t_indices))
        TEC_vli3 = model.addConstr(TEC >= quicksum(q[i] * R_min * t[i, j, k] for (i, j, k) in t_indices))
        valid_inequalities = [TEC_vli1,TEC_vli2,TEC_vli3]
    elif obj == "Cmax":
        # 添加Cmax的有效不等式
        C_max_vli1 = model.addConstrs( quicksum(t[i,j,k] for j in range(n) for k in range(K)) <= C_max for i in range(m))
        C_max_vli2 = model.addConstrs( C_max 
                        >= quicksum(p[i][j] * y[i, j] for j in range(n))  for i in range(m))
        C_max_vli3 = model.addConstr( C_max 
                        >= 1/m * quicksum(p[i][j] * y[i, j] for j in range(n) for i in range(m)) )
        valid_inequalities = [C_max_vli1,C_max_vli2,C_max_vli3]
    return valid_inequalities

def Remove_valid_inequalities(model,valid_inequalities):
    for constr in valid_inequalities:
        model.remove(constr)

def epsilon_constraint_solve(model,y,t,w,z,C_max,TEC,p):
    gaps = []
    # compute minimum TEC
    Configure_model(model, y, t, w, z, C_max, p)
    
    # add TEC valid inequalities
    TEC_vlis = add_valid_inequalities(model,y,t,w,C_max,TEC,p,"TEC")
    optimize_constrained_TEC(model,max_val,None,C_max,TEC) 
    model.optimize()
    min_TEC = model.objVal
    corresponding_Cmax = C_max.X

    # drop TEC valid inequalities
    Remove_valid_inequalities(model,TEC_vlis)
    # add Cmax valid inequalities
    # C_max_vlis = add_valid_inequalities(model,y,t,w,C_max,TEC,p,"Cmax")
    # compute minimum Cmax
    optimize_constrained_Cmax(model,max_val,None,C_max,TEC)
    model.optimize()
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            gaps.append(model.MIPGap)
    min_Cmax = model.objVal
    corresponding_TEC = TEC.getValue()
    # compute epsilon
    epsilon = min_TEC
    epsilon2 = min_Cmax
    f1 = f2 = 0
    z_lu = [0,0]
    z_rl = [0,0]
    # solve the problem
    Pareto_set = set()
    solution_list = []
    
    # remove Cmax valid inequalities
    # Remove_valid_inequalities(model,C_max_vlis)
    # add TEC valid inequalities
    TEC_vlis = add_valid_inequalities(model,y,t,w,C_max,TEC,p,"TEC")
    optimize_constrained_TEC(model,epsilon2,None,C_max,TEC) # 添加TEC的有效不等式
    model.optimize()
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount >0:
            gaps.append(model.MIPGap)
            f1 = round(model.objVal,4)
            f2 = round(C_max.X,4)
            z_rl[0] = f1
            z_rl[1] = f2
            Pareto_set.add(tuple(z_rl))
            solution = store_solution(y,t,w,z)
            solution_list.append(solution)
        else:
            z_rl[0] = corresponding_TEC
            z_rl[1] = min_Cmax
            Pareto_set.add(tuple(z_rl))
    # optimize_constrained_Cmax(model,epsilon,'const_Cmax',C_max,TEC) # 除去TEC有效不等式, 添加Cmax的有效不等式
    model.remove(model.getConstrByName("const_Cmax"))

    # remove TEC valid inequalities
    Remove_valid_inequalities(model,TEC_vlis)

    # add Cmax valid inequalities
    # C_max_vlis = add_valid_inequalities(model,y,t,w,C_max,TEC,p,"Cmax")
    model.addConstr(TEC <= epsilon, "const_TEC")
    model.setObjective(C_max, GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount >0:
            gaps.append(model.MIPGap)
            f1 = round(TEC.getValue(),4)
            f2 = round(model.objVal,4)
            z_lu[0] = f1
            z_lu[1] = f2
            Pareto_set.add(tuple(z_lu))
            solution = store_solution(y,t,w,z)
            solution_list.append(solution)
        else:
            z_lu[0] = min_TEC
            z_lu[1] = corresponding_Cmax
            Pareto_set.add(tuple(z_lu))
    print("z_lu 左上角: ",z_lu) # f1 最小
    print("z_rl 右下角: ",z_rl) # f2 最小
    # remove Cmax valid inequalities
    # Remove_valid_inequalities(model,C_max_vlis)
    model.remove(model.getConstrByName("const_TEC"))

    epsilon = z_lu[1] - 1 # 逐渐减小epsilon，限制f1 TEC,最小化Cmax
    decrement = 26
    f1_pre = z_lu[1]
    Updatable_constr = model.addConstr(C_max <= epsilon)
    # add TEC valid inequalities
    # model.Params.MIPGap = 0
    # f2_pre = z_rl[1]
    while epsilon > min_Cmax:
        # optimize_constrained_Cmax2(model,epsilon,C_max,TEC) 
        Updatable_constr.setAttr(GRB.Attr.RHS,epsilon) # constraint min_TEC to minimize Cmax
        model.setObjective(TEC, GRB.MINIMIZE)
        # for i in range(m):
        #     constrs_Cmax[i].setAttr(GRB.Attr.RHS,1.1 * epsilon)
        model.optimize()
        if model.status == GRB.Status.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.solCount > 0:
                if model.status == GRB.TIME_LIMIT:
                        gaps.append(model.MIPGap) # 记录gap
                f1 = round(model.objVal,4)
                f2 = round(C_max.X,4)
                point = (f1,f2)
                Pareto_set.add(point)
                solution = store_solution(y,t,w,z)
                solution_list.append(solution)
                gap = abs(f1_pre - f1)
                if gap < 31: # the region is dense.
                    decrement = decrement*0.5
                else:
                    decrement = decrement*1.5
        epsilon -= decrement
    
    if len(gaps) == 0:
        return Pareto_set,solution_list,0
    else:
        return Pareto_set,solution_list,sum(gaps)/len(gaps)

def temp(model,pr,q):
    # 初始化模型
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")
    t = model.addVars(m, n, K, vtype=GRB.CONTINUOUS, name="t")
    w = model.addVars(n, K, vtype=GRB.BINARY, name="w")
    z = model.addVars(m, K, vtype=GRB.BINARY, name="z")
    C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
    t_indices = [(i, j, k) for i in range(m) for j in range(n) for k in range(K)]
    TEC = quicksum(q[i] * R[k] * t[i, j, k] for (i, j, k) in t_indices)
    #解决问题
    time_start = time.time()
    Pareto_set,solution_list,avg_gap = epsilon_constraint_solve(model,y,t,w,z,C_max,TEC,pr)
    time_end = time.time()

    # print("time: ",time_end - time_start)  
    time_cost = time_end - time_start

    return sorted(Pareto_set),time_cost, solution_list,avg_gap


p_v_ij_2 = [[p[i][j] + min(v_ij_1[i][j],v_ij_2[i][j]) for j in range(n)] for i in range(m)]

model2 = Model("ToU")
model2.Params.TimeLimit = 600
pareto_set2,time_cost2,solution_list2,avg_gap = temp(model2,p_v_ij_2,q)

# 读取数据
def load_dataset_from_file(filename):
    """
    Load a dataset from a file.
    
    :param filename: Name of the file to load the dataset from.
    :return: The dataset as a Python dictionary.
    """
    with open(filename, 'r') as file:
        dataset = json.load(file)
    return dataset

result_file = "E:/ToU重做3/dict_experiment_data/Tools/Valid_inequalities_tests/V2_Bio_results_D_0.01/PolicyA/8_56_result.json"
dataset = load_dataset_from_file(result_file)

dataset["Time_vi1,2,3"] = time_cost2
dataset["Gap_vi1,2,3"] = avg_gap
# dataset["Time_vis_plus"] = time_cost2
# dataset["Gap_vis_plus"] = avg_gap
print("P2_time_cost: ",dataset["time_cost"])
# print("Time_vi1,2,3: ",dataset["Time_vi1,2,3"])
print("time_cost2: ",time_cost2)
# Save the updated dataset to a file
with open(result_file, 'w') as file:
    json.dump(dataset, file, indent=4)

# dict_result = {
#     "pareto_set": pareto_set2,
#     "time_cost": time_cost2,
#     "solution_list": solution_list2,
#     "avg_gap": avg_gap
# }

# def convert_np_types(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()
#     raise TypeError

# # Convert all keys to string format
# def convert_keys_to_str(d):
#     if isinstance(d, dict):
#         return {str(k): convert_keys_to_str(v) for k, v in d.items()}
#     elif isinstance(d, list):
#         return [convert_keys_to_str(v) for v in d]
#     else:
#         return d

# # Convert the dictionary with tuple keys to use string keys
# dict_result_str_keys = convert_keys_to_str(dict_result)
# # print(dict_result_str_keys)
# result_file = "E:/ToU重做3/dict_experiment_data/Tools/Valid_inequalities_tests/V2_Bio_results_D_0.01/PolicyA/8_56_result.json"
# # Save to JSON file
# with open(result_file, 'w') as json_file:
#     json.dump(dict_result_str_keys, json_file, default=convert_np_types, indent=4)