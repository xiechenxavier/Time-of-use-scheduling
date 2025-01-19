from gurobipy import Model, GRB, quicksum
import numpy as np
import time
import os
import json

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
file_path = "E:/ToU重做3/dict_experiment_data/D_0.01/alpha=0.1/9_72.json"
m, n, p, q, v_ij_1, v_ij_2, T, R, S, B, K = extract_data_from_file(file_path,3)

    
max_val = 1e10


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

def optimize_constrained_TEC(model,epsilon2,drop_constraint_name,C_max,total_electricity_cost):
    # model.clear()
    model.setObjective(total_electricity_cost, GRB.MINIMIZE)
    if drop_constraint_name is not None: # 如果已经有了约束，先删除再添加
        model.remove(model.getConstrByName(drop_constraint_name))
        model.addConstr(C_max <= epsilon2, "const_Cmax")
    elif epsilon2 < max_val-1:
        model.addConstr(C_max <= epsilon2, "const_Cmax")
    else:
        pass
    model.update()

def optimize_constrained_Cmax(model,epsilon,drop_constraint_name2,C_max,total_electricity_cost):
    model.setObjective(C_max, GRB.MINIMIZE)
    if drop_constraint_name2 is not None: # 如果已经有了约束，先删除再添加
        model.remove(model.getConstrByName(drop_constraint_name2))
        model.addConstr(total_electricity_cost <= epsilon, "const_TEC")
    elif epsilon < max_val-1:
        model.addConstr(total_electricity_cost <= epsilon, "const_TEC")
    else:
        pass
    model.update()

def optimize_constrained_Cmax2(model,epsilon,C_max,total_electricity_cost):
    model.setObjective(C_max, GRB.MINIMIZE)
    model.addConstr(total_electricity_cost <= epsilon, "const_TEC")
    model.update()

def optimize_constrained_TEC2(model,epsilon2,C_max,total_electricity_cost):
    model.setObjective(total_electricity_cost, GRB.MINIMIZE)
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

def epsilon_constraint_solve(model,y,t,w,z,C_max,total_electricity_cost,pr):
    # compute minimum TEC
    Configure_model(model,y,t,w,z,C_max,pr)
    optimize_constrained_TEC(model,max_val,None,C_max,total_electricity_cost)
    model.optimize()
    min_TEC = model.objVal
    # compute minimum Cmax
    optimize_constrained_Cmax(model,max_val,None,C_max,total_electricity_cost)
    model.optimize()
    min_Cmax = model.objVal
    # compute epsilon
    epsilon = min_TEC
    epsilon2 = min_Cmax
    f1 = f2 = 0
    z_lu = [0,0]
    z_rl = [0,0]
    # solve the problem
    Pareto_set = set()
    # binary_vars = {}
    solution_list = []
    optimize_constrained_TEC(model,epsilon2,None,C_max,total_electricity_cost) # constraint min_Cmax to minimize TEC
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        # binary_vars['z_rl'] = model.getAttr('X', x)
        f1 = round(model.objVal,4)
        f2 = round(C_max.X,4)
        z_rl[0] = f1
        z_rl[1] = f2
        # solution_tuple = tuple(solution.items())
        Pareto_set.add( tuple(z_rl) )
        solution = store_solution(y,t,w,z)
        solution_list.append(solution)
        

    # Configure_model()
    optimize_constrained_Cmax(model,epsilon,'const_Cmax',C_max,total_electricity_cost) # constraint min_TEC to minimize Cmax
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        # binary_vars['z_lu'] = model.getAttr('X', x)
        f1 = round(total_electricity_cost.getValue(),4)
        f2 = round(model.objVal,4)
        z_lu[0] = f1
        z_lu[1] = f2
        # binary_vars['z_lu'] = z_lu
        # solution_tuple = tuple(solution.items())
        Pareto_set.add( tuple(z_lu)  )
        solution = store_solution(y,t,w,z)
        solution_list.append(solution)
    
    print("z_lu 左上角: ",z_lu) # f1 最小
    print("z_rl 右下角: ",z_rl) # f2 最小
    # Nadir_point = (z_rl[0],z_lu[1])
    model.remove(model.getConstrByName("const_TEC"))

    epsilon = z_rl[0] - 0.01 # 逐渐减小epsilon，限制f1 TEC,最小化Cmax
    decrement = 0.1
    f1_pre = z_rl[0]
    # f2_pre = z_rl[1]
    while epsilon > min_TEC:
        optimize_constrained_Cmax2(model,epsilon,C_max,total_electricity_cost) # constraint min_TEC to minimize Cmax
        model.optimize()
        if model.status == GRB.Status.OPTIMAL:
            # binary_vars['point'+str(i) ] = model.getAttr('X', x)
            f1 = round(total_electricity_cost.getValue(),4)
            f2 = round(model.objVal,4)
            point = (f1,f2)
            Pareto_set.add( point )
            solution = store_solution(y,t,w,z)
            solution_list.append(solution)
            # gap = abs(f1_pre - f1)
            # if gap < 0.1: # the region is dense.
            #     decrement = decrement*0.5
            # else:
            #     decrement = decrement*1.5
        epsilon -= decrement
    
    return Pareto_set,solution_list


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
    Pareto_set,solution_list = epsilon_constraint_solve(model,y,t,w,z,C_max,TEC,pr)
    time_end = time.time()

    # print("time: ",time_end - time_start)  
    time_cost = time_end - time_start

    return sorted(Pareto_set),time_cost, solution_list

model = Model("ToU")

# pareto_set,time_cost,solution_list = temp(model,p,q)
# print("Pareto_set: ",pareto_set)
# print("time_cost: ",time_cost)
# print("nb of solution_list: ",len(solution_list))

# p+v_ij_2
p_v_ij_2 = [[p[i][j] + v_ij_2[i][j] for j in range(n)] for i in range(m)] # min(v_ij_1[i][j],v_ij_2[i][j])

model2 = Model("ToU")
model2.params.TimeLimit = 600
pareto_set2,time_cost2,solution_list2 = temp(model2,p_v_ij_2,q)
# print("time_cost2: ",time_cost2)

def compute_rate_vij1_2_for_muij(muij,v_ij_1,v_ij_2):
    # 计算v_ij_1和v_ij_2中元素值占p_ij元素值的平均比例
    m,n = muij.shape
    rate_vij1 = np.zeros((m,n))
    rate_vij2 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            rate_vij1[i][j] = v_ij_1[i][j] / muij[i][j]
            rate_vij2[i][j] = v_ij_2[i][j] / muij[i][j]
    # 计算平均比例
    rate_vij1_avg = np.mean(rate_vij1)
    rate_vij2_avg = np.mean(rate_vij2)
    return round(rate_vij1_avg,4),round(rate_vij2_avg,4)

# 计算v_ij_1和v_ij_2中元素值占p_ij元素值的平均比例
rate_vij1_avg,rate_vij2_avg = compute_rate_vij1_2_for_muij(np.array(p),np.array(v_ij_1),np.array(v_ij_2))

dict_result = {
    "rate_vij1_avg": rate_vij1_avg,
    "rate_vij2_avg": rate_vij2_avg,
    # "mu_PF": pareto_set,
    # "mu_time": time_cost,
    # "mu_solutions": solution_list,
    "P2_PF": pareto_set2,
    "P2_time": time_cost2,
    "P2_solutions": solution_list2
}

# 存储结果
# result_file = "E:/ToU重做3/dict_experiment_data/D_0.01/results/9_72_result.json"
# with open(result_file, 'w') as f:
#     json.dump(dict_result, f, indent=4)
# Convert numpy types to native Python types for JSON serialization
def convert_np_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

# Convert all keys to string format
def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(v) for v in d]
    else:
        return d

# Convert the dictionary with tuple keys to use string keys
dict_result_str_keys = convert_keys_to_str(dict_result)
# print(dict_result_str_keys)
result_file = "E:/ToU重做3/dict_experiment_data/D_0.01/results/PolicyB/alpha=0.1/9_72_result.json"
# Save to JSON file
with open(result_file, 'w') as json_file:
    json.dump(dict_result_str_keys, json_file, default=convert_np_types, indent=4)