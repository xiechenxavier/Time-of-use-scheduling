from gurobipy import Model, GRB, quicksum
import numpy as np
import json
import os
import time


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
file_path = "E:/ToU重做3/dict_experiment_data/D_0.01/9_72.json"
m, n, p, q, v_ij_1, v_ij_2, T, R, S, B, K = extract_data_from_file(file_path,5)
Tmin = min(T)
Tmax = max(T)

p_v_ij_2 = [[p[i][j] + min(v_ij_1[i][j],v_ij_2[i][j]) for j in range(n)] for i in range(m)]


def Configure_model(p,q):

    model = Model("Makespan")
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")
    t = model.addVars(m, n, K, vtype=GRB.CONTINUOUS, name="t")
    w = model.addVars(n, K, vtype=GRB.BINARY, name="w")
    z = model.addVars(m, K, vtype=GRB.BINARY, name="z")
    C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
    t_indices = [(i, j, k) for i in range(m) for j in range(n) for k in range(K)]
    TEC = quicksum(q[i] * R[k] * t[i, j, k] for (i, j, k) in t_indices)
    model.addConstrs((quicksum(y[i, j] for i in range(m)) == 1 for j in range(n)), name="AssignJob")
    model.addConstrs((quicksum(t[i, j, k] for k in range(K)) == p[i][j] * y[i, j] for i in range(m) for j in range(n)), name="ProcTime")
    model.addConstrs((quicksum(t[i, j, k] for i in range(m)) <= T[k] * w[j, k] for j in range(n) for k in range(K)), name="Link_w_t")
    model.addConstrs((quicksum(t[i,j,k] for i in range(m)) >= T[k]*(w[j, k - 1] + w[j, k + 1] - 1) for j in range(n) for k in range(1, K-1)), name="Continuity") # to change
    model.addConstrs((quicksum(w[j, kp] for kp in range(K) if kp > k + 1) <= (K - k - 1) * (1 - w[j, k] + w[j, k + 1]) for j in range(n) for k in range(K-2)), name="ProcessingOrder")
    model.addConstrs((z[i, k] >= (1 / T[k]) * quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="Link_z_t")
    model.addConstrs((C_max >= z[i, k] * S[k] + quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="makespan")
    return model,C_max,TEC,y,t,w,z

def Add_inequalities(inequality_type,p,q,objective_type):
    model,C_max,TEC,y,t,w,z = Configure_model(p,q)
    # According to the inequality type, we will add the corresponding inequality
    if inequality_type == 0:
        # Add the first inequality
        model.addConstrs( quicksum(t[i,j,k] for j in range(n) for k in range(K)) 
                     <= C_max for i in range(m))
    elif inequality_type == 1:
        # Add the second inequality
        model.addConstrs( C_max 
                     >= quicksum(p[i][j] * y[i, j] for j in range(n))  for i in range(m))
    elif inequality_type == 2:
        # Add the third inequalities
        model.addConstr( C_max 
                     >= 1/m * quicksum(p[i][j] * y[i, j] for j in range(n) for i in range(m)) )
    elif inequality_type == 3:
        # Add all the three inequalities
        model.addConstrs( quicksum(t[i,j,k] for j in range(n) for k in range(K)) 
                     <= C_max for i in range(m))
        model.addConstrs( C_max >= quicksum(p[i][j] * y[i, j] for j in range(n))  for i in range(m))
        model.addConstr( C_max >= 1/m * quicksum(p[i][j] * y[i, j] for j in range(n) for i in range(m)) )
    
    # elif inequality_type == 4:
    #     # Add first two inequalities
    #     model.addConstrs( quicksum(t[i,j,k] for j in range(n) for k in range(K)) 
    #                  <= C_max for i in range(m))
    #     model.addConstrs(quicksum(z[i,k]*T[k] for k in range(K)) >= C_max for i in range(m))

    # elif inequality_type == 5:
    #     # Add the first and third inequalities
    #     model.addConstrs( quicksum(t[i,j,k] for j in range(n) for k in range(K)) 
    #                  <= C_max for i in range(m))
    #     model.addConstr( C_max >= 1/m * quicksum(p[i][j] * y[i, j] for j in range(n) for i in range(m)) )
    
    # elif inequality_type == 6:
    #     # Add the second and third inequalities
    #     model.addConstrs( C_max >= quicksum(p[i][j] * y[i, j] for j in range(n))  for i in range(m))
    #     model.addConstr( C_max >= 1/m * quicksum(p[i][j] * y[i, j] for j in range(n) for i in range(m)) )

    else:
        pass
    

    if objective_type == 0:
        model.setObjective(TEC, GRB.MINIMIZE)
    elif objective_type == 1:
        model.setObjective(C_max, GRB.MINIMIZE)
    return model


def execute_function(inequality_type,p,q,objective_type):
    total_time = 0
    frequency = 1
    obj_val = 0
    gap = 0
    for _ in range(frequency):
        model = Add_inequalities(inequality_type,p,q,objective_type)
        # model.setParam('gap', 0.0001)
        model.setParam('TimeLimit', 3600)
        time_start = time.time()
        model.optimize()
        time_end = time.time()
        # print('time cost', time_end-time_start, 's')
        total_time += (time_end-time_start)
        # 输出结果
        if model.status == GRB.Status.OPTIMAL:
            obj_val = model.objVal
        
        if model.status == GRB.Status.TIME_LIMIT:
            if model.SolCount > 0:
                obj_val = model.objVal
                gap += model.MIPGap
            else:
                obj_val = 0
                gap += 0

    # print('Average time cost', total_time/frequency, 's') 
    return obj_val,total_time/frequency,gap/frequency
    

def Run(p,q):
    # Optimize the C_max
    print("Without adding any inequality")
    OPT_Cmax1, timep_1,gap1 = execute_function(20,p,q,1)
    print("Adding the first inequality")
    OPT_Cmax2, timep_2,gap2 = execute_function(0,p,q,1)
    # print("Adding the second inequality")
    # OPT_Cmax3, timep_3 = execute_function(1,p,q,1)
    # print("Adding the third inequality")
    # OPT_Cmax4, timep_4 = execute_function(2,p,q,1)
    # print("Adding all the three inequalities")
    # OPT_Cmax5, timep_5 = execute_function(3,p,q,1)
    # print("Adding the fifth inequality")
    # OPT_Cmax6, timep_6 = execute_function(4,p,q,1)
    # print("Adding the sixth inequality")
    # OPT_Cmax7, timep_7 = execute_function(5,p,q,1)

    dataset = dict()
    dataset["C_max"] = [OPT_Cmax1, OPT_Cmax2]
    dataset["Time"] = [timep_1, timep_2]
    dataset["Gap"] = [gap1, gap2]
    # dataset["C_max"] = [OPT_Cmax5]
    # dataset["Time"] = [timep_5] #3600, 1.53%, 577.5; PolicyB: 3600, 1.73%, 

    print(dataset)
    file_path = "E:/ToU重做3/dict_experiment_data/Tools/Valid_inequalities_tests/Results_D_0.01/PolicyA/9_72res.json"
    # write the results to a file
    with open(file_path, 'w') as file:
        json.dump(dataset, file)

Run(p_v_ij_2,q)

# 第四个之后都有问题

# 4.070119380950928,46.8394033908844,18.103373050689697 第四个不等式
# 3.28920578956604
# {'C_max': [594.1, 584.0, 584.0, 584.0, 584.0], 'Time': [3600, 3843.6484899520874, 36.475815296173096, 459.1981391906738, 6.143837928771973]}