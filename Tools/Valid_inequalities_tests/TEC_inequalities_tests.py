import gurobipy as gp
from gurobipy import GRB, quicksum, Model
import time
import json
import os
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


file_path = "E:/ToU重做3/dict_experiment_data/D_0.01/alpha=0.1/9_81.json"
m, n, p, q, v_ij_1, v_ij_2, T, R, S, B, K = extract_data_from_file(file_path,5)
p_v_ij_2 = [[p[i][j] + min(v_ij_1[i][j],v_ij_2[i][j]) for j in range(n)] for i in range(m)]

# Calculate min and max of T over set K
T_min = min(T)
T_max = max(T)
R_min = min(R)
R_max = max(R)
K_low = [k for k in range(K) if T[k] <= T_min]
K_high = [k for k in range(K) if T[k] >= T_max]
Delta = math.ceil(sum([T[k] for k in K_high])/sum([T[k] for k in K_low]))
# print(Delta)

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

# print(p_v_ij_2)

def Add_inequalities(num_ineq,p,q):
    model,C_max,TEC,y,t,w,z = Configure_model(p,q)
    t_indices = [(i, j, k) for i in range(m) for j in range(n) for k in range(K)]
    if num_ineq == 0:
        # Constraint 0: Sum over k of w[j, k] <= (sum over i of p[i][j] * y[i, j]) / T_min + 2, for all j in range(n)
        # model.addConstrs((quicksum(w[j, k] for k in range(K)) <= quicksum(p[i][j] * y[i, j] for i in range(m)) / T_min + 2) for j in range(n))
        model.addConstr((Delta*quicksum(t[i,j,k] for k in K_low for j in range(n) for i in range(m)) 
                     - quicksum(t[i,j,k] for k in K_high for j in range(n) for i in range(m) )>=0
                     ), 
                     name="Constraint")
    elif num_ineq == 1:
        # Constraint 2: Sum over k of z[i, k] <= (sum over j of p[i][j] * y[i, j]) / T_min, for all i in range(m)
        # model.addConstrs((quicksum(z[i, k] for k in range(K)) <= quicksum(p[i][j] * y[i, j] for j in range(n)) / T_min for i in range(m)))
        model.addConstr(TEC <= quicksum(q[i] * R_max * t[i, j, k] for (i, j, k) in t_indices))
        model.addConstr(TEC >= quicksum(q[i] * R_min * t[i, j, k] for (i, j, k) in t_indices))
        
    elif num_ineq == 2:
        # Add the first inequality
        model.addConstrs( quicksum(t[i,j,k] for i in range(m) ) 
                        >= quicksum(p[i][j]*(y[i,j] -1 + w[j,k]) - quicksum(t[i,j,k_prime] for k_prime in range(0,k))
                                    - p[i][j]*quicksum(w[j,k_prime] for k_prime in range(k,K))
                                    for i in range(m))
                        for j in range(n) for k in range(1,K-1))
    # elif num_ineq == 3:
    #     # Constraint 3: Sum over k of z[i, k] >= (sum over j of p[i][j] * y[i, j]) / T_max, for all i in range(m)
    #     model.addConstrs((quicksum(z[i, k] for k in range(K)) >= quicksum(p[i][j] * y[i, j] for j in range(n)) / T_max for i in range(m)))

    elif num_ineq == 3:
        model.addConstrs( quicksum(t[i,j,k] for i in range(m) ) 
                        >= quicksum(p[i][j]*(y[i,j] -1 + w[j,k]) - quicksum(t[i,j,k_prime] for k_prime in range(0,k))
                                    - p[i][j]*quicksum(w[j,k_prime] for k_prime in range(k,K))
                                    for i in range(m))
                        for j in range(n) for k in range(1,K-1))
        model.addConstr(TEC <= quicksum(q[i] * R_max * t[i, j, k] for (i, j, k) in t_indices))
        model.addConstr(TEC >= quicksum(q[i] * R_min * t[i, j, k] for (i, j, k) in t_indices))
        # model.addConstrs((quicksum(z[i, k] for k in range(K)) <= quicksum(p[i][j] * y[i, j] for j in range(n)) / T_min for i in range(m)))

    else:
        pass
    
    model.setObjective(TEC, GRB.MINIMIZE)
    return model

def execute_function(inequality_type,p,q):
    total_time = 0
    frequency = 3
    obj_val = 0
    for _ in range(frequency):
        model = Add_inequalities(inequality_type,p,q)
        # model.setParam('gap', 0.0001)
        time_start = time.time()
        model.optimize()
        time_end = time.time()
        # print('time cost', time_end-time_start, 's')
        total_time += (time_end-time_start)
        # 输出结果
        if model.status == GRB.Status.OPTIMAL:
            obj_val = model.objVal

    # print('Average time cost', total_time/frequency, 's') 
    return obj_val,total_time/frequency
    

def Run(p,q):
    # Optimize the C_max
    print("Without adding any inequality")
    OPT_Cmax1, timep_1 = execute_function(20,p,q)
    print("Adding the first inequality")
    OPT_Cmax2, timep_2 = execute_function(0,p,q)
    print("Adding the second inequality")
    OPT_Cmax3, timep_3 = execute_function(1,p,q)
    print("Adding the third inequality")
    OPT_Cmax4, timep_4 = execute_function(2,p,q)
    print("Adding the forth inequalities")
    OPT_Cmax5, timep_5 = execute_function(3,p,q)
    # print("Adding two inequalities")
    # OPT_Cmax6, timep_6 = execute_function(4,p,q)
    # print("Adding the fifth inequality")
    # OPT_Cmax6, timep_6 = execute_function(4,p,q)
    # print("Adding the sixth inequality")
    # OPT_Cmax7, timep_7 = execute_function(5,p,q)

    dataset = dict()
    dataset["TEC"] = [OPT_Cmax1, OPT_Cmax2, OPT_Cmax3, OPT_Cmax4, OPT_Cmax5]
    dataset["Time"] = [timep_1, timep_2, timep_3, timep_4, timep_5]
    # dataset["C_max"] = [OPT_Cmax5]
    # dataset["Time"] = [timep_5] #3600, 1.53%, 577.5; PolicyB: 3600, 1.73%, 

    print(dataset)
    # file_path = "E:/ToU重做3/dict_experiment_data/Tools/Valid_inequalities_tests/TEC_results_D_0.01/PolicyA/9_81res.json"
    # # write the results to a file
    # with open(file_path, 'w') as file:
    #     json.dump(dataset, file)

Run(p_v_ij_2,q)