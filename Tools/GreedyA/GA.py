from gurobipy import Model, GRB, quicksum
import numpy as np
import json
import os
import time
import math

def is_dominated(point, others, index):
    x1, y2 = point
    for i, (x2, y1) in enumerate(others):
        if i != index:  # 排除当前索引对应的点
            if x2 <= x1 and y1 <= y2 and (x2 < x1 or y1 < y2): # and (x2 < x1 or y1 < y2)
                return True
    return False

def brute_force_skyline(points):
    skyline = []
    for i, point in enumerate(points):
        if not is_dominated(point, points, i):
            skyline.append(point)
    return skyline


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
file_path = "E:/ToU重做3/dict_experiment_data/D_0.01/alpha=0.1/7_49.json"
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


def Configure_model(model, p):
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")
    t = model.addVars(m, n, K, vtype=GRB.CONTINUOUS, name="t")
    w = model.addVars(n, K, vtype=GRB.BINARY, name="w")
    z = model.addVars(m, K, vtype=GRB.BINARY, name="z")
    C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
    TEC = quicksum(q[i] * R[k] * t[i, j, k] for (i, j, k) in t_indices)

    model.addConstrs((quicksum(y[i, j] for i in range(m)) == 1 for j in range(n)), name="AssignJob")
    model.addConstrs((quicksum(t[i, j, k] for k in range(K)) == p[i][j] * y[i, j] for i in range(m) for j in range(n)), name="ProcTime")
    model.addConstrs((quicksum(t[i, j, k] for i in range(m)) <= T[k] * w[j, k] for j in range(n) for k in range(K)), name="Link_w_t")
    model.addConstrs((quicksum(t[i,j,k] for i in range(m)) >= T[k]*(w[j, k - 1] + w[j, k + 1] - 1) for j in range(n) for k in range(1, K-1)), name="Continuity") # to change
    model.addConstrs((quicksum(w[j, kp] for kp in range(K) if kp > k + 1) <= (K - k - 1) * (1 - w[j, k] + w[j, k + 1]) for j in range(n) for k in range(K-2)), name="ProcessingOrder")
    model.addConstrs((z[i, k] >= (1 / T[k]) * quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="Link_z_t")
    model.addConstrs((C_max >= z[i, k] * S[k] + quicksum(t[i, j, k] for j in range(n)) for i in range(m) for k in range(K)), name="makespan")

    return model, y, t, w, z, C_max, TEC

# 设定每个机器的最大电力消耗量
def optimize_constrained_TEC(model2,y2, t2, w2, z2, C_max2, TEC2):
    model2.setObjective(TEC2, GRB.MINIMIZE)
    model2.addConstrs( quicksum(t2[i,j,k] for i in range(m) ) 
                            >= quicksum(p[i][j]*y2[i,j] * (w2[j,k] - quicksum(w2[j,k_prime] for k_prime in range(k,K)) )
                                        - quicksum(t2[i,j,k_prime] for k_prime in range(0,k))
                                        for i in range(m))
                            for j in range(n) for k in range(1,K-1))
    model2.addConstr(TEC2 <= quicksum(q[i] * R_max * t2[i, j, k] for (i, j, k) in t_indices))
    model2.addConstr(TEC2 >= quicksum(q[i] * R_min * t2[i, j, k] for (i, j, k) in t_indices))

def optimize_constrained_Cmax(model1, y1, t1, w1, z1, C_max1, TEC1):
    model1.setObjective(C_max1, GRB.MINIMIZE)
    model1.addConstrs( quicksum(t1[i,j,k] for j in range(n) for k in range(K)) 
                                     <= C_max1 for i in range(m))
    model1.addConstrs( C_max1 
                     >= quicksum(p[i][j] * y1[i, j] for j in range(n))  for i in range(m))
    model1.addConstr( C_max1 
                     >= 1/m * quicksum(p[i][j] * y1[i, j] for j in range(n) for i in range(m)) )
    model1.update()

def DualRestrict_routine(model2,TEC2,C_max2):
    model2.setObjective(TEC2, GRB.MINIMIZE)
    model2.optimize()
    if model2.status == GRB.OPTIMAL:
            f1 = (C_max2.X)
            f2 = round(model2.objVal,2)
            return (f1,f2)
    elif model2.status == GRB.TIME_LIMIT and model2.solCount > 0:
            if model2.status == GRB.TIME_LIMIT:
                f1 = (C_max2.X)
                f2 = round(model2.objVal*(1-model2.MIPGap),2)
                return (f1,f2)
        # return (round(model.objVal,2),round(C_max.X,2))
    return None

def Restrict_routine(model1,TEC1,C_max1):
    model1.setObjective(C_max1, GRB.MINIMIZE)
    model1.optimize()
    if model1.status == GRB.OPTIMAL:
            f1 = (model1.objVal)
            f2 = round(TEC1.getValue(),2)
            return (f1,f2)
    elif model1.status == GRB.TIME_LIMIT and model1.solCount > 0:
            if model1.status == GRB.TIME_LIMIT:
                f1 = round(model1.objVal*(1-model1.MIPGap),2)
                f2 = round(TEC1.getValue(),2)
                return (f1,f2)
    return None

def Greedy_Algorithm(model1,model2,p):
    # compute minimum TEC
    APX = []
    APX2 = []
    model1, y1, t1, w1, z1, C_max1, TEC1 = Configure_model(model1,p) # model1 to compute Cmax
    model2, y2, t2, w2, z2, C_max2, TEC2 = Configure_model(model2,p) # model2 to compute TEC
    # compute minimum Cmax
    optimize_constrained_Cmax(model1, y1, t1, w1, z1, C_max1, TEC1)
    model1.optimize()
    OPTCmax = (model1.objVal)

    optimize_constrained_TEC(model2, y2, t2, w2, z2, C_max2, TEC2) # set objective function
    model2.optimize()
    OPTtec = round(model2.objVal,2) # get the optimal value of TEC

    epsilon = 0.03
    itr = OPTtec * (1+epsilon)
    s1 = [0,0]

    # Step 2: Iterative model
    modifiable_const_tec = model1.addConstr(TEC1 <= itr)
    model1.optimize()
    if model1.status == GRB.OPTIMAL:
        s1[0] = model1.objVal
        s1[1] = round(TEC1.getValue(),2)
        APX.append((s1[0],s1[1]))
        APX2.append((s1[0],s1[1]))
    
    R1 = (s1[0]/(1+epsilon))
    
    modifiable_const_cmax = model2.addConstr(C_max2 <= R1)
    # i = 1
    while R1 > OPTCmax:
        modifiable_const_cmax.setAttr(GRB.Attr.RHS,R1)
        si_prime = DualRestrict_routine(model2,TEC2,C_max2)
        APX2.append(si_prime)
        if si_prime is None:
            break
        R2 = si_prime[1]*(1+epsilon)
        modifiable_const_tec.setAttr(GRB.Attr.RHS,R2)
        si = Restrict_routine(model1,TEC1,C_max1)
        if si is not None:
            APX.append(si)
            APX2.append(si)
            R1 = (si[0]/(1+epsilon))
        else:
            break

    APX = brute_force_skyline(APX)
    # print(APX)
    return APX,APX2
    

model1 = Model("Cmax")
model2 = Model("TEC")
# model1.setParam("MIPGap", 0.01)
t2 = time.time()
APX,APX2 = Greedy_Algorithm(model2,model1,p_v_ij_2)
t1 = time.time()
print("time: ",t1-t2)
print(APX)

result_result = "E:/ToU重做3/dict_experiment_data/Tools/GreedyA/Real_results/PolicyB/10_120.json"

# 读取json文件数据
with open(result_result, 'r') as file:
    dict_result = json.load(file)

dict_result["0.03_APX"] = APX 
dict_result["0.03_APX2"] = APX2
dict_result["0.03_APX_time"] = t2-t1

with open(result_result, 'w') as file:
    json.dump(dict_result, file, indent=4)