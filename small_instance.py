import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

import pulp as lp

import process_data as input_ops


data_dir = 'data'

input_dir = os.path.join(data_dir, 'input')
output_dir = os.path.join(data_dir, 'output')


workers_file = 'data_1.csv'
tasks_file = 'data_2.csv'


workers_df, tasks_df =  input_ops.parse_input_data(input_dir, workers_file, tasks_file)



# Model-1 No Transition Time
model = lp.LpProblem("Warehouse_Assignment", lp.LpMinimize)

x = lp.LpVariable.dicts("x", (workers_df.index, tasks_df.index), cat=lp.LpInteger, lowBound=0, upBound=1)

t = lp.LpVariable("t", cat=lp.LpContinuous)

model += lp.lpSum(t), "total time"

for w in workers_df.index:
    model += lp.lpSum([x[w][task]*tasks_df['d'].loc[task] for task in tasks_df.index]+[-1*t]) <= 0, "max_time"+str(w)

for task in tasks_df.index:
    model += lp.lpSum([x[w][task] for w in workers_df.index]) == 1, "assign"+task


# model.writeLP("model.lp")

model.solve()
  
lp.LpStatus[model.status]

df_results = pd.DataFrame(0, columns=tasks_df.index, index=workers_df.index)

for v in model.variables():
    if v.varValue == 1:
        _, w, n_task, task = v.name.split("_")
        task_id = n_task + "_" + task
        df_results.at[int(w), task_id] = tasks_df['d'].loc[task_id]

df_results.to_csv("model_1.csv")
lp.value(model.objective)



# Model - 2 With Transition Time

tasks_df_old = tasks_df
workers_df_old = workers_df
tasks_df = tasks_df_old.iloc[:5]
workers_df = workers_df_old.iloc[:2]

task_0 = 'Task_0'
tasks_df.loc[task_0] = [0,0,0,0,0]

tasks_df.sort_index(inplace=True)

d = pd.DataFrame(0, columns=tasks_df.index, index=tasks_df.index)

for i in tasks_df.index:
    for j in tasks_df.index:
        if i != j:
            x1, y1 = tasks_df[["x2", "y2"]].loc[i]
            x2, y2 = tasks_df[["x1", "y1"]].loc[j]
            d.at[i,j] = input_ops.get_distance(x1,y1,x2,y2) + tasks_df["d"].loc[j]

M = d.values.sum()

model_1 = lp.LpProblem("multiple_worker_with_schedule", lp.LpMinimize)

n = len(tasks_df.index)
m = len(workers_df.index)

x = lp.LpVariable.dicts("x", (tasks_df.index, tasks_df.index),lowBound=0, upBound=1, cat= lp.LpInteger)
u = lp.LpVariable.dicts("u", (tasks_df.index), lowBound=0, upBound=M,cat=lp.LpContinuous)
Cmax = lp.LpVariable("Cmax", lowBound=0, upBound=M)



model_1 +=  lp.lpSum(Cmax), "Max Time"


for task_1 in tasks_df.index:
    ls_row = []
    ls_col = []
    for task_2 in tasks_df.index:
        if task_1 != task_2:
            ls_row.append(x[task_1][task_2])
            ls_col.append(x[task_2][task_1])

    sm = 1 if task_1 != task_0 else m
     
    model_1 += lp.lpSum(ls_row) == sm, "Sum_of_Row_"+task_1
    model_1 += lp.lpSum(ls_col) == sm, "Sum_of_Col_"+task_1

for i in tasks_df.index:
    for j in tasks_df.index:
        if  ((j != task_0) & (i != j)) :
            model_1 += lp.lpSum([u[j], -1*u[i], -1*x[i][j]*(d[j].loc[i] + M)]) >= -M, "sub_tour_elim_{}_{}".format(i,j)           
    model_1 += lp.lpSum(x[i][i]) == 0, "Unreq_{}_{}".format(i,i)
    model_1 += lp.lpSum([u[i], -1*Cmax]) <= 0, "Cmax_{}".format(i)

model_1 += lp.lpSum(u[task_0]) == 0, "initial_seq"


model_1.writeLP("Model_1.LP")
model_1.solve()

lp.LpStatus[model_1.status]

result = {}

for v in model_1.variables():
    if v.varValue != 0:
        result[v.name] = v.varValue


with open(os.path.join(output_dir, 'model_2.json'), 'w') as fp:
    json.dump(result, fp,  indent=4)
    
    
workers_df.to_csv("worker.csv")
tasks_df.to_csv("tasks.csv")
