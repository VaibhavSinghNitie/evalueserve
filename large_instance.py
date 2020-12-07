import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime as dt
import json
import copy 

import pulp as lp

import process_data as input_ops


import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def initial_solution(tasks_df, task_0='Task_0'):

    model = lp.LpProblem("Warehouse_Assignment", lp.LpMinimize)

    x = lp.LpVariable.dicts("x", (workers_df.index, tasks_df.index), cat=lp.LpInteger, lowBound=0, upBound=1)
    t = lp.LpVariable("t", cat=lp.LpContinuous)

    model += lp.lpSum(t), "total time"

    for w in workers_df.index:
        model += lp.lpSum([x[w][task]*tasks_df['d'].loc[task] for task in tasks_df.index]+[-1*t]) <= 0, "max_time"+str(w)

    for task in tasks_df.index:
        model += lp.lpSum([x[w][task] for w in workers_df.index]) == 1, "assign"+task

    #model.writeLP("model.lp")
    model.solve(lp.PULP_CBC_CMD(msg=0))
    
    #print(lp.LpStatus[model.status])

    df_results = pd.DataFrame(0, columns=tasks_df.index, index=workers_df.index)

    assignments = {}
    [assignments.update({w:{'seq': [task_0], 'cost':0, 'bottleneck': ''}}) for w in workers_df.index]

    for v in model.variables():
        if (v.varValue == 1) & (v.name.split("_")[0] == "x"):
            #print(v.name, v.varValue)
            _, w, n_task, task = v.name.split("_")
            task_id = n_task + "_" + task
            df_results.at[int(w), task_id] = tasks_df['d'].loc[task_id]
            assignments[int(w)]['seq'].append(task_id)

    return assignments

def solve_tsp(tasks_df, d, task_0='Task_0'):
        
    tsp = lp.LpProblem("multiple_worker_with_schedule", lp.LpMinimize)

    n = len(tasks_df.index)

    x = lp.LpVariable.dicts("x", (tasks_df.index, tasks_df.index),lowBound=0, upBound=1, cat= lp.LpInteger)
    u = lp.LpVariable.dicts("u", (tasks_df.index), lowBound=0, upBound=n,cat=lp.LpInteger)

    obj = []

    for i in tasks_df.index:
        for j in tasks_df.index:
            obj.append(x[i][j]*d[j].loc[i])

    tsp +=  lp.lpSum(obj), "Max Time"


    for task_1 in tasks_df.index:
        ls_row = []
        ls_col = []
        for task_2 in tasks_df.index:
            if task_1 != task_2:
                ls_row.append(x[task_1][task_2])
                ls_col.append(x[task_2][task_1])
        
        tsp += lp.lpSum(ls_row) == 1, "Sum_of_Row_"+task_1
        tsp += lp.lpSum(ls_col) == 1, "Sum_of_Col_"+task_1

    for i in tasks_df.index:
        for j in tasks_df.index:
            if  ((j != task_0) & (i != j)) :
                tsp += lp.lpSum([u[j], -1*u[i], -1*x[i][j]*n]) >= 1-n, "dist_travelled_{}_{}".format(i,j)
        
                            
        tsp += lp.lpSum(x[i][i]) == 0, "Unreq_{}_{}".format(i,i)
    tsp += lp.lpSum(u[task_0]) == 1, "initial_seq"

    tsp.solve(lp.PULP_CBC_CMD(msg=0))
    #tsp.writeLP('tsp.lp')
    #print(lp.LpStatus[tsp.status])

    seq = [None for x in range(n)]
    distances = []
    for v in tsp.variables():
        if v.varValue != 0:
            #print(v.name, v.varValue)
            if  v.name.split("_")[0] == 'x':
                _, task_i, x_i, task_j, x_j = v.name.split("_")
                t_i = task_i+"_"+x_i
                t_j = task_j+"_"+x_j
                distances.append((t_i, t_j, d[t_j].loc[t_i]))

        
        if v.name.split("_")[0] == 'u':
            _, task, id = v.name.split("_")
            idx = int(v.varValue)-1
            seq[idx] = task+"_"+id
        
    distances = pd.DataFrame(distances, columns=("i", "j", "d"))
    idx = distances['d'].idxmax()
    bottleneck = distances['j'].loc[idx] if distances['j'].loc[idx] != task_0 else distances['i'].loc[idx]

    return seq, lp.value(tsp.objective), bottleneck

def get_makespan(assign):
    sm = []
    for key in assign.keys():
        sm.append(assign[key]['cost'])
    
    bottleneck = max(sm)

    for key in assign.keys():
        if assign[key]['cost'] == bottleneck:
            break

    return bottleneck, key

def get_neighbour(task, d, assign, n=5):
    neighbour_task = d[task][d.index != task].nsmallest(n).sample(1).index[0]
    for key in assign.keys():
        if neighbour_task in assign[key]['seq']:
            neighbour_worker = key
            break
    return neighbour_task, neighbour_worker


def check_legality(w, test_assign, test_0='Test_0'):
    if test_assign[w]['bottleneck'] == test_0:
        return False
    if len(test_assign[w]['seq']) <= 2:
        return False
    
    return True

def update(assign, n):
    
    temp_assign = assign.copy()
    cost_old, w = get_makespan(temp_assign)


    n_task, n_w = get_neighbour(assign[w]['bottleneck'], d, assign, n)


    temp_assign[w]['seq'].remove(assign[w]['bottleneck'])
    temp_assign[n_w]['seq'].append(assign[w]['bottleneck'])

    temp_assign[w]['seq'], temp_assign[w]['cost'], temp_assign[w]['bottleneck'] = solve_tsp(tasks_df.loc[temp_assign[w]['seq']], d)
    temp_assign[n_w]['seq'], temp_assign[n_w]['cost'], temp_assign[n_w]['bottleneck'] = solve_tsp(tasks_df.loc[temp_assign[n_w]['seq']], d)

    #print("Removing Task: {} of Worker: {} ! Adding Task: {} to worker: {}".format(assignments[w]['bottleneck'], w, assignments[w]['bottleneck'], n_w))

    cost, w = get_makespan(temp_assign)
    #print("Old: {} | New: {}".format(cost_old, cost))


    return  cost, cost_old, temp_assign



def get_travel_distance(ls, d, tasks_df, task_0='Task_0'):
    dist = []
    nm = []
    for i in range(1,len(ls)+1):
        if i == 1:
            frm = task_0
        else:
            frm = ls[i-1]
        if i == len(ls):
            to = task_0
        else:
            to = ls[i]
        dist.append(d[to].loc[frm]-tasks_df['d'].loc[to])
        nm.append('Transit: {} to {}'.format(frm, to))
        dist.append(tasks_df['d'].loc[to])
        nm.append(to)
    return dist, nm


if __name__ == "__main__":

    # SA Hyper parameter
    T0 = 1000
    N = 20
    M = 5
    alpha = 0.4

    n_neighbour = 5
    
    data_dir = 'data'
    viz_dir = 'viz'

    input_dir = os.path.join(data_dir, 'input')
    output_dir = os.path.join(data_dir, 'output')


    workers_file = 'data_1.csv'
    tasks_file = 'data_2.csv'


    workers_df, tasks_df =  input_ops.parse_input_data(input_dir, workers_file, tasks_file)


    task_0 = 'Task_0'
    assignments = initial_solution(tasks_df,task_0)


    tasks_df.loc[task_0] = [0,0,0,0,0]
    tasks_df.sort_index(inplace=True)


    d = pd.DataFrame(0, columns=tasks_df.index, index=tasks_df.index)

    for i in tasks_df.index:
        for j in tasks_df.index:
            if i != j:
                x1, y1 = tasks_df[["x2", "y2"]].loc[i]
                x2, y2 = tasks_df[["x1", "y1"]].loc[j]
                d.at[i,j] = input_ops.get_distance(x1,y1,x2,y2) + tasks_df["d"].loc[j]


    for key in assignments.keys():
        assignments[key]['seq'], assignments[key]['cost'], assignments[key]['bottleneck'] = solve_tsp(tasks_df.loc[assignments[key]['seq']], d)


    Temp = []
    min_cost = []
    for i in tqdm(range(N), desc="epoch"):
        for j in range(N):
            best = 10000
            
            temp_assignments = {}
            
            assignment_old = copy.deepcopy(assignments)

            cost_new, cost_old, assignments = update(assignments, n_neighbour)

            if cost_new < best:
                best_assignment =  copy.deepcopy(assignments)
                best = cost_new
            
            rand_1 = np.random.rand()
            form = 1/(np.exp((cost_new-cost_old)/T0))
            
            if cost_new <= cost_old:
                min_cost.append(cost_new)

            elif rand_1 <= form:
                min_cost.append(cost_new)

            else:
                min_cost.append(cost_old)
                assignments = copy.deepcopy(assignment_old)

            cost = get_makespan(assignments)

            print("Epoch: {}, Iter: {}, Temp: {}, Cost_old: {}, Cost_new: {}, Final: {}, change: {}, form: {} ".format(i,j,T0, cost_old, cost_new, cost,((rand_1<=form) | (cost_old>=cost_new)), form))
        Temp.append(T0)
        T0 = alpha*T0
    
    if cost[0] > best:
        assignments = copy.deepcopy(best_assignment)
    
    print(get_makespan(assignments))
    #Saving Assignments

    with open(os.path.join(output_dir, 'model_3.json'), 'w') as fp:
        json.dump(assignments, fp,  indent=4)
    
    
    
    # Saving Visualization
    data_df = []

    grand_start = dt.datetime(year=2020, month=1, day=1, hour=8)

    for key in assignments.keys():

        x, y = get_travel_distance(assignments[key]['seq'], d, tasks_df)
        start = grand_start
        activity_type = ['non value adding', 'value adding']
        for i in range(len(x)):
            end = start+dt.timedelta(minutes=int(x[i]))
            data_df.append([y[i], start, end, workers_df.loc[key].values[0], activity_type[i%2], x[i]])
            start = end


    clr_map = {'non value adding': '#FA8072', 'value adding': '#33beff'}

    df = pd.DataFrame(data_df, columns=["Task", "Start", "Finish", "Resource", "Activity", "Time"])
    fig = px.timeline(df,title="Gantt Chart", x_start="Start", x_end="Finish", y="Resource", color="Activity", hover_data=['Task'], color_discrete_map=clr_map)
    fig.write_html(os.path.join("viz", "gantt_chart.html"))

    #fig.show()

    fig = px.pie(df, values='Time', names='Activity', title='Value Adding vs Non Value Adding', color='Activity', color_discrete_map=clr_map)
    fig.write_html(os.path.join("viz", "va_vs_nva.html"))

    #fig.show()

    fig = px.histogram(df, title="Time distribution", x="Time", color="Activity", color_discrete_map=clr_map)
    fig.write_html(os.path.join("viz", "dist_time.html"))

    #fig.show()

    df['Waiting'] = [(row['Start'] - grand_start).total_seconds()/60  if row['Activity'] == activity_type[1] else 0 for i,row in df.iterrows() ]

    df_waiting = df[(df.Activity == activity_type[1]) & (df.Task != task_0)]
    df_waiting['Activity'] = activity_type[0]

    fig = px.bar(df_waiting.sort_values('Waiting', ascending=False), title="Task Waiting Time (Min)",y="Waiting", x="Task", color="Activity",color_discrete_map=clr_map)
    fig.write_html(os.path.join("viz", "idle_time.html"))

    #fig.show()

    assignment_matrix = []

    for key in assignments.keys():
        assignment_matrix.append([ workers_df.loc[key].values[0], assignments[key]['seq']+[task_0], assignments[key]['cost'] ])

    df_assign = pd.DataFrame(assignment_matrix, columns= ['Worker', 'Assignment Sequence', 'Total Time(Min)'])
    fig = go.Figure(data=[go.Table(
        columnwidth=[80,300,80],
        header=dict(values=list(df_assign.columns),
                    fill_color='#88d0f2',
                    align='left'),
        cells=dict(values=[df_assign['Worker'], df_assign['Assignment Sequence'], df_assign['Total Time(Min)']],
                fill_color='#d1e6f0',
                align='left'))
    ])
    fig.update_layout(title="Assignments")
    fig.write_html(os.path.join("viz", "assignments.html"))

    #fig.show()

			
