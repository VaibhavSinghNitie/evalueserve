import os
import pandas as pd


def get_distance(x1, y1, x2, y2):

    return abs(x1-x2) + abs(y1-y2)


def parse_input_data(_dir, w_file, t_file):
    
    w_df = pd.read_csv(os.path.join(_dir, w_file))
    w_df.set_index('EMPLOYEE_ID', inplace=True)

    t_df = pd.read_csv(os.path.join(_dir, t_file))
    t_df.set_index('TASK_ID', inplace=True)
    t_df.rename({
        "START_X_COORD": "x1",  
        "START_Y_COORD": "y1",
        "STOP_X_COORD": "x2",
        "STOP_Y_COORD": "y2"
    }, axis=1, inplace=True)

    t_df['d'] = [get_distance(row['x1'],row['y1'],row['x2'],row['y2']) for _,row in t_df.iterrows()]
    
    return w_df, t_df