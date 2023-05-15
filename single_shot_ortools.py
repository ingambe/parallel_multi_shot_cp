import collections
import os
import random
import numpy as np
import time
import argparse
from os import listdir
from os.path import isfile
from typing import List, Tuple

from ortools.sat.python import cp_model
from tqdm import tqdm
import argparse

import pandas as pd


def solve_one_instance(instance_path: str, limit_time: int) -> Tuple[int, int, int, float]:
    jobs_data = []
    machines_count = 0
    jobs_count = 0

    instance_file = open(instance_path, 'r')
    line_str = instance_file.readline()
    line_cnt = 1
    horizon = 0
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, op_time = int(split_data[i]), int(split_data[i + 1])
                horizon += op_time
                data.append((machine, op_time))
                i += 2
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    start = time.time()

    all_machines = range(machines_count)
    # horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    all_partial_tasks = collections.defaultdict(list)
    machine_to_intervals = collections.defaultdict(list)

    model = cp_model.CpModel()

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = "_%i_%i" % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)

            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            all_partial_tasks[job_id].append(all_tasks[job_id, task_id])
            machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, "makespan")
    objs = []
    for job_id, job in enumerate(jobs_data):
        if len(all_partial_tasks[job_id]) > 0:
            objs.append(all_partial_tasks[job_id][-1].end)
    model.AddMaxEquality(obj_var, objs)
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit_time
    solver.parameters.num_search_workers = 32
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    correct_solution = status == cp_model.OPTIMAL or status == cp_model.FEASIBLE
    assert correct_solution
    return solver.ObjectiveValue(), jobs_count, machines_count, time.time() - start


def solver_multiple_instances(instances: List[str], total_time: int, hints: bool = True, csv_name: str = 'results.csv'):
    import wandb
    wandb.init(project='CP_ICAPS', entity='cp_project', name=f'single_solving_ortools_{total_time // 60}min')
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    results = collections.defaultdict(list)
    pbar = tqdm(instances, unit=" instances")
    for instance in pbar:
        pbar.set_description(f'Currently solving instance {"".join(instance.split("/")[1:])}')
        results['name'].append(instance)
        output = solve_one_instance(instance, total_time)
        results['jobs'].append(output[1])
        results['machine'].append(output[2])
        results['objective'].append(output[0])
        results['time'].append(output[3])
        print(f'for {instance} we have {output[0]}')
    df = pd.DataFrame.from_dict(results)
    df.to_csv(csv_name)
    print(df.to_string())
    wandb.log({'results': wandb.Table(dataframe=df)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OR Tool classic')
    parser.add_argument('--time', type=int, default=600, help='Total time')
    args = parser.parse_args()
    all_instances = [os.path.join('instances', f) for f in listdir('instances') if
                     isfile(os.path.join('instances', f)) and not f.startswith('.')]
    all_instances.sort()
    solver_multiple_instances(all_instances, args.time)