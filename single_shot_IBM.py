import collections
import random
import numpy as np
import time
import argparse
from os import listdir
from os.path import isfile
from typing import List, Tuple
from docplex.cp.model import *

from docplex.cp.model import CpoModel
from tqdm import tqdm

import pandas as pd


def solve_one_instance(instance_path: str, limit_time: int, hints: bool = True) -> Tuple[int, int, int, float]:
    context.verbose = 0
    context.model.add_source_location = False
    context.model.length_for_alias = 10
    context.model.name_all_constraints = False
    context.model.dump_directory = None
    context.model.sort_names = None
    context.solver.trace_cpo = False
    context.solver.trace_log = False
    context.solver.add_log_to_solution = False

    mdl = CpoModel()

    jobs_data = []
    machines_count = 0
    jobs_count = 0

    instance_file = open(instance_path, 'r')
    line_str = instance_file.readline()
    line_cnt = 1
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, op_time = int(split_data[i]), int(split_data[i + 1])
                data.append((machine, op_time))
                i += 2
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    start = time.time()

    all_machines = range(machines_count)

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = "_%i_%i" % (job_id, task_id)

            interval_variable = interval_var(size=duration, name="interval" + suffix)
            machine_to_intervals[machine].append(interval_variable)
            all_tasks[job_id, task_id] = interval_variable

        # Create and add disjunctive constraints.
    for machine in all_machines:
        mdl.add(no_overlap(machine_to_intervals[machine]))

        # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(1, len(job)):
            mdl.add(end_before_start(all_tasks[job_id, task_id - 1], all_tasks[job_id, task_id]))

    mdl.add(minimize(max(end_of(task_interval) for task_interval in all_tasks.values())))

    # Solve model.
    res = mdl.solve(TimeLimit=limit_time, Workers=32, LogVerbosity='Quiet')
    correct_solution = res.is_solution()
    assert correct_solution, "No solution found"
    return res.get_objective_value(), jobs_count, machines_count, time.time() - start


def solver_multiple_instances(instances: List[str], limit_time: int, hints: bool = True, csv_name: str = 'results.csv'):
    import wandb
    wandb.init(project='CP_ICAPS', entity='cp_project', name=f'single_solving_ibm_{limit_time // 60}min')
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    results = collections.defaultdict(list)
    pbar = tqdm(instances, unit=" instances")
    for instance in pbar:
        pbar.set_description(f'Currently solving instance {"".join(instance.split("/")[1:])}')
        results['name'].append(instance)
        output = solve_one_instance(instance, limit_time)
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
    parser = argparse.ArgumentParser(description='IBM classic')
    parser.add_argument('--time', type=int, default=600, help='Total time')
    args = parser.parse_args()
    all_instances = [os.path.join('instances', f) for f in listdir('instances') if
                     isfile(os.path.join('instances', f)) and not f.startswith('.')]
    all_instances.sort()
    solver_multiple_instances(all_instances, args.time)