import collections
import os
import random
import sys
import time

import numpy as np
import argparse
from os import listdir
from os.path import isfile
from typing import List, Tuple

from docplex.cp import modeler
from docplex.cp.expression import interval_var
from docplex.cp.model import CpoModel, context

import pandas as pd
from docplex.cp.modeler import end_of, minimize, end_before_start, no_overlap
from docplex.cp.parameters import CpoParameters


def solve_one_instance(instance_path: str, workers_nb: int = 1) -> Tuple[int, int, int, float]:
    mdl = CpoModel()
    stp = mdl.create_empty_solution()

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

    mdl.set_starting_point(stp)

    mdl.add(minimize(modeler.max(end_of(task_interval) for task_interval in all_tasks.values())))


    params = CpoParameters()
    params.LogVerbosity = 'Quiet'
    params.Workers = workers_nb

    # Solve model.
    res = mdl.solve(TimeLimit=600, Workers=workers_nb, LogVerbosity='Quiet')
    correct_solution = res.is_solution()
    assert correct_solution, "No solution found"
    return res.get_objective_value(), jobs_count, machines_count, time.time() - start


def solver_multiple_instances(instances: List[str], workers_nb:int = 1, csv_name: str = 'results.csv'):
    import wandb
    wandb.init(project='test_benchmark_cp_parallel_windows', name='single_solving')
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    results = collections.defaultdict(list)
    for _ in range(5):
        for instance in instances:
            if instance.split("/")[1].startswith('tai'):
                #pbar.set_description(f'Currently solving instance {"".join(instance.split("/")[1:])} with {workers_nb} workers')
                results['name'].append(instance)
                output = solve_one_instance(instance, workers_nb)
                results['jobs'].append(output[1])
                results['machine'].append(output[2])
                results['objective'].append(output[0])
                results['time'].append(output[3])
                results['nb_workers'].append(workers_nb)
                print(f'for {instance} we have {output[0]}')
    df = pd.DataFrame.from_dict(results)
    df.to_csv(csv_name)
    print(df.to_string())
    wandb.Table(dataframe=df)
    wandb.log({'results': wandb.Table(dataframe=df)})


if __name__ == '__main__':
    if sys.platform.startswith("linux"):
        context.solver.local.execfile = (
            "/opt/ibm/ILOG/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer"
        )
    elif sys.platform == "darwin":
        context.solver.local.execfile = (
            "/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer"
        )
    parser = argparse.ArgumentParser(description='Main')
    all_instances = [os.path.join('instances', f) for f in listdir('instances') if
                     isfile(os.path.join('instances', f)) and not f.startswith('.')]
    all_instances.sort()
    args = parser.parse_args()
    for cpu_workers in [1] + list(range(2, 32, 2)):
        solver_multiple_instances(all_instances, cpu_workers)