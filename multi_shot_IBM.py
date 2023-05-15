import collections
import argparse
import ray
import os
import time
import random
from os import listdir
from os.path import isfile
from typing import List
from docplex.cp.model import *

import numpy as np
from tqdm import tqdm

import pandas as pd


def solve_using_partial_solution(
        partial_solution: np.ndarray,
        solution: np.ndarray,
        horizon: int,
        all_machines,
        jobs_data,
        run_for=60.0,
        workers=2,
        one_solution_only=False
) -> (np.ndarray, int):
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
    stp = mdl.create_empty_solution()

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = "_%i_%i" % (job_id, task_id)

            interval_variable = interval_var(size=duration, name="interval" + suffix)
            interval_variable.set_end_max(horizon + 1)

            if not one_solution_only:
                if partial_solution[job_id][task_id] != -1:
                    mdl.add(equal(start_of(interval_variable), solution[job_id][task_id]))
                stp.add_interval_var_solution(interval_variable, True, solution[job_id][task_id],
                                              solution[job_id][task_id] + duration, duration)
            machine_to_intervals[machine].append(interval_variable)
            all_tasks[job_id, task_id] = interval_variable

    # Create and add disjunctive constraints.
    for machine in all_machines:
        mdl.add(no_overlap(machine_to_intervals[machine]))

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(1, len(job)):
            mdl.add(end_before_start(all_tasks[job_id, task_id - 1], all_tasks[job_id, task_id]))


    if not one_solution_only:
        mdl.set_starting_point(stp)
        mdl.add(minimize(max(end_of(task_interval) for task_interval in all_tasks.values())))

    # Solve model.
    res = mdl.solve(TimeLimit=run_for, Workers=workers, LogVerbosity='Quiet')
    solution_new = np.zeros_like(solution)
    if res and res.is_solution():
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                solution_new[job_id][task_id] = int(res.get_var_solution(all_tasks[job_id, task_id]).start)
        objective = res.get_objective_value()
        if one_solution_only:
            objective = 0
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    objective = max(res.get_var_solution(all_tasks[job_id, task_id]).end, objective)
        return solution_new, objective, res.is_solution_optimal(), res.get_solve_time()
    return solution, float('inf'), False, run_for


@ray.remote
def parallel_solve_using_partial_solution(
        partial_solution: np.ndarray,
        solution: np.ndarray,
        horizon: int,
        all_machines,
        jobs_data,
        run_for=60.0,
        workers=2,
        one_solution_only=False
) -> (np.ndarray, int):
    return solve_using_partial_solution(
        partial_solution, solution, horizon, all_machines, jobs_data, run_for, workers, one_solution_only
    )


def solve_one_instance(instance_path: str, limit_time: int):

    jobs_data = []
    machines_count = 0
    jobs_count = 0

    instance_file = open(instance_path, "r")
    line_str = instance_file.readline()
    line_cnt = 1
    objective = 0
    while line_str:
        data = []
        split_data = line_str.split()
        if line_cnt == 1:
            jobs_count, machines_count = int(split_data[0]), int(split_data[1])
        else:
            i = 0
            while i < len(split_data):
                machine, op_length = int(split_data[i]), int(split_data[i + 1])
                data.append((machine, op_length))
                i += 2
                objective += op_length
            jobs_data.append(data)
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()

    all_machines = range(machines_count)

    number_op = jobs_count * machines_count

    start = time.time()

    total_cores = 32

    output_iter = ray.get([parallel_solve_using_partial_solution.remote(
        None,
        np.zeros((jobs_count, machines_count), dtype=int),
        objective,
        all_machines,
        jobs_data,
        run_for=limit_time,
        workers=total_cores,
        one_solution_only=True
    )])

    output_iter = np.array(output_iter, dtype=object)
    solutions = np.array(output_iter[:, 0], dtype=object)
    results = np.array(output_iter[:, 1], dtype=float)
    optimum = np.array(output_iter[:, 2], dtype=bool)
    wall_times = np.array(output_iter[:, 3], dtype=float)
    best_pop = np.argmin(results)
    solution = solutions[best_pop]
    previous_obj = objective
    objective = results[best_pop]

    print(f'first single-shot found {objective}')

    nb_workers = 16

    window_size = number_op // nb_workers

    iter_nb = 0

    run_for_seconds = limit_time // 10

    while (time.time() - start) < limit_time:
        flatten_sol: np.ndarray = solution.flatten()
        flatten_sol: List = flatten_sol.tolist()
        flatten_sol.sort()
        workers = []
        samples = list(range(0, len(flatten_sol), window_size))
        for tmp in samples:
            start_time = flatten_sol[tmp]
            partial_solution = np.copy(solution)
            partial_solution[partial_solution >= start_time] = -1
            workers.append(
                    parallel_solve_using_partial_solution.remote(
                        partial_solution,
                        solution,
                        int(objective),
                        all_machines,
                        jobs_data,
                        run_for=run_for_seconds
                        if limit_time - (time.time() - start) > run_for_seconds and nb_workers > 1
                        else limit_time - (time.time() - start),
                        workers=total_cores // nb_workers
                    )
                )
        output_iter = ray.get(workers)
        output_iter = np.array(output_iter, dtype=object)
        solutions = np.array(output_iter[:, 0], dtype=object)
        results = np.array(output_iter[:, 1], dtype=float)
        optimum = np.array(output_iter[:, 2], dtype=bool)
        wall_times = np.array(output_iter[:, 3], dtype=float)
        best_pop = np.argmin(results)
        solution = solutions[best_pop]
        previous_obj = objective
        objective = results[best_pop]
        iter_nb += 1
        if previous_obj == objective:
            run_for_seconds = limit_time
        print(
            f"this iter new objective {objective}, spent total {(time.time() - start) // 60} min {(time.time() - start) % 60} sec {nb_workers} workers"
        )

    return objective, jobs_count, machines_count, time.time() - start


def solver_multiple_instances(instances: List[str], total_time: int, csv_name: str = "results.csv"):
    import wandb
    wandb.init(project='CP_ICAPS', entity='cp_project', name=f'parallel_window_ibm_{total_time // 60}min')
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    results = collections.defaultdict(list)
    pbar = tqdm(instances, unit=" instances")
    for instance in pbar:
        pbar.set_description(f'Currently solving instance {"".join(instance.split("/")[1:])}')
        results['name'].append("".join(instance.split("/")[1:]))
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


if __name__ == "__main__":
    context.solver.local.execfile = '/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
    parser = argparse.ArgumentParser(description='IBM Parallel')
    parser.add_argument('--time', type=int, default=600, help='Total time')
    args = parser.parse_args()
    all_instances = [
        os.path.join("instances", f)
        for f in listdir("instances")
        if isfile(os.path.join("instances", f)) and not f.startswith(".")
    ]
    all_instances.sort()
    solver_multiple_instances(all_instances, args.time)


