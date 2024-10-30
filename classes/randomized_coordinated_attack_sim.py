import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import sys
from random import randint
from classes.process import Process
import random as rd
from typing import Callable


def save_plot(path):
    generate_file(path)
    plt.savefig(path)


def generate_file(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


class CoordinatedAttackSim:
    def __init__(self, process_count: int, round_num: int, key_selector: int = 1, decisions_list: list | set = None,
                 process_initial_vals: list = None, communication_graph: nx.Graph = None,
                 prob_func: Callable[[], bool] = lambda: randint(0, 9) > 0, simulation_num: int = 100,
                 differ_initial_val_on_round: bool = False, base_path: str = 'result/complete-graph/failure-prob-50/',
                 fixed_key: int = None):
        """

        :param process_count: number of all processes
        :param round_num: number of rounds, r in the algorithm description
        :param key_selector: the process that generates the ky in the beginning
        :param decisions_list: all possible decisions, first one, index zero, is the default (if None, default is [0, 1] 0 for failure, 1 for commit)
        :param process_initial_vals: initial_values of processes, if none, it will be generated uniformly from decision_list
        :param communication_graph: the communication graph, default is complete graph with $process_count$ nodes
        :param prob_func: defining probability function that returns true or false (true for sending message and false for having link failure)
        :param simulation_num: number of simulations
        :param differ_initial_val_on_round: whether to generate new initial values for processes
        :param base_path: base path for saving data
        :param fixed_key: if we should focus on a fixed key
        """
        if decisions_list is None:
            decisions_list = [0, 1]
        self.__process_count = process_count
        self.__set_communication_graph(communication_graph)
        self.__key_selector_index = key_selector - 1  # storing index
        self.__round_num = round_num
        self.__decisions_list = decisions_list.copy()  # first index is the default when we don't have same values, default is zero or not commiting
        self.__initial_process_initial_values(process_initial_vals)
        # self.__initial_process() # we do it on each simulation iteration
        self.__prob_func = prob_func
        self.__simulation_num = simulation_num
        self.__df = None
        self.__differ_initial_val_on_round = differ_initial_val_on_round
        self.__base_path = base_path
        self.__fixed_key = fixed_key

    def __initial_process_initial_values(self, process_initial_vals: list):
        if process_initial_vals is None:
            # choose a uniformly distributed number between zero and max decision list index, and set decision of that index as initial decision for that process
            max_decision_index = len(self.__decisions_list) - 1
            process_initial_vals = [randint(0, max_decision_index) for _ in range(self.__process_count)]
        elif len(process_initial_vals) != self.__process_count:
            raise ValueError('Number of values must be equal to the number of processes')
        for initial_dec in set(process_initial_vals):
            if initial_dec not in self.__decisions_list:
                raise ValueError('Initial value must be one of the possible decisions')
        self.__process_initial_values = process_initial_vals

    def __initial_process(self):
        self.__processes = [self.__generate_process(i) for i in range(self.__process_count)]
        self.__set_neighbors()

    def __set_neighbors(self):
        for i in range(len(self.__processes)):
            neighbors = self.__communication_graph.neighbors(i)
            neighbor_processes = [self.__processes[i] for i in neighbors]
            self.__processes[i].set_neighbors(
                neighbor_processes)  # this neighbors function gives all processes index that input process can sends message to

    def __generate_process(self, index: int) -> Process:
        return Process(uid=index + 1, initial_val=self.__process_initial_values[index],
                       process_count=self.__process_count, default_decision_val=self.__decisions_list[0],
                       prob_func=self.__prob_func, is_key_generator=self.__key_selector_index == index)

    def __set_communication_graph(self, communication_graph: nx.Graph):
        if communication_graph is None:
            communication_graph = nx.complete_graph(self.__process_count)
        elif communication_graph.number_of_nodes() != self.__process_count:
            raise ValueError('communication_graph must have equal number of nodes with process count')
        self.__communication_graph = communication_graph.copy()

    def get_communication_graph(self) -> nx.Graph:
        return self.__communication_graph.copy()

    def plot_communication_graph(self, default_color: str = 'yellow', key_selector_color: str = 'orange',
                                 figsize: tuple[int, int] = (15, 15), save_path: str = None):
        nodes = list(self.__communication_graph.nodes())
        nodes_labels = [f'process: {node + 1}' for node in nodes]
        labels_dic = dict(zip(nodes, nodes_labels))
        color_maps = [default_color] * self.__process_count
        color_maps[self.__key_selector_index] = key_selector_color
        if figsize is not None:
            plt.figure(figsize=figsize)
        node_size = [5000] * len(nodes)
        nx.draw(self.get_communication_graph(), node_size=node_size, labels=labels_dic, with_labels=True,
                node_color=color_maps)
        if save_path is not None:
            save_plot(save_path)
        plt.show()

    def __reset(self):
        if self.__differ_initial_val_on_round:
            self.__initial_process_initial_values(None)
        self.__initial_process()

    def simulate(self, log: bool = True) -> pd.DataFrame:
        results = []
        for i in range(self.__simulation_num):
            if log:
                print(
                    f'********************************************* simulation round {i + 1} *********************************************')
                print()
            result_dict = self.__single_simulate(log=log)
            if log:
                print(f'sim round: {i + 1}, result:{result_dict}')
            results.append(result_dict)
        df = pd.DataFrame.from_dict(results)
        return df

    def __single_simulate(self, log: bool = True) -> dict[str, str]:
        self.__reset()
        self.__processes[self.__key_selector_index].generate_key(self.__round_num, self.__fixed_key)
        total_received_count = 0
        total_failed_count = 0
        for i in range(self.__round_num):
            received_count, failed_count = self.__simulate_round(log=log)
            total_received_count += received_count
            total_failed_count += failed_count
        decisions = [p.get_final_decision() for p in self.__processes]
        info_levels = [p.get_information_level() for p in self.__processes]
        keys = [p.get_key() for p in self.__processes]
        is_correct_answer, reason = self.__check_correct_answer(decisions, self.__process_initial_values.copy(),
                                                                total_failed_count, log=log)
        return {'initial-vals': str(self.__process_initial_values),
                'decisions': str(decisions),
                'info-levels': str(info_levels),
                'received-count': str(total_received_count),
                'failed-count': str(total_failed_count),
                'key': str(keys),
                'is-correct-answer': str(is_correct_answer),
                'reason': str(reason)}

    def __simulate_round(self, log: bool = False) -> [int, int]:
        rd.seed()  # generate a new seed in order to get new
        total_received = 0
        total_failed = 0
        # sending messages
        for p in self.__processes:
            p.send_messages()
        # receiving messages
        for p in self.__processes:
            received, failed = p.update_from_messages(log=log)
            total_received += received
            total_failed += failed
        return total_received, total_failed

    def __check_correct_answer(self, decisions: list, process_initial_vals: list, total_failed_count: int,
                               log: bool) -> [bool, str]:
        """
        checks if results are satisfying the requirements or not
        :param decisions:
        :param process_initial_vals:
        :param total_failed_count:
        :param log:
        :return: if the results are correct [True, None] otherwise returns False, reason(the rule that has been violated)
        """
        if self.__has_termination_violation(decisions):
            if log:
                print('Termination violation')
            return False, 'Termination'
        if self.__has_agreement_violation(decisions):
            if log:
                print('Agreement violation')
            return False, 'Agreement'
        if self.__has_validation_violation(decisions, process_initial_vals, total_failed_count):
            if log:
                print('Validation violation')
            return False, 'Validation'
        return True, '-'

    def __has_termination_violation(self, decisions: list) -> bool:
        # return true if any process has the decision value None(hasn't decided yet)
        return None in decisions

    def __has_agreement_violation(self, decisions: list) -> bool:
        # return if we have more than one distinguish decision among process decisions
        return len(set(decisions)) > 1

    def __has_validation_violation(self, decisions: list, process_initial_vals: list, total_failed_count: int) -> bool:
        if len(set(process_initial_vals)) == 1 and total_failed_count == 0:
            # if all processes have started with same initial value, and we had no link failure
            if decisions[0] != process_initial_vals[0]:
                # if we have decided on something else than initial value (everyone decided on same thing otherwise we would have been
                # violated agreement, and since above if we know that initial values are all same) we only check first elements to be equal.
                return True
        return False

    def get_results_compare(self, df: pd.DataFrame, col: str = 'is-correct-answer'):
        if df is None:
            raise ValueError('df is required')
        if type(df) != pd.DataFrame:
            raise ValueError('df must be of type pd.DataFrame')
        if col not in df:
            raise ValueError(f'column {col} is not in the dataframe columns: {df.columns}')
        return df.groupby(col).size()

    @staticmethod
    def __change_stdout_to_file(file_path):
        orig_stdout = sys.stdout
        generate_file(file_path)
        file = open(file_path, 'w')  # open a file in order to save logs in it instead of console
        sys.stdout = file
        return orig_stdout, file

    @staticmethod
    def __change_stdout_to_console(origi_stdout, file):
        sys.stdout = origi_stdout
        file.close()

    def simulate_n_times(self, simulate_num=100, log: bool = False, figsize: tuple[int, int] = (15, 5),
                         save_res: bool = False, key_values: list[int] = None):
        """
        apply the simulation simulate_num times and draw the plot
        :param simulate_num:
        :param log:
        :param figsize: figure size
        :param save_res: should save results or not
        :param key_values: different keys that would be applied to each group simulation
        :return: a dataframe containing the simulated results (two columns: 'success-count', 'failure-counts')
        """
        # preparing log files (setting system stdout to file instead of console)
        orig_std_out, file = None, None
        if log:
            path = self.__base_path + f'n-{self.__process_count}/r-{self.__round_num}/' + 'log.txt'
            orig_std_out, file = CoordinatedAttackSim.__change_stdout_to_file(path)

        # saving communication graph if we should
        if save_res:
            path = self.__base_path + f'n-{self.__process_count}/r-{self.__round_num}/' + 'communication-graph.png'
            self.plot_communication_graph(figsize=figsize, save_path=path)

        # setting key values:
        initial_key_value = self.__fixed_key
        if key_values is None:
            key_values = [self.__fixed_key] * simulate_num
        if len(key_values) != simulate_num:
            raise ValueError('number of keys is not equal to simulate num')

        # applying simulation simulate_num times
        success_counts, failure_counts = self.__apply_simulation_n_times(simulate_num=simulate_num, log=log,
                                                                         save_res=save_res, key_values=key_values)

        # restoring primary key value;
        self.__fixed_key = initial_key_value

        total_df = pd.DataFrame({'failure-counts': failure_counts, 'success-count': success_counts, 'key': key_values})
        self.__generate_simulation_n_times_df_plot(total_df=total_df, figsize=figsize, key_values=key_values)
        if save_res:
            # saving final comparison plot
            path = self.__base_path + f'n-{self.__process_count}/r-{self.__round_num}/' + 'results.png'
            save_plot(path)
            # saving final dataframe results
            path = self.__base_path + f'n-{self.__process_count}/r-{self.__round_num}/' + 'results.csv'
            total_df.to_csv(path, index=False)
        plt.show()

        # returning system stdout  for print to its original state
        if log:
            CoordinatedAttackSim.__change_stdout_to_console(orig_std_out, file)
        return total_df

    def __apply_simulation_n_times(self, simulate_num: int, log: bool = False, save_res: bool = False,
                                   key_values: list[int] = None):
        failure_counts = []
        success_counts = []

        for i in range(simulate_num):
            self.__fixed_key = key_values[i]
            df = self.simulate(log=log)
            if save_res:
                path = self.__base_path + f'n-{self.__process_count}/r-{self.__round_num}/' + f'dataframes/df-{i + 1}.csv'
                generate_file(path)
                df.to_csv(path)
            failure_counts.append(len(df[df['is-correct-answer'] == str(False)]))
            success_counts.append(len(df[df['is-correct-answer'] == str(True)]))
        return success_counts, failure_counts

    def __generate_simulation_n_times_df_plot(self, total_df: pd.DataFrame, figsize: tuple[int, int], key_values:list[int]=None):
        if key_values is None or (len(set(key_values)) == 1 and key_values[0] == None): # if key values is None or only contains None
            total_df.plot(kind='bar', y=['success-count', 'failure-counts'], figsize=figsize)
            plt.xlabel('group simulation index')
        else:
            total_df.plot(kind='bar', x='key', y=['success-count', 'failure-counts'], figsize=figsize)
            plt.xlabel('key')
        plt.ylabel('result count')
        plt.title('failure vs success count')
