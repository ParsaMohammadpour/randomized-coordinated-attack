from random import randint
from classes.message import Message
from typing import Callable


class Process:
    def __init__(self, uid: int, initial_val, process_count: int, default_decision_val: int,
                 prob_func: Callable[[], bool], is_key_generator: bool = False):
        """

        :param uid: process uid (number between 1 to n)
        :param initial_val: initial value of process (must be in decision)
        :param process_count: number of all processes
        :param default_decision_val: default value of decision
        :param prob_func: probability of not having link failure
        :param is_key_generator: whether to generate a key for process
        """
        self.__key = None
        self.__uid_index = uid - 1  # index of process uid in process list which is their uid - 1
        self.__initial_decision_val = initial_val
        self.__process_count = process_count
        self.__default_decision_val = default_decision_val
        self.__initial_process_values()
        self.__initial_info_levels()
        self.__neighbors = None
        self.__received_messages = []
        self.__is_key_generator = is_key_generator
        self.__prob_func = prob_func

    def __initial_process_values(self):
        self.__process_values = [None] * self.__process_count
        self.__process_values[self.__uid_index] = self.__initial_decision_val

    def __update_initial_values(self, values: list):
        if len(values) != self.__process_count:
            raise ValueError('Number of values must be equal to the number of processes')
        for i in range(len(values)):
            if values[i] is not None:
                self.__process_values[i] = values[i]

    def __initial_info_levels(self):
        self.__info_levels = [-1] * self.__process_count
        self.__info_levels[self.__uid_index] = 0

    def __update_my_info_level(self):
        self.__info_levels[self.__uid_index] = min(self.__info_levels) + 1

    def __update_info_levels(self, info_levels: list[int]):
        if not len(info_levels) == len(self.__info_levels):
            raise ValueError(
                f'information levels must have equal length. {len(self.__info_levels)} and {len(info_levels)} are not equal')
        for i in range(len(info_levels)):
            self.__info_levels[i] = max(info_levels[i], self.__info_levels[i])
        self.__update_my_info_level()

    def set_neighbors(self, neighbors: list):
        if self.__neighbors is None:
            self.__neighbors = neighbors

    def generate_key(self, round_number: int, key: int = None) -> int:
        if not self.__is_key_generator:
            raise ValueError('Only key generator process can generates key')
        if key is None:
            self.__key = randint(1, round_number)
        else:
            if key > round_number:
                raise ValueError('Key must be less than round number')
            self.__key = key
        return self.__key

    def __set_key_from_msg(self, key: int = None):
        if key is not None:
            self.__key = key

    def get_key(self) -> int:
        return self.__key

    def get_uid(self) -> int:
        return self.__uid_index + 1  # Because we saved uid index (process index)

    def get_initial_val(self) -> int:
        return self.__initial_decision_val

    def get_information_level(self) -> int:
        return self.__info_levels[self.__uid_index]

    def __generate_message(self) -> Message:
        return Message(self.get_uid(), self.__process_values, self.__info_levels, self.__key)

    def send_messages(self):
        msg = self.__generate_message()
        for p in self.__neighbors:
            can_send = self.__prob_func()
            if can_send:
                p.receive_message(msg)
            else:
                p.receive_message(None)

    def receive_message(self, msg: Message):
        self.__received_messages.append(msg)

    def update_from_messages(self, log: bool = False) -> [int, int]:
        if log:
            print(f'process: {self.get_uid()} receive messages list: {[str(msg) for msg in self.__received_messages]}')
            print(
                f'process: {self.get_uid()} before update state: key: {self.__key}, info_levels: {self.__info_levels}, '
                f'initial_values: {self.__process_values}, messages_count: {len(self.__received_messages)}')
        link_failures = 0
        successes = 0
        for msg in self.__received_messages:
            if msg is not None:
                successes += 1
                self.__update_single_message(msg)
            else:
                link_failures += 1
        self.__received_messages.clear()
        if log:
            print(
                f'process: {self.get_uid()} after update state: key: {self.__key}, info_levels: {self.__info_levels}, '
                f'initial_values: {self.__process_values}, messages_count: {len(self.__received_messages)}')
        return successes, link_failures

    def __update_single_message(self, msg: Message):
        self.__set_key_from_msg(msg.get_key())
        self.__update_info_levels(msg.get_info_levels())
        self.__update_initial_values(msg.get_initial_values())

    def get_final_decision(self) -> any:
        if self.__key is None:
            return self.__default_decision_val
        if not self.get_information_level() >= self.__key:
            return self.__default_decision_val
        if len(set(self.__process_values)) > 1:
            return self.__default_decision_val
        return self.__initial_decision_val
