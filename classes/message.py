class Message:
    def __init__(self, sender_uid: int, initial_values: list, info_levels: list[int], key: int = None):
        self.__initial_values = initial_values.copy()
        self.__info_levels = info_levels.copy()
        self.__sender_uid = sender_uid
        self.__key = key

    def get_info_levels(self) -> list[int]:
        return self.__info_levels.copy()

    def get_initial_values(self) -> list:
        return self.__initial_values.copy()

    def get_sender_uid(self) -> int:
        return self.__sender_uid

    def get_key(self):
        return self.__key

    def __str__(self):
        return f'sender: {self.__sender_uid}, key: {self.__key}, info_levels: {self.__info_levels}'