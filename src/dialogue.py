class DialogSession():
    def __init__(self, sys_name, user_name, case, init_query, system_sys, user_sys) -> None:
        self.SYS = sys_name
        self.USR = user_name
        self.case = case
        self.init_query = init_query
        self.system_sys = system_sys
        self.user_sys = user_sys
        self.history = [(user_name, -1, init_query)]  # [(role, da, utt), ....]
        self.system_messages = [{"role": "system", "content": system_sys}, 
                                {"role": "user", "content": init_query}]
        self.user_messages = [{"role": "system", "content": user_sys},
                              {"role": "assistant", "content": init_query}]
    
    def from_history(self, history, system_messages, user_messages):
        self.history = history
        self.system_messages = system_messages
        self.user_messages = user_messages
        return self

    def copy(self):
        new_session = DialogSession(self.SYS, self.USR, self.case, self.init_query, self.system_sys, self.user_sys)
        new_session.from_history(self.history.copy(), self.system_messages.copy(), self.user_messages.copy())
        return new_session

    def add_single(self, role, da, utt):
        if len(self.history) % 2 != 0:
            assert(role == self.SYS)
            self.system_messages.append({"role": "assistant", "content": utt})
            self.user_messages.append({"role": "user", "content": utt})
        else:
            assert(role == self.USR)
            self.user_messages.append({"role": "assistant", "content": utt})
            self.system_messages.append({"role": "user", "content": utt})
        self.history.append((role, da, utt))
        return
    
    def get_turn_utt(self, turn, role):
        if role == self.SYS:
            return self.history[turn * 2 + 1][-1]
        else:
            return self.history[turn * 2][-1]
    
    def __iter__(self):
        return iter(self.history)

    def __len__(self):
        return len(self.history) //  2  # number of turns

    def __getitem__(self, index):
        return self.history[index]

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, DialogSession):
            return False
        return self.history == __o.history
