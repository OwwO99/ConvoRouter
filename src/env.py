import time
import openai
import pickle
from loguru import logger
from task import Task
from dialogue import DialogSession


def query(client: openai.Client, model, messages, max_tokens=128, temperature=0.0):
    flag = True
    while flag:
        try:
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=None,
                temperature=temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            output = completions.choices[0].message.content.strip()
            flag = False
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    return output


class Env(object):
    def __init__(
            self,
            task: Task,
            user_client,
            user_model,
            eval_client,
            eval_model,
            system_clients,
            system_models,
            temperature=0.0,
            max_new_tokens=128,
            max_turn=8,
        ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_turn = max_turn

        self.user_client = user_client
        self.user_model = user_model
        self.eval_client = eval_client
        self.eval_model = eval_model
        self.system_clients = system_clients
        self.system_models = system_models

        self.task = task
    
    def _to_string_rep(self, state: DialogSession):
        das = []
        for (speaker, da, _) in state:
            if speaker == state.SYS:
                das.append(str(da))
        return "__".join(das)
        
    def _get_next_state(self, state: DialogSession, action):
        if self._to_string_rep(state) == '':
            prefetch_state = str(action)
        else:
            prefetch_state = self._to_string_rep(state) + "__" + str(action)
        if prefetch_state in self.realizations:
            next_state = self.realizations[prefetch_state]
            return next_state[0], next_state[1], next_state[2]
        return None, None, None
    
    def _add_new_realizations(self, state: DialogSession, C_t=None, done=None):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.realizations:
            self.realizations[hashable_state] = (state.copy(), C_t, done)

    def reset(self, case_id, realizations=None):
        self.cur_conver_step = 1
        logger.info('---------------step:1-------------')

        init_query, system_sys, user_sys = self.task.get_init_prompt(self.task.dataset[case_id])
        self.state = DialogSession(sys_name=self.task.system_role, user_name=self.task.user_role, case=self.task.dataset[case_id], 
                                   init_query=init_query, system_sys=system_sys, user_sys=user_sys)
        self.conversation_all = [{"turn": 1, "user": init_query}]
        logger.info({"role": self.task.user_role, "content": init_query})

        if realizations == None:
            self.realizations: dict = {}
            hashable_state = self._to_string_rep(self.state)
            self.realizations[hashable_state] = (self.state.copy(), 0, 0)
        else:
            self.realizations = realizations
        
        return self.state.copy()

    def step(self, model_id):
        next_state, next_reward, next_done = self._get_next_state(self.state, model_id)
        if next_state == None:
            # system
            system_response = self.generate_response(self.system_clients[model_id], self.system_models[model_id], self.state.system_messages)
            self.state.add_single(self.task.system_role, model_id, system_response)
            self.conversation_all[self.cur_conver_step - 1]['system'] = system_response
            self.conversation_all[self.cur_conver_step - 1]['model_id'] = int(model_id)
            logger.info({"role": self.task.system_role, "content": system_response})

            self.cur_conver_step += 1
            logger.info('---------------step:{}-------------'.format(self.cur_conver_step))

            # user
            user_response = self.generate_response(self.user_client, self.user_model, self.state.user_messages)
            self.state.add_single(self.task.user_role, -1, user_response)
            self.conversation_all.append({"turn": self.cur_conver_step, "user": user_response})
            logger.info({"role": self.task.user_role, "content": user_response})

            # eval
            reward, done = self.compute_reward()
            self._add_new_realizations(self.state, reward, done)
        else:
            # system
            system_response = next_state.history[-2][2]
            self.state.add_single(self.task.system_role, model_id, system_response)
            self.conversation_all[self.cur_conver_step - 1]['system'] = system_response
            self.conversation_all[self.cur_conver_step - 1]['model_id'] = int(model_id)
            logger.info({"role": self.task.system_role, "content": system_response})

            self.cur_conver_step += 1
            logger.info('---------------step:{}-------------'.format(self.cur_conver_step))

            # user
            user_response = next_state.history[-1][2]
            self.state.add_single(self.task.user_role, -1, user_response)
            self.conversation_all.append({"turn": self.cur_conver_step, "user": user_response})
            logger.info({"role": self.task.user_role, "content": user_response})

            # eval
            self.conversation_all[self.cur_conver_step - 2]['reward'] = next_reward
            self.conversation_all[self.cur_conver_step - 2]['done'] = next_done
            reward = next_reward
            done = next_done

        return self.state.copy(), reward, done
    
    def step_sim(self, state: DialogSession, model_id):
        system_response = self.generate_response(self.system_clients[model_id], self.system_models[model_id], state.system_messages)
        state.add_single(self.task.system_role, model_id, system_response)
        user_response = self.generate_response(self.user_client, self.user_model, state.user_messages)
        state.add_single(self.task.user_role, -1, user_response)

        return state
    
    def generate_response(self, client, model, messages):
        output = query(
            client=client,
            model=model,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return output

    def compute_reward(self):
        dial = ''
        for turn in self.state.history:
            dial += '%s: %s ' % (turn[0], turn[-1])
        eval_prompt = self.task.get_eval_prompt(self.state.case, dial)
        messages = [{"role": "user", "content": eval_prompt}]
        flag = True
        while flag:
            try:
                output = query(
                    client=self.eval_client,
                    model=self.eval_model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                reward, done = sum(eval(output)['checklist']), eval(output)['done']
                flag = False
            except Exception as e:
                logger.error(e)

        if len(self.state) == self.max_turn:
            logger.info('--> Maximum number of turns reached !')
            done = -1
        self.conversation_all[self.cur_conver_step - 2]['eval'] = output
        self.conversation_all[self.cur_conver_step - 2]['reward'] = reward
        self.conversation_all[self.cur_conver_step - 2]['done'] = done
        return reward, done
    
    def compute_reward_sim(self, state: DialogSession):
        dial = ''
        for turn in state.history:
            dial += '%s: %s ' % (turn[0], turn[-1])
        eval_prompt = self.task.get_eval_prompt(state.case, dial)
        messages = [{"role": "user", "content": eval_prompt}]
        flag = True
        while flag:
            try:
                output = query(
                    client=self.eval_client,
                    model=self.eval_model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                reward, done = sum(eval(output)['checklist']), eval(output)['done']
                flag = False
            except Exception as e:
                logger.error(e)
        if len(state) == self.max_turn:
            done = -1
        return reward, done
    
    def save_dia(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.realizations, file)
