"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
import alfworld
import alfworld.agents.environment
from env_history import EnvironmentHistory
from typing import List, Dict, Any, Tuple
from statistics import mean
import math
import copy
from tenacity import retry, stop_after_attempt, retry_if_exception_type, retry_if_not_exception_type
import time
# Set OpenAPI key from environment or config file
api_key = os.environ.get("OPENAI_API_KEY")
assert(api_key != None)
openai.api_key = api_key
openai.api_version = "2023-06-01-preview"

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
VALUE_PROMPT_FILE = 'alfworld_value.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

with open(os.path.join(FOLDER, VALUE_PROMPT_FILE), 'r') as f:
    value_d = json.load(f)

@retry(
    stop=stop_after_attempt(4),
    retry=retry_if_not_exception_type((ValueError, OSError))
)
def call_openai_api(prompt, stop, n, temperature=2.0):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        logprobs=0,
        temperature=temperature,
        max_tokens=100,
        top_p=0.8,
        n=n,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response

def llm(prompt, stop=["\n"], n=8, temperature=2.0):
    response = call_openai_api(prompt, stop, n=n, temperature=temperature)
    for tries in range(1, 10):
        if response == {}:
            response = call_openai_api(prompt, stop, n=n, temperature=temperature)
        elif all(item["text"].strip() == '' for item in response["choices"]):
                response = call_openai_api(prompt, stop, n=n, temperature=temperature)
        else:
            break
    response_list = []
    for choice in response["choices"]:
        try:
            response_text = choice["text"].strip()
            response_prob = math.exp(mean(choice["logprobs"]["token_logprobs"]))
            response_list.append((response_text, response_prob))
        except:
            pass
    if n > 1:
        response_list = sorted(response_list, key=lambda x: x[1], reverse=True)
    try:
        return response_list[0], response_list
    except:
        return ('skip', 0.0), response_list


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


sample_per_node = 2
depth = 2  # depth - 1, as the first layer is not counted
scale = 0.1
replan = False
def alfworld_run(env, base_prompt, value_prompt, memory: List[str], to_print=True, ob='', temp_envs=None, temp_envs_before_init=None, num_reset=None) -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
        env_value_history = EnvironmentHistory(value_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
        env_value_history = EnvironmentHistory(value_prompt, ob, memory, [])

    env_history.reset()
    env_value_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    env_value_estimate = 0.0
    while cur_step < 50:
        temp_history = [copy.deepcopy(env_history) for _ in range(sample_per_node ** depth)]
        temp_value_history = [copy.deepcopy(env_value_history) for _ in range(sample_per_node ** depth)]
        temp_reward = [0.0 for _ in range(sample_per_node ** depth)]
        value_estimate = [env_value_estimate for _ in range(sample_per_node ** depth)]
        for dep in range(depth):
            layer_samples = sample_per_node ** dep
            for parent_idx in range(layer_samples):
                parent_effective_start_idx = sample_per_node ** (depth - dep) * parent_idx
                _, response_list = llm(str(temp_history[parent_effective_start_idx]) + "\n>", stop=['\n'])
                response_list = list(dict(response_list).items())
                response_list = [(key, res) for key, res in response_list if key != '']
                if response_list == []:
                    continue
                response_list = response_list[:sample_per_node] + [response_list[0]] * (sample_per_node - len(response_list))
                (value_response, _), _ = llm(str(temp_value_history[parent_effective_start_idx]) + "\n>", stop=['\n'], n=1, temperature=0.0)
                nonempty_i = 0
                for i, (res, probability) in enumerate(response_list):
                    effect_start_idx = parent_effective_start_idx + sample_per_node ** (depth - dep - 1) * i
                    effect_end_idx = parent_effective_start_idx + sample_per_node ** (depth - dep - 1) * (i + 1)
                    if nonempty_i >= sample_per_node:
                        break
                    nonempty_i += 1
                    for env_id in range(effect_start_idx, effect_end_idx):
                        if res.startswith('think:'):
                            think_action = True
                        resp = res
                        prob = probability
                        while think_action:
                            (resp, prob), _ = llm(str(temp_history[env_id]) + "\n>", stop=['\n'])
                            if not resp.startswith('think:'):
                                think_action = False
                            else:
                                observation = 'OK.'
                                temp_history[env_id].add("action", resp)
                                temp_history[env_id].add("observation", observation)
                        observation, _, _, _ = temp_envs[env_id].step([resp])
                        observation = process_ob(observation[0])
                        if value_response.startswith('critic:'):
                            temp_value_history[env_id].add("critic", value_response)
                            str_value = value_response.partition('=')[-1]
                            if str_value.endswith('.'):
                                value_estimate[env_id] = float(str_value[:-1])
                            elif str_value != '':
                                value_estimate[env_id] = float(str_value)
                        temp_reward[env_id] += prob * scale
                        temp_history[env_id].add("action", resp)
                        temp_history[env_id].add("observation", observation)
                        temp_value_history[env_id].add("action", "OK.")
                        temp_value_history[env_id].add("observation", observation)
                        if dep == depth - 1:  # terminal value
                            (value_response, _), _ = llm(str(temp_value_history[env_id]) + "\n>", stop=['\n'], n=1, temperature=0.0)
                            if value_response.startswith('critic:'):
                                temp_value_history[env_id].add("critic", value_response)
                                str_value = value_response.partition('=')[-1]
                                if str_value.endswith('.'):
                                    value_estimate[env_id] = float(str_value[:-1])
                                elif str_value != '':
                                    value_estimate[env_id] = float(str_value)
        rew_value = temp_reward + value_estimate
        argmax = rew_value.index(max(rew_value))
        env_value_estimate = value_estimate[argmax]
        rollout = 1 if replan else (len(temp_history[argmax]._history)-len(env_history._history)) // 2
        for _ in range(rollout):
            if len(temp_history[argmax]._history) > len(env_history._history):
                action = temp_history[argmax]._history[len(env_history._history)]['value']
            else:
                action = 'skip'
            env_history.add("action", action)
            if len(temp_value_history[argmax]._history) > len(env_value_history._history):
                value_response = temp_value_history[argmax]._history[len(env_value_history._history)]['value']
                if value_response.startswith('critic'):
                    env_value_history.add("critic", value_response)
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith('think:'):
                observation = 'OK.'
            else:
                env_value_history.add("action", 'OK.')
                env_value_history.add("observation", observation)

            env_history.add("observation", observation)
            if to_print:
                print(f'{cur_step}> {action}\n{observation}')
                sys.stdout.flush()
            if reward:
                return env_history, True
            cur_step += 1
        for ii, tem_e in enumerate(temp_envs):
            tem_e = temp_envs_before_init[ii].init_env(batch_size=1)
            for _ in range(num_reset + 1):  # the first num_reset makes tem_e at the same environment as env
                _, _ = tem_e.reset()
            for prev_step in env_history._history:
                if prev_step['label'] == "action" and not prev_step["value"].startswith('think:'):
                    _, _, _, _ = tem_e.step([prev_step["value"]])
            temp_envs[ii] = tem_e
    return env_history, False

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    temp_envs_before_init = [copy.deepcopy(env) for _ in range(sample_per_node ** depth)]
    env = env.init_env(batch_size=1)
    temp_envs = [tem.init_env(batch_size=1) for tem in temp_envs_before_init]

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        print(f'{z} / {len(env_configs)}')
        ob, info = env.reset()
        for tem_e in temp_envs:
            tem_ob, tem_info = tem_e.reset()

        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                value_prompt = 'You are a value critic of states in a household task. Here are two examples.\n' + value_d[f'value_{v}_1'] + value_d[f'value_{v}_0']
                final_env_history, is_success = alfworld_run(env, base_prompt, value_prompt, env_config["memory"] if use_memory else [],
                                                             to_print=True, ob=ob, temp_envs=temp_envs,
                                                             temp_envs_before_init=temp_envs_before_init, num_reset=z)

                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()
    for tem_e in temp_envs:
        tem_e.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs