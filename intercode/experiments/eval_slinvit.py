import argparse, config, openai, os, random, re
import copy

from intercode.envs import (
    BashEnv, SqlEnv, ACTION_EXEC
)
import simplejson as json
from tqdm import tqdm
from typing import Dict
from experiments.utils import TemplateDirectReason, ACTION_PARSER_MAP

parser = argparse.ArgumentParser(description='Plan & Solve evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, help='path to dataset to evaluate on')
parser.add_argument('--env', choices=['sql', 'bash'], help='Intercode environment to run eval on')
parser.add_argument('--image_name', type=str, help='name of docker image to build environment with')
parser.add_argument('--log_dir', type=str, help='folder to save experiment run log file to')
parser.add_argument('--proportion', type=float, help="proportion of the dataset to use for evaluation")
parser.add_argument('--refine', action='store_true', help="whether to run refine step")
parser.add_argument('--refine_turns', type=int, help="number of turns to run refine step for")
parser.add_argument('--seed', type=int, help="seed for randomness")
parser.add_argument('--verbose', action='store_true', help="print out logs")
args = parser.parse_args()

SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell"
}

def preprocess_sql(record: Dict) -> str:
    db = record["extra"]["db"]
    return f"use {db}"

# Set OpenAPI key from environment or config file
api_key = os.environ.get("OPENAI_API_KEY")
if (api_key is None or api_key == "") and os.path.isfile(os.path.join(os.getcwd(), "keys.cfg")):
    cfg = config.Config('keys.cfg')
    api_key = cfg["OPENAI_API_KEY"]
assert(api_key != None)
openai.api_key = api_key
openai.api_version = "2023-06-01-preview"

def llm(messages, top_p=1.0, temperature=1.0):
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
            n=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        action_sim = 'None'
        return action_sim

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args
        self.horizon = 5
        self.sample_per_node = 1
        self.depth = 1
        self.replan = True

        # Set environment (No logging for env)
        self.env = None
        if args.env == 'sql':
            self.env = SqlEnv(image_name=args.image_name,
                data_path=args.data_path, preprocess=preprocess_sql)
            self.temp_envs = [SqlEnv(image_name=f'intercode-sql{i+1}', data_path=args.data_path, preprocess=preprocess_sql) for i in range(self.sample_per_node ** self.depth)]
        elif args.env == 'bash':
            self.env = BashEnv(image_name=args.image_name,
                data_path=args.data_path)
            self.temp_envs = [BashEnv(image_name=f'intercode-nl2bash{i}', data_path=args.data_path) for i in range(self.sample_per_node ** self.depth)]
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{self.env.name}_plan_solve_{args.data_path[-6]}.json"
        if args.refine and args.refine_turns:
            log_file_name = f"{self.env.name}_plan_solve_refine_{args.refine_turns}_turns.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}

        # Define dialogue, template, parser
        self.temp_plans = []
        self.temp_actions = []
        self.temp_value = []
        self.template = TemplateDirectReason(self.args.env.upper(), SETTING_MAP[self.args.env])
        self.parser = ACTION_PARSER_MAP[self.args.env]

    def run_expr(self):
        try:
            indices = range(len(self.env.data_loader))
            if self.args.seed and self.args.proportion:
                indices = random.Random(self.args.seed).choices(list(indices),
                    k=int(len(indices) * self.args.proportion))[6:]
            for idx in tqdm(indices, disable=self.args.verbose):
                self.opt_dial = []
                # Reset variables per task
                self.env.reset(idx)
                for i, temp_e in enumerate(self.temp_envs):
                    temp_e.reset(idx)
                    self.temp_envs[i] = temp_e
                observation, self.dialogue = None, []
                turn_history = {"actions": [], "observations": [], "rewards": [], "steps": [], "valid_action": []}
                record = self.env.data_loader.get(idx)
                if self.args.verbose:
                    print(f'------\nQuery {idx}: {self.env.query}')
                # Get plan
                self.opt_dial.append({"role": "system", "content": self.template.get_init_msg(self.sample_per_node)})
                self.opt_dial.append({"role": "user", "content": self.template.get_query_msg(self.env.query)})
                observation, info = None, None
                done = False
                self.env_history = []
                self.log_plan = []
                self.rollout = 1
                for idx_plan in range(self.horizon):
                    if done:
                        break
                    self.temp_value = [0.0 for _ in range(self.sample_per_node ** self.depth)]
                    self.temp_plans = [copy.deepcopy(self.opt_dial) for _ in range(self.sample_per_node ** self.depth)]
                    self.temp_log_actions = [copy.deepcopy(self.env_history) for _ in range(self.sample_per_node ** self.depth)]
                    self.temp_log_sim_actions = [copy.deepcopy([]) for _ in range(self.sample_per_node ** self.depth)]
                    self.temp_log_plans = [copy.deepcopy(self.log_plan) for _ in range(self.sample_per_node ** self.depth)]
                    for dep in range(self.depth):
                        layer_samples = self.sample_per_node ** dep
                        for parent_idx in range(layer_samples):
                            parent_effective_start_idx = self.sample_per_node ** (self.depth - dep) * parent_idx
                            prev_plan = 'Previous actions and outputs:\n'
                            if dep == 0 and idx_plan == 0:
                                prev_plan += 'None.\n\n'
                            for count, k in enumerate(self.temp_log_plans[parent_effective_start_idx]):
                                prev_plan += f"{count+1}. {k}\n"
                            self.temp_plans[parent_effective_start_idx].append({"role": "user", "content": prev_plan})
                            responses = llm(self.temp_plans[parent_effective_start_idx])
                            self.temp_plans[parent_effective_start_idx].pop()
                            plan = re.findall(f"Choice \d\:\s+(.*?)(?=\n|\Z)", responses, re.DOTALL)
                            plan = [p[2:].strip() if p.startswith(f"{idx_plan + dep + 1}.") else p for p in plan]
                            pad = ['Done! No action is needed.' for _ in range(self.sample_per_node - len(plan))]
                            plan = plan[:self.sample_per_node] + pad  # pad the response_list
                            for i, action in enumerate(plan):
                                if done:
                                    break
                                effect_start_idx = parent_effective_start_idx + self.sample_per_node ** (self.depth - dep - 1) * i
                                effect_end_idx = parent_effective_start_idx + self.sample_per_node ** (self.depth - dep - 1) * (i + 1)

                                if isinstance(observation, str) and len(observation) > 1000:
                                    observation = observation[:1000]
                                elif isinstance(observation, list) and len(observation) > 25:
                                    observation = observation[:25]
                                action_parsed, is_code = self.parser(action)
                                for env_id in range(effect_start_idx, effect_end_idx):
                                    if not is_code:
                                        observation = self.template.get_retry_msg()
                                        valid_action, reward = False, 0
                                    else:
                                        observation, reward, _, info = self.temp_envs[env_id].step(action_parsed)
                                        _, reward, _, info = self.temp_envs[env_id].step("submit")
                                    self.temp_log_plans[env_id].append(f"Action: {action}\nOutput:{observation}\n")
                                    self.temp_log_actions[env_id].append(action_parsed)
                                    if reward == 1:
                                        done = True
                                        self.temp_value[env_id] = reward
                                    elif dep == self.depth - 1:
                                        sim_plan = copy.deepcopy(self.temp_log_plans[env_id])
                                        if self.horizon - idx_plan - dep > 0:
                                            for _ in range(self.horizon - idx_plan - dep):
                                                prev_plan = 'Previous actions and outputs:\n'
                                                for count, k in enumerate(sim_plan):
                                                    prev_plan += f"{count + 1}. {k}\n"
                                                prev_plan += "\nNext action: "
                                                prompt = []
                                                prompt.append({"role": "system", "content": self.template.get_simulation_msg()})
                                                prompt.append({"role": "user", "content": self.template.get_query_msg(self.env.query)})
                                                prompt.append({"role": "user", "content": prev_plan})
                                                action_sim = llm(prompt)
                                                action_parsed, is_code = self.parser(action_sim)
                                                self.temp_log_sim_actions[env_id].append(action_parsed)
                                                if not is_code:
                                                    observation = self.template.get_retry_msg()
                                                    valid_action, reward = False, 0
                                                else:
                                                    observation, reward, _, info = self.temp_envs[env_id].step(action_parsed)
                                                    valid_action = info[ACTION_EXEC]
                                                    _, reward, _, info = self.temp_envs[env_id].step("submit")
                                                    if reward == 1:
                                                        self.rollout = len(self.temp_log_sim_actions[0])
                                                        break
                                                sim_plan.append(f"Action: {action_sim}\nOutput:{observation}\n")
                                            self.temp_value[env_id] = reward
                    if all(elem == self.temp_value[0] for elem in self.temp_value):
                        argmax = random.randrange(len(self.temp_value))
                    else:
                        argmax = self.temp_value.index(max(self.temp_value))
                    self.log_plan.append(self.temp_log_plans[argmax][len(self.log_plan)])

                    self.opt_dial = copy.deepcopy(self.temp_plans[argmax])
                    action = self.temp_log_actions[argmax][len(self.env_history)]
                    self.env_history.append(action)
                    observation, _, _, info = self.env.step(action)
                    valid_action = info[ACTION_EXEC]
                    _, reward, _, info = self.env.step("submit")
                    print(reward, self.rollout)
                    if reward == 1:
                        done = True
                    if self.rollout > 1:
                        for action_sim_opt in self.temp_log_sim_actions[argmax]:
                            self.env_history.append(action_sim_opt)
                            observation, _, _, info = self.env.step(action_sim_opt)
                            valid_action = info[ACTION_EXEC]
                            _, reward, _, info = self.env.step("submit")
                            if reward == 1:
                                done = True
                                break
                    if not done:
                        for ii, tem_e in enumerate(self.temp_envs):
                            tem_e.reset(idx)
                            for prev_step in self.env_history:
                                observation, _, _, info = self.env.step(prev_step)
                            self.temp_envs[ii] = tem_e
                    turn_history["actions"].append(action)
                    turn_history["rewards"].append(reward)
                    turn_history["observations"].append(str(observation)) # To avoid serialization issues
                    turn_history["valid_action"].append(valid_action)
                
                # Logging
                max_reward, max_reward_idx = 0, -1
                if len(turn_history["rewards"]) > 0:
                    max_reward = max(turn_history["rewards"])
                    max_reward_idx = turn_history["rewards"].index(max_reward)

                log_episode = {
                    "environment": self.env.name,
                    "dataset": self.args.data_path,
                    "task_id": idx,
                    "query": self.env.query,
                    "turn_history": turn_history,
                    "info": info,
                    "summary": {
                        "max_reward": max_reward,
                        "max_reward_idx": max_reward_idx,
                    }
                }
                if "extra" in record and "hardness" in record["extra"]:
                    log_episode["hardness"] = record["extra"]["hardness"]
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump({
                    "meta": vars(self.args),
                    "logs": self.log_data
                }, fp, indent=2)
            self.env.close()
            for temp_e in self.temp_envs:
                temp_e.close()


if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()
