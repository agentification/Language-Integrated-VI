import argparse, json, os
from intercode.envs import (
    BashEnv, SqlEnv, ACTION_EXEC
)
from tqdm import tqdm
from typing import Dict
import openai
from experiments.utils import TemplateDirectReasonBaseline, ACTION_PARSER_MAP

parser = argparse.ArgumentParser(description='N-turn evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, help='path to dataset to evaluate on')
parser.add_argument('--env', choices=['sql', 'bash'], help='Intercode environment to run eval on')
parser.add_argument('--image_name', type=str, help='name of docker image to build environment with')
parser.add_argument('--log_dir', type=str, help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, help='max number of interaction turns')
parser.add_argument('--verbose', action='store_true', help="print out logs")
args = parser.parse_args()

SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell"
}

api_key = os.environ.get("OPENAI_API_KEY")
assert(api_key != None)
openai.api_key = api_key
openai.api_version = "2023-06-01-preview"
def llm(messages, stop=["\n"], top_p=1.0, temperature=1.0):
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,  # 512
            n=1,
        )
        # if top_p < 1.0:
        #     print('response: ', response.choices[0].message.content)
        return response.choices[0].message.content
    #    print('prompyt:', prompt)
    except Exception as e:
        print(e)
        action_sim = 'Reduce token length!'
        return action_sim


def preprocess_sql(record: Dict) -> str:
    db = record["extra"]["db"]
    return f"use {db}"

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args
        self.horizon = 6
        # Set environment (No logging for env)
        self.env = None
        if args.env == 'sql':
            self.env = SqlEnv(image_name=args.image_name,
                data_path=args.data_path, preprocess=preprocess_sql)
        elif args.env == 'bash':
            self.env = BashEnv(image_name=args.image_name,
                data_path=args.data_path)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{self.env.name}_multiturn_{args.max_turns}_turns.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}

        self.template = TemplateDirectReasonBaseline(self.args.env.upper(), SETTING_MAP[self.args.env])
        self.parser = ACTION_PARSER_MAP[self.args.env]

    def run_expr(self):
        try:
            for idx in tqdm(range(0,len(self.env.data_loader)), disable=self.args.verbose):
                # Reset variables per task
                self.env.reset(idx)
                observation, reward, valid_action = None, None, None
                turn_history = {"actions": [], "observations": [], "rewards": [], "valid_action": []}
                record = self.env.data_loader.get(idx)


                if self.args.verbose:
                    print(f'------\nQuery {idx}: {self.env.query}')
                done = False
                for turn in range(self.args.max_turns):
                    if done:
                        break
                    self.env.reset(idx)
                    prompt = []
                    sim_plan = []
                    for _ in range(self.horizon):
                        prev_plan = 'Previous actions and outputs:\n'
                        if len(sim_plan) == 0:
                            prev_plan += 'None.\n'
                        for count, k in enumerate(sim_plan):
                            prev_plan += f"{count + 1}. {k}\n"
                        prev_plan += "\nNext action: "
                        prompt.append({"role": "system", "content": self.template.get_simulation_msg()})
                        prompt.append({"role": "user", "content": self.template.get_query_msg(self.env.query)})
                        prompt.append({"role": "user", "content": prev_plan})
                        action_sim = llm(prompt)
                        action, is_code = self.parser(action_sim)

                        if not is_code:
                            reward = 0
                            observation = self.template.get_retry_msg()
                            valid_action = False
                        else:
                            observation, _, _, info = self.env.step(action)
                            valid_action = info[ACTION_EXEC]
                            _, reward, _, info = self.env.step("submit")
                        sim_plan.append(f"Action: {action_sim}\nOutput:{observation}\n")

                        # Logging
                        turn_history["actions"].append(action)
                        turn_history["observations"].append(str(observation)) # To avoid serialization issues
                        turn_history["rewards"].append(reward)
                        turn_history["valid_action"].append(valid_action)

                        # End episode upon perfect reward
                        if reward == 1:
                            done = True
                            break
                
                max_reward = max(turn_history["rewards"])
                log_episode = {
                    "environment": self.env.name,
                    "dataset": self.args.data_path,
                    "task_id": idx,
                    "query": self.env.query,
                    "turn_history": turn_history,
                    # "info": info,
                    "summary": {
                        "max_reward": max_reward,
                        "max_reward_idx": turn_history["rewards"].index(max_reward),
                        "turns_taken": turn + 1,
                        "turns_max": self.args.max_turns,
                    }
                }
                if "extra" in record and "hardness" in record["extra"]:
                    log_episode["hardness"] = record["extra"]["hardness"]
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn+1}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()


if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()
