import os
os.environ['OPENAI_API_KEY'] = '0'
import yaml
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
import random
import numpy as np
from src.plan_forward import forward_plan
from src.models import QueryLlama, QueryVicuna

import torch
from llama import *
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import json
import time


def validate_plan(domain, instance, plan_file):
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()

    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True, response
    else:
        return False, response


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    # torch.manual_seed(1)
    return local_rank, world_size

def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator

success_template = "{} {} {} {}"
verbose_template="""
{}
--------- LLM response ---------
{}
--------- Extracted plan ---------
{}
-------- Ground truth plan ---------
{}
{}
"""

class ReasoningTasks():

    def __init__(self, model_name="LLaMA", ckpt_path="", data_path="",
                 model_path='lmsys/vicuna-7b-v1.3', num_gpus=1):
        # self.engine = engine
        self.max_gpt_response_length = 500
        self.data_files = json.load(open(data_path, 'r'))
        self.model_name = model_name

        self.plan_file = "sas_plan"
        self.lm_plan_file = "gpt_sas_plan"

        if local_rank > 0:
            sys.stdout = open(os.devnull, 'w')
            log_file = None
        else:
            log_file = "logs/interactive.log"

        self.local_rank = local_rank

        if self.model_name == "LLaMA":
            llm = ckpt_path
            # the parent directory of the checkpoint directory
            tokenizer_path = os.path.join(os.path.dirname(llm), "tokenizer.model")
            llama = load(llm, tokenizer_path, local_rank, world_size, 3)
            self.model = QueryLlama(llama, max_response_length=100, log_file=log_file)
        elif self.model_name == "Vicuna":
            self.model = QueryVicuna(model_path, num_gpus)
        else:
            raise NotImplementedError
        

    # ========================================== UTILS ========================================== #
    def compute_plan(self, domain, instance, timeout=30):
        fast_downward_path = os.getenv("FAST_DOWNWARD")
        # Remove > /dev/null to see the output of fast-downward
        assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        
        if local_rank == 0:
            if os.path.exists(self.plan_file):
                try:
                    os.remove(self.plan_file)
                except Exception as e:
                    print(e)

            while not os.path.exists(self.plan_file):
                cmd = f"timeout {timeout}s {fast_downward_path}/fast-downward.py --log-level debug {domain} {instance} --search \"astar(lmcut())\"  > /dev/null 2>&1"
                os.system(cmd)
                time.sleep(2)
                
        torch.distributed.barrier()
        
        if not os.path.exists(self.plan_file):
            return ""
        
        return Path(self.plan_file).read_text()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain):
        plan_executor = Executor(domain, instance)
        return plan_executor

    def save_output(self, output_file, final_output):
        os.makedirs(f"outputs/{self.model_name}/", exist_ok=True)
        with open(f"outputs/{self.model_name}/" + output_file + ".txt", 'w+') as f:
            f.write(final_output)
    # ========================================== TASKS ========================================== #

    def run_all(self,
                 plan_method,
                 config_file,
                 name="",
                 prompts="",
                 prompt_path="",
                 resume_file_idx=0,
                 single_run=10000,
                 alpha=0.5,
                 horizon=10,
                 search_depth=6,
                 sample_per_node=2,
                 sampler='heuristic',
                 discount=1,
                 use_lang_goal=False):
        self.read_config(config_file)

        # make directory for logs
        os.makedirs(f"logs/{name}/json/", exist_ok=True)
        os.makedirs(f"logs/{name}/tree/", exist_ok=True)
        os.makedirs(f"logs/{name}/pkl/", exist_ok=True)
        os.makedirs(f"logs/{name}/sample/", exist_ok=True)

        n_files = len(self.data_files)
        domain_pddl = f'gpt-plan-benchmark/gpt_plan_test/instances/{self.data["domain_file"]}'

        final_output = ""
        correct_plans = 0

        if local_rank == 0:
            if os.path.exists(self.plan_file):
                os.remove(self.plan_file)
            if os.path.exists(self.lm_plan_file):
                os.remove(self.lm_plan_file)
        

        with open(prompt_path) as f:
            prompts = json.load(f)

        task_pool = np.arange(n_files) if single_run == 10000 else [single_run]
        print(f'There are {len(task_pool)} files.')
        for i in task_pool:
            print(f'We are dealing with {i}-th file now.')
            if i < resume_file_idx:
                if self.local_rank == 0:
                    correct_plans += 1
                continue

            cur_instance = self.data_files[i]
            problem = self.get_problem(cur_instance[0], domain_pddl)
            INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, False, self.data)

            query = prompts["baseline_action"]
            query += fill_template(*instance_to_text_blocksworld(problem, False, self.data)) + "\n"
            
            result, her_plan, tot_sample = plan_method(
                f'I have that, {INIT}.', 
                f'My goal is to have that {GOAL}.',
                prompts, 
                self.model, 
                alpha=alpha,
                horizon=horizon,
                search_depth=search_depth,
                sample_per_node=sample_per_node,
                sampler=sampler,
                discount=discount,
                use_lang_goal=use_lang_goal,
            )

            torch.distributed.barrier()

            if self.local_rank == 0:
                with open(os.path.join(f'./logs/{name}/json/', f'{i:04d}.json'), 'w') as f:
                    json.dump(her_plan, f, indent=2)
                with open(os.path.join(f'./logs/{name}/sample/', f'{i:04d}.accn'), 'w') as f:
                    f.write(f'{result},{tot_sample}')

            torch.distributed.barrier()
            correct_plans += int(result)

        if local_rank == 0:
            if os.path.exists(self.plan_file):
                os.remove(self.plan_file)
            if os.path.exists(self.lm_plan_file):
                os.remove(self.lm_plan_file)

        final_output += f"[+]: The number of correct plans is {correct_plans}/{len(task_pool)}={correct_plans / (len(task_pool)) * 100}%"
        print(f"[+]: The number of correct plans is {correct_plans}/{len(task_pool)}={correct_plans / (len(task_pool)) * 100}%")
        self.save_output(name, final_output)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    local_rank, world_size = setup_model_parallel()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['LLaMA', 'Vicuna'])
    parser.add_argument('--name', type=str, default="unnamed", help='Name of the experiment')
    parser.add_argument('--data_path', type=str, default="data", help='Path to data')
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--search_depth', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for reward')
    parser.add_argument('--prompt_path', type=str, default="data/blocksworld/my_mcts_prompts_update.json", help='Path to prompts')
    parser.add_argument('--ckpt_path', type=str, default="", help='path to LLaMA checkpoint')
    parser.add_argument('--resume_file_idx', type=int, default=0, help='resume experiment from a certain task')
    parser.add_argument('--single_run', type=int, default=10000)
    parser.add_argument('--sample_per_node', type=int, default=2, help='number of samples we take in the lookahead trajectory search')
    parser.add_argument('--sampler', type=str, default='heuristic')
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--model_path', type=str, required=True, choices=['lmsys/vicuna-7b-v1.3', 'lmsys/vicuna-13b-v1.3', 'lmsys/vicuna-33b-v1.3'])
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--use_lang_goal', action='store_true')


    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    alpha = args.alpha
    name = args.name
    prompt_path = args.prompt_path
    ckpt_path = args.ckpt_path

    tasks_obj = ReasoningTasks(model_name=model_name, data_path=data_path, ckpt_path=ckpt_path, model_path=args.model_path, num_gpus=args.num_gpus)
    config_file = 'data/blocksworld/bw_config.yaml'

    plan_method=forward_plan

    tasks_obj.run_all(
        plan_method=plan_method,
        config_file=config_file,
        name=name,
        prompts="",
        prompt_path=prompt_path,
        resume_file_idx=args.resume_file_idx,
        single_run=args.single_run,
        alpha=alpha,
        horizon=args.horizon,
        search_depth=args.search_depth,
        sample_per_node=args.sample_per_node,
        sampler=args.sampler,
        discount=args.discount,
        use_lang_goal=args.use_lang_goal
    )