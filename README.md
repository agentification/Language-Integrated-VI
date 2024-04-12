# Language-Integrated Value Iteration 

Code for [How Can LLM Guide RL? A Value-Based Approach](https://arxiv.org/abs/2402.16181).


Authors: [Shenao Zhang](https://shenao-zhang.github.io)&ast;, [Sirui Zheng](https://openreview.net/profile?id=~Sirui_Zheng2)&ast;, [Shuqi Ke](https://openreview.net/profile?id=~Shuqi_Ke1), [Zhihan Liu](https://scholar.google.com/citations?user=uEl_TtkAAAAJ&hl=en), [Wanxin Jin](https://wanxinjin.github.io), [Jianbo Yuan](https://scholar.google.com/citations?user=B1EhbCsAAAAJ&hl=en), [Yingxiang Yang](https://scholar.google.com/citations?user=0SKlCbgAAAAJ&hl=en), [Hongxia Yang](https://scholar.google.com/citations?user=iJlC5mMAAAAJ&hl=en), [Zhaoran Wang](https://zhaoranwang.github.io) (&ast; indicates equal contribution)

---

## ALFWorld
### Environment setup
- Clone the repository:
```bash
git clone https://github.com/agentification/Language-Integrated-VI.git
cd Language-Integrated-VI/alfworld
```

- Create a virtual environment and install the required packages:
```bash
pip install -r requirements.txt
```

-  Install the ALFWorld environment. Please refer to https://github.com/alfworld/alfworld.


- Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

### Run the code
```bash
./run.sh
```


---
## InterCode
Steps to run our algorithm in the [InterCode](https://arxiv.org/abs/2306.14898) environment.

### Environment setup

- Clone the repository, create a virtual environment, and install necessary dependencies:
```bash
git clone https://github.com/agentification/Language-Integrated-VI.git
cd Language-Integrated-VI/intercode
conda env create -f environment.yml
conda activate intercode
```

- Run `setup.sh` to create the docker images for the InterCode Bash, SQL, and CTF environments.

- Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```
### Run the code

- For InterCode-SQL, run
```bash
./scripts/expr_slinvit_sql.sh
```
- For InterCode-Bash, run
```bash
./scripts/expr_slinvit_bash.sh
```

---
## BlocksWorld

### Environment setup

- Our experiments are conducted with Vicuna-13B/33B (v1.3). The required packages can be installed by
    ```
    pip install -r requirements.txt
    ```

### Run the code

- To run the RAP experiments, here is a shell script of the script
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --master_port 1034 --nproc_per_node 1 run_mcts.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --name m6ct_roll60 --rollouts 60 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 3
    ```

- To run the SLINVIT experiments, here is a shell script example
    ```bash
    CUDA_VISIBLE_DEVICES=3,4,5 nohup python -m torch.distributed.run --master_port 39855 --nproc_per_node 1 run.py \
    --model_name Vicuna \
    --name planning_step6_13b \
    --data data/blocksworld/step_6.json \
    --horizon 6 \
    --search_depth 5 \
    --alpha 0 \
    --sample_per_node 2 \
    --model_path lmsys/vicuna-13b-v1.3 \
    --num_gpus 3 \
    --use_lang_goal
    ```

## Citation

```bibtex
@article{zhang2024can,
  title={How Can LLM Guide RL? A Value-Based Approach},
  author={Zhang, Shenao and Zheng, Sirui and Ke, Shuqi and Liu, Zhihan and Jin, Wanxin and Yuan, Jianbo and Yang, Yingxiang and Yang, Hongxia and Wang, Zhaoran},
  journal={arXiv preprint arXiv:2402.16181},
  year={2024}
}
```
