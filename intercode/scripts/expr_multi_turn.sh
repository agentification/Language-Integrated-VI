# !bin/bash

# Data Paths
# - (SQL)  ./data/spider/dev_spider.json
# - (SQL)  ./data/test/sql_queries.csv
# - (Bash) ./data/nl2bash/nl2bash.json
# - (Bash) ./data/test/bash_queries.json
# - (Bash) ./data/nl2cmd.json

# Environments
# - sql, bash

# Image Names
# - (SQL)  docker-env-sql
# - (Bash) intercode-bash
# - (Bash) intercode-nl2bash
# - (Py)   intercode-python

# Policies
# - chat

# Bash Call
# python -m experiments.eval_n_turn \
#     --data_path ./data/nl2bash/nl2bash_fs_4.json \
#     --dialogue_limit 7 \
#     --env bash \
#     --image_name intercode-nl2bash \
#     --log_dir base_try_15 \
#     --max_turns 15 \
#     --policy chat \
#     --template v2 \
#     --model gpt-3.5-turbo \
    # --verbose

# SQL Call
python -m experiments.eval_n_turn \
    --data_path ./data/spider/dev_spider.json \
    --dialogue_limit 5 \
    --env sql \
    --image_name docker-env-sql \
    --log_dir base_try_15 \
    --max_turns 15 \
    --policy chat \
    --template game_sql \
    --model gpt-3.5-turbo
    # --handicap 
    # --verbose

python -m experiments.eval_n_turn \
    --data_path ./data/spider/dev_spider.json \
    --dialogue_limit 5 \
    --env sql \
    --image_name docker-env-sql \
    --log_dir base_try_20 \
    --max_turns 20 \
    --policy chat \
    --template game_sql \
    --model gpt-3.5-turbo

python -m experiments.eval_n_turn \
    --data_path ./data/spider/dev_spider.json \
    --dialogue_limit 5 \
    --env sql \
    --image_name docker-env-sql \
    --log_dir base_try_25 \
    --max_turns 25 \
    --policy chat \
    --template game_sql \
    --model gpt-3.5-turbo

python -m experiments.eval_n_turn \
    --data_path ./data/spider/dev_spider.json \
    --dialogue_limit 5 \
    --env sql \
    --image_name docker-env-sql \
    --log_dir base_try_30 \
    --max_turns 30 \
    --policy chat \
    --template game_sql \
    --model gpt-3.5-turbo