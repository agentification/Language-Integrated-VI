# Bash Call

#python -m experiments.eval_react \
#    --data_path ./data/nl2bash/nl2bash_fs_4.json \
#    --env bash \
#    --image_name intercode-nl2bash \
#    --log_dir react_gpt35 \
#    --max_turns 10

# SQL Call

python -m experiments.eval_react \
     --data_path ./data/spider/dev_spider.json \
     --env sql \
     --image_name docker-env-sql \
     --log_dir react_gpt35_rem \
     --max_turns 20
