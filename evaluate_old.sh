#set -x

for MULT in true; do
  for AL in 15; do
    EXP="15:250:lstm:max:300:0.2:100:75:0.003:100000.0:2:adam:30"
    CMD="python code/main.py \
      --mode=official_eval \
      --json_in_path=data/dev-v1.1.json \
      --ckpt_load_dir=experiments/$EXP/best_checkpoint/ \
      --experiment_name=$EXP \
      --multiply_probabilities=$MULT \
      --answer_len=$AL \
      --batch_size=1100 \
      --use_bidaf=true"
#      > /dev/null 2>&1"
    eval $CMD
    #echo "len: "$AL", mult: "$MULT
    python code/evaluate.py data/dev-v1.1.json predictions.json
  done
done
