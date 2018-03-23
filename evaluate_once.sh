#set -x

EXP="15:170:lstm:max:300:0.2:42:100:0.003:10002.0:2:75:1:adam:30"
CMD="python code/main.py \
  --mode=official_eval \
  --json_in_path=data/dev-v1.1.json \
  --multiply_probabilities=true \
  --ckpt_load_dir=experiments/$EXP/best_checkpoint/ \
  --experiment_name=$EXP \
  --use_bidaf=false \
  --share_encoder=true \
  --batch_size=550 \
  --reuse_question_states=true \
  --answer_len=15"
#> /dev/null 2>&1
eval $CMD
python code/evaluate.py data/dev-v1.1.json predictions.json
