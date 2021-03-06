set -x

OPT='adam'
CTX=450
EMB=42
NE=0
B=150
LR=0.003
HS=100
NL=1
MS=75
ML=2
GN=100000.
D=0.2
CELL='lstm'
AL=23
BIDAF=true
SHARE=true
STATES=true
COMB='concat'
EVERY=`expr 85000 / $B / 2`
KEEP=3
python main.py \
  --mode=train \
  --reuse_question_states=$STATES \
  --share_encoder=$SHARE \
  --use_bidaf=$BIDAF \
  --h_model_size=$MS \
  --h_model_layers=$ML \
  --h_optimizer=$OPT \
  --num_epochs=$NE \
  --h_learning_rate=$LR \
  --h_context_len=$CTX \
  --h_embedding_size=$EMB \
  --h_batch_size=$B \
  --h_hidden_size=$HS \
  --h_dropout=$D \
  --h_cell_type=$CELL \
  --h_num_layers=$NL \
  --h_max_gradient_norm=$GN \
  --h_answer_len=$AL \
  --h_combiner=$COMB \
  --eval_every=$EVERY \
  --save_every=$EVERY \
  --keep=$KEEP \
  --experiments_results=`readlink -f ../data/`"/experiments_results.json"
