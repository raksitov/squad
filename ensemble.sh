#set -x

#DATA="tiny-dev.json"
DATA="dev-v1.1.json"

####"experiments/ensemble1"
####"experiments/ensemble2"
####"experiments/ensemble6"
####"experiments/ensemble9"
####"experiments/ensemble3"
####"experiments/ensemble4"

#E01="experiments/ensemble01"
#E03="experiments/ensemble03"

E02="experiments/ensemble02"
E04="experiments/ensemble04"
E05="experiments/ensemble05"
E06="experiments/ensemble06"
E07="experiments/ensemble07"
E08="experiments/ensemble08"
E09="experiments/ensemble09"
E10="experiments/ensemble10"

T0="experiments/temp0"
T1="experiments/temp1"
T2="experiments/temp2"
T3="experiments/temp3"

for AL in 23; do
  CMD="python code/main.py \
    --mode=official_eval \
    --json_in_path=data/$DATA \
    --ckpt_load_dir=$E02,$E04,$E05,$E06,$E07,$E08,$E09,$E10,$T0,$T1,$T2,$T3 \
    --batch_size=400 \
    --answer_len=$AL \
    --h_embedding_size=42 \
    --ensemble \
    > /dev/null 2>&1"
  eval $CMD
  echo "Len:"$AL
  python code/evaluate.py data/$DATA predictions.json
done
