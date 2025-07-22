echo $1, $2
seed=$2
export CUDA_VISIBLE_DEVICES=0,1
output_dir=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/multi_finetune_model
base_model=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/llama-7b
train_data=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/TALLRec/data/book/train.json
train_data2=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/TALLRec/data/movie/train.json
val_data=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/TALLRec/data/book/valid.json
val_data2=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/TALLRec/data/movie/valid.json
instruction_model=/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/TALLRec/alpaca-lora-7B
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 16
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                python -u finetune_multi_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --train_data_path2 $train_data2 \
                    --val_data_path $val_data \
                    --val_data_path2 $val_data2 \
                    --output_dir ${output_dir}_${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 64 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16 \
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2
        done
    done
done

