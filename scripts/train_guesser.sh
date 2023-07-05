work_path=.
python -u $work_path/train.py \
	--mode guesser \
	--train_data_file $work_path/data/1013/spot_diff_train.json \
	--eval_data_file $work_path/data/1013/spot_diff_val.json \
	--img_feat_file $work_path/data/img_feat_3ee94.h5 \
	--output_dir $work_path/checkpoints/guesser \
	--model_name_or_path $work_path/checkpoints/pretrained/bert-base-uncased \
	--block_size 512 \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--evaluate_during_training \
	--num_train_epochs 30 \
	--save_total_limit 20 \
	--num_workers 8 \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 8 \
	--overwrite_output_dir \
	--save_steps 1000
