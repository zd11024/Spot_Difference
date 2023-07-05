work_dir=/apdcephfs/share_47076/tmpv_dozheng/Spot_Difference/bottom-up-attention-vqa
data_root=/apdcephfs/share_47076/tmpv_dozheng/Spot_Difference/data/0123
image_file=/apdcephfs/share_47076/tmpv_dozheng/Spot_Difference/data/img_feat_0123.h5
model_path=/apdcephfs/share_47076/tmpv_dozheng/Spot_Difference/checkpoints/butd_vqa/model.pth

python -u $work_dir/train_vqa.py --output $work_dir/checkpoints/vqa \
	--dataroot $data_root \
	--img_feature_file $image_file \
	--mode val \
	--model_path $model_path \


