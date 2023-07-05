work_dir=.
dataroot=$work_dir/data/0206
image_feat_file=$work_dir/data/img_feat_0123.h5

python -u $work_dir/bottom-up-attention-vqa/train_vqa.py \
	--output $work_dir/checkpoints/butd_vqa \
	--dataroot $dataroot \
	--img_feature_file $image_feat_file

