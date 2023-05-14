gpu=0,1,2,3

CUDA_VISIBLE_DEVICES=${gpu} python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=11240 --use_env main_classifier.py \
--model deit_small_lctr \
--batch-size 128 \
--data-path ImageNet/train/ILSVRC2012_img_train/data \
--output_dir log \
--resume log/deit_small_patch16_224.pth \
--epochs 150 \
--data-set IMNET \
--lr 0.00015
