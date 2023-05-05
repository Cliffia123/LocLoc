gpu=0,1

CUDA_VISIBLE_DEVICES=${gpu} python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=11241 --use_env main_classifier.py \
--model deit_small_lctr \
--batch-size 128 \
--data-path /GPUFS/nsccgz_ywang_zfd/caoxz/data/CUB_200_2011/images \
--output_dir log \
--resume log/deit_small_patch16_224.pth \
--epochs 300 \
--data-set CUB \
--lr 0.00015