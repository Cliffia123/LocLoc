python3 main_eval.py \
--eval \
--model deit_small_lctr \
--data-set IMNET \
--data-path /GPUFS/nsccgz_ywang_zfd/ImageNet/train/ILSVRC2012_img_train/data \
--resume log/deit_small_patch16_224.pth \
--classifier log/IMNET_cls_best_50.pth \
--locator log/IMNET_loc_best_50.pth