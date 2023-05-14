python3 main_eval.py \
--eval \
--model deit_small_lctr \
--data-set CUB \
--data-path data/CUB_200_2011/images \
--resume log/deit_small_patch16_224.pth \
--classifier log/CUB_cls_best.pth \
--locator log/CUB_loc_best_12.pth
