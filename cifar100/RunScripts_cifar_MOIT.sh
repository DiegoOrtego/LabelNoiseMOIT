


python3 train_MOIT.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --M 125 --M 200 --noise_ratio 0.4 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --method "MOIT" --download True --noise_type "asymmetric" \
--batch_t 0.1 --headType "Linear" --mix_labels 1  --xbm_use 1 --xbm_begin 3 --xbm_per_class 100 \
--balance_crit "median" --discrepancy_corrected 1 --validation_exp 0 \
--startLabelCorrection 130 --PredictiveCorrection 1 --k_val 250 \
--experiment_name Test_MOIT_asymmetric --cuda_dev 0


python3 train_MOIT.py --epoch 250 --num_classes 100 --batch_size 128 --low_dim 128 --M 125 --M 200 --noise_ratio 0.4 \
--network "PR18" --lr 0.1 --wd 1e-4 --dataset "CIFAR-100" --method "MOIT" --download True --noise_type "symmetric" \
--batch_t 0.1 --headType "Linear" --mix_labels 1  --xbm_use 1 --xbm_begin 3 --xbm_per_class 100 \
--balance_crit "median" --discrepancy_corrected 1 --validation_exp 0 \
--startLabelCorrection 130 --PredictiveCorrection 1 --k_val 250 \
--experiment_name Test_MOIT_symmetric --cuda_dev 0
