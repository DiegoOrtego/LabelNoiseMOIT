

python3 train_MOITplus.py --epoch 70 --num_classes 100 --batch_size 128 --noise_ratio 0.4 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --method "MOIT" \
--headType "Linear" --noise_type "asymmetric" --DA "Simple" --validation_exp 0 --ReInitializeClassif 1  \
--startLabelCorrection 30 --PredictiveCorrection 1 \
--experiment_name Test_MOITplus_asymmetric --cuda_dev 0

python3 train_MOITplus.py --epoch 140 --num_classes 100 --batch_size 128 --M 120 --noise_ratio 0.8 \
--network "PR18" --lr 0.001 --wd 1e-4 --dataset "CIFAR-100" --method "MOIT" \
--headType "Linear" --noise_type "symmetric" --DA "Simple" --validation_exp 0 --ReInitializeClassif 1  \
--startLabelCorrection 30 --PredictiveCorrection 1 \
--experiment_name Test_MOITplus_symmetric --cuda_dev 0


