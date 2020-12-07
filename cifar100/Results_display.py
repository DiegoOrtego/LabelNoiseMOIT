import os
import numpy as np
import matplotlib.pyplot as plt


res_path = 'metricsPR18_Test_MOIT_asymmetric_SI1_SD42'
#res_path = 'metricsPR18_Test_MOIT_symmetric_SI1_SD42'


k_vec = ["MOIT"]



noise_levels = np.array([0.4])


numNoise = noise_levels.size


loss_clean_vec = np.zeros((1,numNoise))
loss_noisy_vec = np.zeros((1,numNoise))

nRows = 1
nCols = 1

fig1 = plt.figure(1)


for i in range(numNoise):

    res_noise_path = os.path.join(res_path,str(noise_levels[i]))
    ##Accuracy
    acc_valClassif = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_accuracy_per_epoch_val_pred.npy')

    #Loss per epoch
    loss_train = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_LOSS_epoch_train.npy')
    loss_val = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_LOSS_epoch_val.npy')

    numEpochs = len(acc_valClassif)
    epochs = range(numEpochs)

    if i==0:
        loss_train_vec = np.zeros((numNoise, numEpochs))
        loss_val_vec = np.zeros((numNoise, numEpochs))

        acc_train_vec = np.zeros((numNoise, numEpochs))
        acc_val_vec = np.zeros((numNoise, numEpochs))
        acc_valL4_vec = np.zeros((numNoise, numEpochs))

        loss_samples_train_clean = np.zeros((numEpochs, numNoise))
        loss_samples_train_noisy = np.zeros((numEpochs, numNoise))

    loss_train_vec[numNoise-i-1, :] = loss_train
    loss_val_vec[numNoise-i-1, :] = loss_val

    #Load clean and noisy samples

    loss_tr = np.load(res_noise_path + '/discrepancy2_per_sample_train.npy')

    loss_tr_t = np.transpose(loss_tr)
    noisy_labels = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_diff_labels.npy')
    labels = np.array(range(loss_tr.shape[1]))
    clean_labels = np.setdiff1d(labels, noisy_labels)


    noisy_labels = noisy_labels.astype(int)

    agreement = np.load(res_noise_path + '/agreement_per_sample_train.npy')
    clean_idx = np.zeros((agreement.shape[1],))
    clean_idx[clean_labels] = 1.0
    extended_clean_idx = np.expand_dims(clean_idx, 1).repeat(agreement.shape[0], 1).transpose()
    noiseDetectionAcc = np.sum((agreement-extended_clean_idx)==0, 1) / agreement.shape[1]
    noiseDetectionRecall = np.sum((agreement + extended_clean_idx) == 2, 1) / np.sum(extended_clean_idx[0, :])
    noiseDetectionPrecision = np.sum((agreement + extended_clean_idx) == 2, 1) / (np.sum((agreement + extended_clean_idx) == 2, 1) + np.sum((agreement - extended_clean_idx) == 1, 1))
    num_clean_detected = np.sum(agreement, 1)


    avg_clean = loss_tr_t[clean_labels].mean(axis=0)
    avg_noisy = loss_tr_t[noisy_labels].mean(axis=0)
    std_clean = loss_tr_t[clean_labels].std(axis=0)
    std_noisy = loss_tr_t[noisy_labels].std(axis=0)

    quart25_clean = np.quantile(loss_tr_t[clean_labels], 0.25, axis=0)
    quart75_clean = np.quantile(loss_tr_t[clean_labels], 0.75, axis=0)
    median_clean = np.quantile(loss_tr_t[clean_labels], 0.5, axis=0)

    if noise_levels[i]!=0:
        quart25_noisy = np.quantile(loss_tr_t[noisy_labels], 0.25, axis=0)
        quart75_noisy = np.quantile(loss_tr_t[noisy_labels], 0.75, axis=0)
        median_noisy = np.quantile(loss_tr_t[noisy_labels], 0.5, axis=0)


    x = np.linspace(0, len(avg_clean),len(avg_clean))

    ax = fig1.add_subplot(str(nRows)+str(nCols)+str(i+1))
    ax.set_title('Noise level: ' + str(noise_levels[i]), y=1.08)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(x, median_clean, 'b-', label='Clean')
    ax.fill_between(x, quart25_clean, quart75_clean, alpha=0.2, color='b')


    if noise_levels[i] != 0:
        ax.plot(x, median_noisy, 'r-', label='Noisy')
        ax.fill_between(x, quart25_noisy, quart75_noisy, alpha=0.2, color='r')
        ax.legend(loc='upper right')

    lossEpochX = loss_tr_t[:, -1] #Epoch 40
    lossEpochX_clean = lossEpochX[clean_labels]
    lossEpochX_noisy = lossEpochX[noisy_labels]


    plt.figure(51)
    plt.plot(epochs, acc_valClassif, label=k_vec[0] + ", " + str(noise_levels[i]) + ', Classif, best (last): ' + str(np.max(acc_valClassif)) + "(" + str(acc_valClassif[-1])  + ")")
    plt.ylabel('Acc test')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    fig60, ax60 = plt.subplots()
    color = "tab:red"
    ax60.set_xlabel("epochs")
    ax60.set_ylabel("Noise detection precision")
    ax60.plot(epochs, noiseDetectionPrecision, label='Max. precision ' + str(np.max(noiseDetectionPrecision)), color=color)

    color = "tab:green"
    ax60.plot(epochs, noiseDetectionRecall, label='Max. recall ' + str(np.max(noiseDetectionRecall)), color=color)
    ax60.tick_params(axis="y", labelcolor=color)

    ax61 = ax60.twinx()
    color = "tab:blue"
    ax61.set_ylabel("Number of samples detected")
    ax61.plot(epochs, num_clean_detected, label='Max. number of samples ' + str(np.max(num_clean_detected)), color=color)
    ax61.tick_params(axis="y", labelcolor=color)

    fig60.legend(loc='lower right')

plt.show()