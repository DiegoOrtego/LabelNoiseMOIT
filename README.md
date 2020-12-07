# LabelNoiseMOIT
Official implementation for: "Multi-Objective Interpolation Training for Robustness to Label Noise"

Interpolated supervised contrastive learning and semi-supervised learning for robustness to label noise!


![plot](https://github.com/DiegoOrtego/LabelNoiseMOIT/blob/main/Overview.png)

Multi-Objective Interpolation Training (MOIT) for improved robustness to label noise. We interpolate samples and impose the same interpolation in the supervised contrastive learning loss $\mathcal{L}^{\mathit{ICL}}$ and the semi-supervised classification loss $\mathcal{L}^{\mathit{SSL}}$ that we jointly use during training. Label noise detection is performed at every epoch and its result is used after training to fine-tune the encoder and classifier to further boost performance.

