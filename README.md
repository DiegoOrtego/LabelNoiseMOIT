# LabelNoiseMOIT
Official implementation for: "Multi-Objective Interpolation Training for Robustness to Label Noise"

Interpolated supervised contrastive learning and semi-supervised learning for robustness to label noise!
Important note: Our method is robust in all scenarios with a single parametrization. We only modify typical training parameters (epochs, learning rate scheduling, batch size or memory size).


<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/DiegoOrtego/LabelNoiseMOIT/blob/main/Overview.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/https://github.com/DiegoOrtego/LabelNoiseMOIT/blob/main/Overview.pdf">Download PDF</a>.</p>
    </embed>
</object>

Multi-Objective Interpolation Training (MOIT) for improved robustness to label noise. We interpolate samples and impose the same interpolation in the supervised contrastive learning loss (ICL) and the semi-supervised classification loss (SSL) that we jointly use during training. Label noise detection is performed at every epoch and its result is used after training MOIT to fine-tune the encoder and classifier to further boost performance with MOIT+.


State-of-the-art Results in CIFAR-10 and CIFAR-100:

![plot](https://github.com/DiegoOrtego/LabelNoiseMOIT/blob/main/CIFAR_results.png)


