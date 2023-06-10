# Rm-LR
## Intrduction
This repository contained code for "Rm-LR: A long-range-based deep learning model for predicting multiple types of RNA modifications"
Any question is welcomed to be asked by issue and I will try my best to solve your problems.

## Paper Abstract
Recent research has highlighted the pivotal role of RNA post-transcriptional modifications in the regulation of RNA expression and function. Accurate identification of RNA modification sites is important for understanding RNA function. In this study, we propose a novel RNA modification prediction method, namely Rm-LR, which leverages a long-range-based deep learning approach to accurately predict multiple types of RNA modifications using RNA sequences only. Rm-LR incorporates two large-scale RNA language pre-trained models to capture discriminative sequential information and learn local important features, which are subsequently integrated through a bilinear attention network. Rm-LR supports a total of ten RNA modification types (, , , , , Ψ, Am, Cm, Gm, and Um) and significantly outperforms the state-of-the-art methods in terms of predictive capability on benchmark datasets. Experimental results show the effectiveness and superiority of Rm-LR in prediction of various RNA modifications, demonstrating the strong adaptability and robustness of our proposed model. We demonstrate that RNA language pretrained models enable to learn dense biological sequential representations from large-scale long-range RNA corpus, and meanwhile enhance the interpretability of the models. This work contributes to the development of accurate and reliable computational models for RNA modification prediction, providing insights into the complex landscape of RNA modifications.

![框架图_01](https://github.com/LiangThree/Rm-LR/assets/81310301/dd951543-19d4-46cb-8f13-46348a33a079)


## How to use it
In banch, you can find train.py, with the whole project, you can run it directly.
