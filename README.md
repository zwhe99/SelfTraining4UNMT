## Bridging the Data Gap between Training and Inference for Unsupervised Neural Machine Translation

This is the implementaion of our paper:

```
Bridging the Data Gap between Training and Inference for Unsupervised Neural Machine Translation
Zhiwei He*, Xing Wang, Rui Wang, Shuming Shi, Zhaopeng Tu
ACL2022 (long paper, main conference)
```

We based this code heavily on the original code of [XLM](https://github.com/facebookresearch/XLM) and [MASS](https://github.com/microsoft/MASS).

## Dependencies

* Python3

* Pytorch1.7.1

  ```shell
  pip3 install torch==1.7.1+cu110
  ```

* fastBPE

* Apex

  ```shell
  git clone https://github.com/NVIDIA/apex
  cd apex
  git reset --hard 0c2c6ee
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
  ```

## Data ready

We prepared the data following the instruction from [XLM (Section III)](https://github.com/facebookresearch/XLM/blob/main/README.md#iii-applications-supervised--unsupervised-mt). We used their released scripts, BPE codes and vocabularies. However, there are some differences with them：

* All available data is used, not just 5,000,000 sentences per language

* For Romanian, we augment it with the monolingual data from WMT16.

* Noisy sentences are removed：

  ```shell
  python3 filter_noisy_data.py --input all.en --lang en --output clean.en
  ```

* For English-German, we used the processed data provided by [KaiTao Song](https://github.com/StillKeepTry).

Considering that it can take a very long time to prepare the data, we provide the processed data for download：

* [English-French](https://drive.google.com/file/d/15OBlFMjuwkbaY47xWdPysMyfpB-CqVoC/view?usp=sharing)

* [English-German](https://drive.google.com/file/d/1W-ngJpUvfRwSmWAUR2GZejMHBlRCMjfS/view?usp=sharing)
* [English-Romanian](https://drive.google.com/file/d/1fTP7PIbebewoLZD1rShFManED9cMysrV/view?usp=sharing)

## Pre-trained models

We adopted the released [XLM](https://github.com/facebookresearch/XLM) and [MASS](https://github.com/microsoft/MASS) models for all language pairs. In order to better reproduce the results for MASS on En-De, we used monolingual data to continue pre-training the MASS pre-trained model for 300 epochs and selected the best model (epoch@270) by perplexity (PPL) on the validation set. 

Here are pre-trained models we used:

| Languages        |                             XLM                              |                             MASS                             |
| :--------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| English-French   | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth) | [Model](https://modelrelease.blob.core.windows.net/mass/mass_enfr_1024.pth) |
| English-German   | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth) | [Model](https://drive.google.com/file/d/13feylC1qFvG8kcNi-9JXVnzEYo0OouRK/view?usp=sharing) |
| English-Romanian | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enro_1024.pth) | [Model](https://modelrelease.blob.core.windows.net/mass/mass_enfr_1024.pth) |

## Model training

We provide training scripts and trained models for UNMT baseline and our approach with online self-training.

**Training scripts**

Train UNMT model with online self-training and XLM initialization:

```shell
cd scripts
sh run-xlm-unmt-st-ende.sh
```

***Note*:** remember to modify the path variables in the header of the shell script.

**Trained model**

We selected the best model by BLEU score on the validation set for both directions. Therefore, we release En-X and X-En models for each experiment.

<table>
<thead>
  <tr>
    <th>Approch</th>
    <th colspan="2">XLM</th>
    <th colspan="2">MASS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">UNMT</td>
    <td><a href="https://drive.google.com/file/d/1nLmt9zpywKB6jufUCJJPjZlNdJEe1Fmb/view?usp=sharing">En-Fr</a></td><td> 
    <a href="https://drive.google.com/file/d/1IjLb_KEPYYtRUgJtp23qYjIfpVh0kM5X/view?usp=sharing">Fr-En</a></td>
    <td><a href="https://drive.google.com/file/d/1ptyrsi_d3NvznHNX2yR5pDBicyiu0rSI/view?usp=sharing">En-Fr</a></td><td> 
    <a href="https://drive.google.com/file/d/11QkkP736ZJePgCNp0F-2fckQ9t-acSXC/view?usp=sharing">Fr-En</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1TJc4nVNvCsDw-Intr3hVOSjk9_yYY8hk/view?usp=sharing">En-De</a></td><td> 
    <a href="https://drive.google.com/file/d/1kZu9kILPtMw9ULvvGRbWDM7tn0ocWvTj/view?usp=sharing">De-En</a></td>
    <td><a href="https://drive.google.com/file/d/1u-aUk9t2muO25Sot-XBP6uyY3SXY88b4/view?usp=sharing">En-De</a></td><td> 
    <a href="https://drive.google.com/file/d/1lU742bZD1jeMCOhPyXIOQ6lhQ1OgZ9JX/view?usp=sharing">De-En</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1D7z0V-8BNdKMQb1Ci_Pe4ru4lvn562t6/view?usp=sharing">En-Ro</a></td><td> 
    <a href="https://drive.google.com/file/d/10n2vOb543rNvIf1d9woDcFK7UxhTMCrV/view?usp=sharing">Ro-En</a></td>
    <td><a href="https://drive.google.com/file/d/11-Twma-XGrZjzlJCbUncQ6rbxtloD-Fx/view?usp=sharing">En-Ro</a></td><td> 
    <a href="https://drive.google.com/file/d/1H4VY4cvOnrvftmQb9Nn2dTIVz9VmTKEM/view?usp=sharing">Ro-En</a></td>
  </tr>
  <tr>
    <td rowspan="3">UNMT-ST</td>
    <td><a href="https://drive.google.com/file/d/1zH3c9Erf9YU3tSLTbzQTAIz_Cf944BsX/view?usp=sharing">En-Fr</a></td><td> 
    <a href="https://drive.google.com/file/d/1WMYUox0jZWGjshSDLdKzlBy5P0LMVdhi/view?usp=sharing">Fr-En</a></td>
    <td><a href="https://drive.google.com/file/d/190iFbUFJ9vgQPcwUqgNJdUthtLfyX9sM/view?usp=sharing">En-Fr</a></td><td> 
    <a href="https://drive.google.com/file/d/1CmCD4BxogK62C9aforhLPe5xVJ4wSXcv/view?usp=sharing">Fr-En</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1Iw5vPNav07k5th79C8JyrVKCvxzvD3x2/view?usp=sharing">En-De</a></td><td> 
    <a href="https://drive.google.com/file/d/1h9cjSm_2_fIxaiubcYzcFh7-Tqo0f1AV/view?usp=sharing">De-En</a></td>
    <td><a href="https://drive.google.com/file/d/1wBxucr4vQYO0rnE1X3tiX5UkBtmUjhLd/view?usp=sharing">En-De</a></td><td> 
    <a href="https://drive.google.com/file/d/11HA-pHGoHQ8MVI0h8nNbUuHDo7q11pkq/view?usp=sharing">De-En</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1IQwj5dVY50s1plBZsPYkrb6rYk3jXi7x/view?usp=sharing">En-Ro</a></td><td> 
    <a href="https://drive.google.com/file/d/1noiff7a3hstCE10b3Aoxytc6HeChX0Lf/view?usp=sharing">Ro-En</a></td>
    <td><a href="https://drive.google.com/file/d/1vJolxhkAWh1fo3B_emoL3NImblnJpbf7/view?usp=sharing">En-Ro</a></td><td> 
    <a href="https://drive.google.com/file/d/1zNbOo-3Li3j0f6Hq0NN0-9ThcA8e8l4j/view?usp=sharing">Ro-En</a></td>
  </tr>
</tbody>
</table>
## Evaluation

#### Generate translations

Input sentences must have the same tokenization and BPE codes than the ones used in the model.

```shell
cat input.en.bpe | \
python3 translate.py \
  --exp_name translate  \
  --src_lang en --tgt_lang de \
  --model_path trained_model.pth  \
  --output_path output.de.bpe \
  --batch_size 8
```

#### Remove bpe

```shell
sed  -r 's/(@@ )|(@@ ?$)//g' output.de.bpe > output.de.tok
```

#### Evaluate

```shell
BLEU_SCRIPT_PATH=src/evaluation/multi-bleu.perl
BLEU_SCRIPT_PATH ref.de.tok < hyp.de.tok
```
