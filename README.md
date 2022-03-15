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
  pip install torch==1.7.1+cu110
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

Considering that it can take a very long time to prepare the data, we provide the processed data for download (TBD)：

* [English-French]()

* [English-German]()
* [English-Romanian]()

## Pretrained models

We adopted the released [XLM](https://github.com/facebookresearch/XLM) and [MASS](https://github.com/microsoft/MASS) models for all language pairs. In order to better reproduce the results for MASS on En-De, we used monolingual data to continue pre-training the MASS pre-trained model for 300 epochs and selected the best model (epoch@270) by perplexity (PPL) on the validation setss. 

We provide the pretrained models we used:

| Languages        |                             XLM                              |                             MASS                             |
| :--------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| English-French   | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth) | [Model](https://modelrelease.blob.core.windows.net/mass/mass_enfr_1024.pth) |
| English-German   | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth) |                     [Model](https://tbd)                     |
| English-Romanian | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_enro_1024.pth) | [Model](https://modelrelease.blob.core.windows.net/mass/mass_enfr_1024.pth) |

## Model training

We provide training scripts for baseline UNMT models and our approach with online self-training. For example, train UNMT model with online self-training and XLM initialization:

```shell
cd scripts
sh run-xlm-unmt-st-ende.sh
```

**Note:** remember to modify the path variables in the header of the shell script.

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
