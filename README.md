# QANet
A Tensorflow implementation of Google's [QANet](https://openreview.net/pdf?id=B14TlG-RW) (previously Fast Reading Comprehension (FRC)) from [ICLR2018](https://openreview.net/forum?id=B14TlG-RW). (Note: This is not an official implementation from the authors of the paper)

## Dataset
The dataset used for this task is [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/).
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

## Requirements
  * Python>=2.7
  * NumPy
  * tqdm
  * TensorFlow>=1.5
  * spacy==2.0.9
  * bottle (only for demo)

## Usage
To download and preprocess the data, run

```bash
# download SQuAD and Glove
sh download.sh
# preprocess the data
python config.py --mode prepro
```

Just like [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net), hyper parameters are stored in config.py. To debug/train/test/demo, run

```bash
python config.py --mode debug/train/test/demo
```

To evaluate the model with the official code, run
```bash
python evaluate-v2.0.py ~/data/squad/dev-v2.0.json train/{model_name}/answer/answer.json
```

The default directory for the tensorboard log file is `train/{model_name}/event`

Set volume mount paths and port mappings (for demo mode)

```
export QANETPATH={/path/to/cloned/QANet}
export CONTAINERWORKDIR=/home/QANet
export HOSTPORT=8080
export CONTAINERPORT=8080
```

bash into the container
```
nvidia-docker run -v $QANETPATH:$CONTAINERWORKDIR -p $HOSTPORT:$CONTAINERPORT -it --rm tensorflow/qanet bash
```

Once inside the container, follow the commands provided above starting with downloading the SQuAD and Glove datasets.

### Pretrained Model
Pretrained model weights are temporarily not available.

## Detailed Implementaion

  * The model adopts character level convolution - max pooling - highway network for input representations similar to [this paper by Yoon Kim](https://arxiv.org/pdf/1508.06615.pdf).
  * The encoder consists of positional encoding - depthwise separable convolution - self attention - feed forward structure with layer norm in between.
  * Despite the original paper using 200, we observe that using a smaller character dimension leads to better generalization.
  * For regularization, a dropout of 0.1 is used every 2 sub-layers and 2 blocks.
  * Stochastic depth dropout is used to drop the residual connection with respect to increasing depth of the network as this model heavily relies on residual connections.
  * Query-to-Context attention is used along with Context-to-Query attention, which seems to improve the performance more than what the paper reported. This may be due to the lack of diversity in self attention due to 1 head (as opposed to 8 heads) which may have repetitive information that the query-to-context attention contains.
  * Learning rate increases from 0.0 to 0.001 in the first 1000 steps in inverse exponential scale and fixed to 0.001 from 1000 steps.
  * At inference, this model uses shadow variables maintained by the exponential moving average of all global variables.
  * This model uses a training / testing / preprocessing pipeline from [R-Net](https://github.com/HKUST-KnowComp/R-Net) for improved efficiency.

## Results
Not yet for SQUAD2.....

## TODO's
- [ ] Training and testing the model
- [ ] Realtime Demo
- [ ] Data augmentation by paraphrasing
- [ ] Train with full hyperparameters (Augmented data, 8 heads, hidden units = 128)

## Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=./
```
