# Efficient Dialog Policy Learning via Positive Memory Retention

Here is the code repository for the paper "Efficient Dialog Policy Learning via Positive Memory Retention".

The paper is published in 2018 IEEE Spoken Language Technology (SLT), link: https://ieeexplore.ieee.org/document/8639617.

The preprint version of the paper is avaliable at: https://arxiv.org/abs/1810.01371

This work was also presented at NIPS 2018 Visually Grounded Interaction and Language (ViGIL) Workshop.

## Installation:

Our code is implemented in [Torch][1] (Lua), tested with GPU, on Ubuntu 16.04.   

The code uses the following packages: [torch/torch7][2], [torch/nn][3], [torch/nngraph][4], [Element-Research/dp][7], [Element-Research/rnn][5], [lua-cjson][6]. These can be installed/updated using:

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

luarocks install dp
luarocks install nngraph
luarocks install lua-cjson

git clone https://github.com/Element-Research/rnn.git
cd rnn
luarocks make rocks/rnn-scm-1.rockspec
```

## Reproducibility:

You can download the pretrain models to reproduce the experiment results on [MNIST GuessNumber dataset][8] quickly.

The training, validation, and testing splits have been already stored in the data folder.

```
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/guesser_float.t7  
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/oracle_float.t7  
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/qgen_float.t7 
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/qgen_reinforce_10_epochs.t7
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/qgen_reinforce_100_epochs.t7
wget https://s3-eu-west-1.amazonaws.com/myownbucket-123456789/pretrained_models/qgen_retention_10_epochs.t7 
```

For testing trained models, simply run:

```
th reinforce.lua --qgen_model 'qgen_float.t7' --epochs 0
th reinforce.lua --qgen_model 'qgen_reinforce_10_epochs.t7' --epochs 0
th reinforce.lua --qgen_model 'qgen_reinforce_100_epochs.t7' --epochs 0
th reinforce.lua --qgen_model 'qgen_retention_10_epochs.t7' --epochs 0
```

The model name 'qgen_float.t7' is the pretrained quentioner model using supervised learning.  

You can also change the name to 'qgen_reinforce_10_epochs.t7', 'qgen_reinforce_100_epochs.t7', or 'qgen_retention_10_epochs.t7', to test the models trained using REINFORCE for 10 epochs, using REINFORCE for 100 epochs, and using Positive Memory Retention for 10 epochs, respectively.  

## Training:

You can also train the model using REINFORCE or Positive Memory Retention by yourself. For example:

```
th reinforce.lua --qgen_model 'qgen_float.t7' --epochs 10
th reinforce.lua --qgen_model 'qgen_float.t7' --epochs 100
th reinforce.lua --qgen_model 'qgen_float.t7' --retention --epochs 10
```

## Results:

After testing, we obtained the following results:

| Method        | Training Epochs     | Test Accuracy   |
| --------      |:-----:              |:-----:          |
| REINFORCE     | 100                 | 69.86%          |
| REINFORCE+PMR | 10 (only)           | 70.27%          |

## Conclusion:

Positive Memory Retention (PMR) improves the sample-efficiency of REINFORCE by a factor of 10 on [this syntactic dataset][8].

## Citation:

Citation of the paper:

```
@inproceedings{zhao2018efficient,
  title={Efficient Dialog Policy Learning via Positive Memory Retention},
  author={Zhao, Rui and Tresp, Volker},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)},
  pages={823--830},
  year={2018},
  organization={IEEE}
}
```

## Licence:

MIT


[1]: http://torch.ch/
[2]: https://github.com/torch/torch7
[3]: https://github.com/torch/nn
[4]: https://github.com/torch/nngraph
[5]: https://github.com/Element-Research/rnn/
[6]: https://luarocks.org/modules/luarocks/lua-cjson
[7]: https://github.com/nicholas-leonard/dp
[8]: https://github.com/ruizhaogit/MNIST-GuessNumber
