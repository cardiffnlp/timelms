# TimeLMs

TimeLMs allows for easy access to models continuously trained on social media over regular intervals for researching language model degradation, as well as cultural shifts affecting language usage on social media.

To learn more and start using TimeLMs, please refer to our [notebook](demo.ipynb).

# Getting Started

You may create a new environment using conda and install dependencies following the commands below.
We assume you already have PyTorch with CUDA support installed (tested with torch==1.8.2+cu111 and CUDA 11.2).

```bash
$ conda create -n timelms python=3.7
$ conda activate timelms
$ pip install -r requirements.txt
```

# License

TimeLMs is released without any restrictions, but our scoring code is based on the [https://github.com/awslabs/mlm-scoring](https://github.com/awslabs/mlm-scoring) repository, which is distributed under [Apache License 2.0](https://github.com/awslabs/mlm-scoring/blob/master/LICENSE). We also refer users to Twitter regulations regarding use of our models and test sets.
