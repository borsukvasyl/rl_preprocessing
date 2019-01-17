# Reinforcement Learning preprocessing

## Installing `deeprl`

```bash
git clone https://github.com/borsukvasyl/deeprl
cd deeprl
pip install -e .
```

## Example of running DDDQN (Double Dueling Deep Q Network) on CartPole game

DDDQN trainer uses Epsilon Greedy policy (with some probability `e` takes random action, which is used for exploration),
so you can get different result in every run

### Adding project root to PYTHONPATH

In the project root directory execute this command:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### Running without data preprocessing

```bash
# train model
cd normalization/dddqn_cartpole
python DDDQN_CartPole.py --dir="temp1"

# displaying learning process
tensorboard --logdir="temp1/model_chkp"
```

### Running with data preprocessing

```bash
# train model
cd normalization/dddqn_cartpole
python DDDQN_CartPole.py --use_norm --dir="temp2"

# displaying learning process
tensorboard --logdir="temp2/model_chkp"
```