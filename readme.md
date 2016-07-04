### Multi-Task Learning

-------

## Introduction

This is an example of how to construct a multi-task neural net in Tensorflow. Here we're looking at Natural Language Processing ("NLP"), specifically on whether by learning about Part of Speech (POS) and Shallow Parsing (Chunking) at the same time we can improve performance on both.

## Network Structure

Our network looks a little bit like this, with Task 1 being Part of Speech (POS) and Task 2 being Chunk:

<img src='https://jg8610.github.io/images/joint_op.png'>

As you can see, you can train either tasks separately (by calling the individual training ops), or you can train the tasks jointly (by calling the join training op).

We have also added in an explicit connection from POS to Chunk, which actually makes the network into something similar to a ladder network with an explicit hidden state representation.

## Quick Start (Mac and Linux)

* This is python3, so please install anaconda3 and tensorflow. This should be enough to get you started.
* Then, go into the data folder and get rid of the ``.tmp`` endings on the data.
* Then run ``$ sh run_all.sh`` - this will start the joint training. Once it's finished, the outputs will be stored in ``./data/outputs``
* You can then print out the evaluations by typing ``python generate_results.py --path "./data/outputs/predictions/"``

## How to do single training

If you want to train each task separately and compare the results you just need to change an argument in the ``run_all.sh`` script.

### POS Single
```bash
python3 run_model.py --model_type "POS" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs/"

```

### Chunk Single
```bash
python3 run_model.py --model_type "CHUNK" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs/"

```

### Joint
```bash
python3 run_model.py --model_type "JOINT" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs/"

```
