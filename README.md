# MLS Workflow Tips

**First, ssh into MLP head node (from Dice):**
	ssh mlp1

You now have access to a filespace on the MLP node that is unique to your UID.

Set up the environment:
	conda create --name env
Activate the environment:
	conda activate env
Install CUDA:
	conda install cuda
Install CuPy:
	conda install cupy

To run an interactive job:
Connect to node:
	srun --gres=gpu:1 --pty bash

Wait until you see a terminal message like “... has been allocated resources”. You are now connected to a node with access to a GPU. This node lets you run interactive jobs, which are short jobs that let you test GPU code (e.g. for debugging).

To run a job

First, write a python script that does whatever you want to do.
If you want to save an output, ideally do it in the script using cupy.savetxt(), etc.

Then, write a shell script that looks this:
#!/bin/bash
~Path to python, in conda envs, to be improved upon~ ~path to python script~

Then, run:
	sbatch --gres=gpu:1 script.sh


and your output will be saved to the directory you called sbatch from.
