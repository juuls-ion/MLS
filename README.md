# MLS Workflow Tips

**First, ssh into MLP head node (from Dice):**

	ssh mlp1

You now have access to a filespace on the MLP node that is unique to your UID.

Set up conda on MLS head node

	opt/conda/bin/conda init bash
 	source .bashrc

Set up the environment:

	conda create --name env
 
Activate the environment:

	conda activate env
 
Install CUDA:

	conda install cuda
 
Install CuPy:

 	conda install -c conda-forge cupy

**To run an interactive job:**
Connect to node:

	srun --gres=gpu:1 --pty bash

Wait until you see a terminal message like 
> “... has been allocated resources”.

You are now connected to a node with access to a GPU. This node lets you run interactive jobs, which are short jobs that let you test GPU code (e.g. for debugging).

## To run a job

First, write a python script that does whatever you want to do.
If you want to save an output, ideally do it in the script using cupy.savetxt(), etc.

Then, write a shell script that looks this:

	#!/bin/bash
	~Path to python, in conda envs, to be improved upon~ ~path to python script~

Then, run:

	sbatch --gres=gpu:1 script.sh


and your output will be saved to the directory you called sbatch from.

## Useful Commands

You can specify GPUs:

	srun --gres=gpu:titan_x:1 --pty bash
	sbatch --gres=gpu:titan_x:1 test.sh
 
to use Titan X GPUs. We are allowed to request a maximum of 8 GTX 1060 GPUs, 4 Titan X GPUs, 1 Titan X Pascal GPU, or 2
A6000 GPUs at a time.

**Check all available GPU types:**
	
 	scontrol show node | grep gpu

**Check current SLURM job status:**

	squeue

 **Cancel a job:** 
 
 	scancel <job_id>

  For more info, check out
  https://computing.help.inf.ed.ac.uk/teaching-cluster

  ## Important Notes

Only install your environment and write your code on the head node. Actions such as Git operations and
similar tasks should also be performed on the head node, as they will not work on the compute nodes.

Only run your code on the compute nodes using srun or sbatch, as the head node has limited computing
resources and does not have GPUs. Running torch.cuda.is_available() on the head node will return
False.

Computing Support Team: https://computing.help.inf.ed.ac.uk/

## PyTorch Demo

They gave us a demo workthrough in rescources/1-pytorch-demo. Its worth a look.

![ScreenShot](/images/pytorch_demo.jpg)

