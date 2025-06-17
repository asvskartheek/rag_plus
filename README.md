# RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning

**UNOFFICIAL IMPLEMENTATION**

[Paper Link](https://arxiv.org/abs/2506.11555)

This paper does not have an official implementation, in this repository trying to implement and re-create the results of the paper with LLMs available to me.

**Note: For all the evaluation, we will use the prompt in the paper as the reference and create a dspy Signature to do the evaluation for consistency.**

## MathQA
They created their own dataset and evaluated on it, but they did not release the high quality 430 points dataset mentioned in the paper. So I will be using the AllenAI MathQA dataset, test split for evaluation.
[Dataset](https://huggingface.co/datasets/allenai/math_qa).
Will be running evaluation thrice for each one and averaging the results and reporting with std deviation.

Shit! need to find source dataset to do RAG. Can't find that either. They just mentioned `custom mathematical knowledge corpus` whatever the fuck that means! Will revisit this in a bit.

# MedQA
## Dataset
> We use the MedQA dataset curated by [Jin et al](https://arxiv.org/abs/2009.13081). We randomly sampled 500 examples from the dataset to serve as our training set. All selected items are in a multiple-choice format.
