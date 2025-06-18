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
> We use the MedQA dataset curated by [Jin et al](https://arxiv.org/abs/2009.13081). We randomly sampled 500 examples from the dataset to serve as our training set. All selected items are in a multiple-choice format. [Source: A.3.3]

[Dataset](https://huggingface.co/datasets/bigbio/med_qa) eventhough the description says it has in all three languages - English, Simplified Chinese and Traditional Chinese. While loading, I only got the english split (TODO: need to see why), so anyway using it directly to do all the experiments.

## Corpus
> The medical corpus is sourced from [Xiong et al., 2024](https://arxiv.org/abs/2402.13178). We use only the texbook portion of the corpus, where knowledge is already structured into discrete knowledge points,each representing a self-contained fragment.
The corpus spans 18 subjects and includes
64,117 knowledge points, with a total size of
99,382 KB. Each knowledge point, averaging less
than 600 words, was treated as a single chunk without further segmentation. The knowledge encompasses both conceptual and procedural content.
We used a chunk size of 800 tokens, consistent
with other domains.

[Dataset](https://huggingface.co/datasets/MedRAG/textbooks) has 125,847 datapoints which is different from the 64,117 mentioned in the paper. But I will use this, need to get some clarification (TODO) but this is good, I will just use this, maybe they did some weird filtering.

## Results
**GPT-4o**

Baseline: 0.87