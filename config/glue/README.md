# GLUE benchmarks

GLUE is a well-established benchmark for NLP systems, comprising various sentence-level language understanding tasks. The most effective solutions for these tasks are large models trained on extensive datasets, typically large pretrained language models (LLMs) followed by fine-tuning with a task-specific head.

We selected two tasks: SST-2, a sentiment analysis task closely related to the objective of the IMDB dataset, and QQP (Quora Question Pairs), which evaluates whether two questions are semantically equivalent. For both tasks, as is standard practice, we used a tokenizer to map sentences into sequences of indices, padded to a fixed length of 128 timesteps. For sentence pairs, as recommended (and done in the reference code), we concatenated both sentences with two separator tokens in between.

This experiment requires the transformer library from HuggingFace (tested with version 4.46.3)
It download the bert tokenizer "bert-base-uncased"

The experiment will produce outputs taht can be submitted to the [Glue benchmark](https://gluebenchmark.com/submit) website