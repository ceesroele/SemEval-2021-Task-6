# SemEval-2021-Task-6

## Introduction
This is the WVOQ team code for participation in [SemEval 2021 Task 6: "Detection of Persuasion Techniques in Texts and Images"](https://propaganda.math.unipd.it/semeval2021task6/index.htm).

WVOQ participated in all three subtasks. Most interesting is the contribution to subtask 2. Simultaneously detecting a span and classifying it was done through a sequence-to-sequence model.

## Architecture

The system is built with [Simple Transformers](https://simpletransformers.ai/), which is a task-oriented framework on top of [Hugging Face Transformers](https://huggingface.co/transformers/).

Configuration is in dev.yaml. Here you find the configuration for the three tasks and options to run specific functionality, e.g. to create predictions on the basis of a trained model.

There are five major modules:
* `load_data.py` - load the data and convert it to a format using the Fragment dataclass
* `train.py` - train any type of model
* `eval.py` - evaluate any type of model
* `postprocess.py` - predict
* `pipeline.py` - wrap the different steps together.

The system is started with: `python pipeline.py` which will select `cur_scenario` from `dev.yaml` for execution.

## Future work

The present system was set up for experimentation and the code still contains many traces of experiments done in the past that weren't used in the final submissions to the tasks.

As only the contribution to subtask 2 is interesting, I will extract it into a standalone system.


