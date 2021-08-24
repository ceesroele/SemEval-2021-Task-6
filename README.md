# SemEval-2021-Task-6

The code here is the full code used by WVOQ for all three tasks of SemEval-2021 Task 6.

You can find my article [WVOQ at SemEval-2021 Task 6: BART for Span Detection and Classification](https://aclanthology.org/2021.semeval-1.32/) 
as part of the [Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)](https://aclanthology.org/volumes/2021.semeval-1/).

The poster presented at the ACL/IJCNLP / SemEval2021 is here at [doc/bart_for_span_detection_poster.pdf](doc/bart_for_span_detection.pdf).

-----

Most likely you are interested only in the code for task 2:
simultaneously detecting a span and classifying it.

That code is now taken out of the code-base, cleaned up, given installation instructions, 
and available at [https://github.com/ceesroele/span_model](https://github.com/ceesroele/span_model).

Improvements:
- Simpler to understand
- Re-usable for simultaneous span detection and classification tasks
- And it is more likely to run on your machine too, not just on mine ...

-----


## Introduction
This is the WVOQ team code for participation in [SemEval 2021 Task 6: "Detection of Persuasion Techniques in Texts and Images"](https://propaganda.math.unipd.it/semeval2021task6/index.html).

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

Deal with the two main causes of systemic errors:
1. Begin and end tags are not matching
2. Words or characters are introduced in the generated sentences that were not in the input

Ideas are:
* Train with half-masked sentences consisting only of begin and end tags (pre-training for tags)
* Add functionality to the generator code in Transformers to prevent tokens other than
tags in the output that are not in the input.

