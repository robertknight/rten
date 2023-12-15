# Reference implementation of BERT extractive question answering using
# Hugging Face Transformers.

from argparse import ArgumentParser
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline


# Run inference on a question answering model using a Hugging Face transformers
# pipeline.
#
# See https://huggingface.co/docs/transformers/v4.35.2/en/tasks/question_answering#inference.
def eval_qa_model(model_name: str, context: str, question: str):
    """
    :param model_name: Name of Hugging Face model trained for question answering
    :param context: Context to search for answer to question
    :param question: Question to answer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    oracle = pipeline(task="question-answering", model=model, tokenizer=tokenizer)

    start = time.perf_counter()
    result = oracle(question=question, context=context)
    end = time.perf_counter()
    print(f"Result from `tokenizers` pipeline in {end-start:.2f}s:", result)


parser = ArgumentParser(
    description="""
Perform extractive question answering using BERT.

This is a reference implementation using Hugging Face Transformers.
"""
)
parser.add_argument("context", help="Path to text file containing context")
parser.add_argument("question", help="Question to answer")
parser.add_argument(
    "--model",
    help="Name of the Hugging Face model",
    # For more models, search for "bert squad" on HF:
    # https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads&search=bert+squad
    default="deepset/bert-base-cased-squad2",
)
args = parser.parse_args()

with open(args.context) as context_fp:
    context = context_fp.read()

eval_qa_model(args.model, context, args.question)
