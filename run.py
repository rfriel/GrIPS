import sys
sys.path.append('GrIPS')

from GrIPS import grips

import automatic_prompt_engineer.template

from functools import partial
from automatic_prompt_engineer.evaluation.likelihood import score_likelihood

words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

eval_data = (words, antonyms)
# few_shot_data = eval_data
few_shot_data = (['fake']*len(words), ['fake']*len(words),)

eval_template = \
"""Instruction: [PROMPT]

Input: [INPUT]

Answer:[OUTPUT]"""

demos_template = '''
Input: [INPUT]

Answer: [OUTPUT]'''

eval_template = automatic_prompt_engineer.template.EvalTemplate(eval_template)
demos_template= automatic_prompt_engineer.template.DemosTemplate(demos_template)

instruction = "Write an antonym to the following word."



score_fn = partial(
    score_likelihood,
    eval_template=eval_template,
    demos_template=demos_template,
    eval_data=eval_data,
    few_shot_data=few_shot_data,
    verbose=0,
)

out = grips.grips(
    instruction=instruction,
    edit_operations_ = ('del', 'swap', 'add'),
    score_fn=score_fn,
    num_candidates=5,
    patience=2,
)

print(out)