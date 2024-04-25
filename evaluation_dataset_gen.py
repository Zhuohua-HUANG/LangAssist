from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
from openai_model import ChatLLM
from IPython.display import display

pd.set_option("display.max_colwidth", None)

filename = "data/all_courses_v2.json"
# Read the JSON data from the file
with open(filename, "r") as file:
    data = json.load(file)

# Extract the 'text' values into a new list
docs_processed = []
SAMPLE_SIZE = 10

i = 0
while i < len(data):
    current_ctx = []
    while len(current_ctx) < 5 and i < len(data):
        item = data[i]
        current_ctx.append(item["text"])
        i += 1
    docs_processed.append(",\n".join(current_ctx))


llm = ChatLLM()

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

I will give you some good examples when given contexts about AI or IT:
1. "what courses I can choose if i am majoring in Artificial Intelligence?"
2. "I am majoring in Information Technology and I want to know some knowledge about database, what courses can I take?"
3. "What can I learn from courses of Information Technology?"

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

import random

N_GENERATIONS = 15  # We intentionally generate only 10 QA couples here for cost and time considerations

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
    # Generate QA couple
    output_QA_couple = llm.call_llm(
        QA_generation_prompt.format(context=sampled_context)
    )
    try:
        question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
        answer = output_QA_couple.split("Answer: ")[-1]
        # assert len(answer) < 300, "Answer is too long"
        output = {
            "context": sampled_context,
            "question": question,
            "answer": answer,
            # "source_doc": sampled_context.metadata["source"],
        }
        print("output:", output)
        outputs.append(output)
    except:
        print("Fail to parse, skip", output_QA_couple)
        continue


# ======

# Evaluation
question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to students learning about the information of potential courses they may care.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

I will give you some good sample questions that should obtain 5:
1. "what courses I can choose if i am majoring in Artificial Intelligence?"
2. "I am majoring in Information Technology and I want to know some knowledge about database, what courses can I take?"

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


print("Generating critique for each QA couple...")

for output in tqdm(outputs):
    evaluations = {
        "groundedness": llm.call_llm(
            question_groundedness_critique_prompt.format(
                context=output["context"], question=output["question"]
            ),
        ),
        "relevance": llm.call_llm(
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        continue


pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(outputs)

print("Evaluation dataset before filtering:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
        ]
    ]
)

generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
]

print("============================================")
print("Final evaluation dataset:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
        ]
    ]
)

eval_dataset = datasets.Dataset.from_pandas(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
        ]
    ],
    preserve_index=False,
)

eval_dataset.to_csv("data/eval_dataset.csv", index=False)
eval_dataset.to_json("data/eval_dataset.json", index=False)
