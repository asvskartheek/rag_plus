from datasets import load_dataset, Dataset
import dspy
from dspy import Signature, InputField, OutputField
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate_medqa(solver):
    # Load and convert to list for length operations
    test_data = list(load_dataset("bigbio/med_qa", split="test"))  # .select(range(10))

    def process_item(item):
        res = solver(
            problem=item["question"],
            options=str(item["options"]),
        )

        return res.answer.lower(), item["answer_idx"].lower(), res.reasoning

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_item, item) for item in test_data]
        for future in tqdm(as_completed(futures), total=len(test_data)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing item: {e}")
                results.append(("error", "error", "error"))

    y_hat, y_true, reasonings = zip(*results)

    # save preds to a csv file, using pandas
    df = pd.DataFrame(
        {
            "id": range(len(y_hat)),
            "prediction": y_hat,
            "truth": y_true,
            "reasoning": reasonings,
        }
    )
    df.to_csv(
        f"preds/medqa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )

    # calculate accuracy using pandas
    accuracy = (df["prediction"] == df["truth"]).mean()
    print(f"Accuracy: {accuracy:.2f}")


# Baseline prompt & equivalent signature by me.
BASE_PROMPT = """
You are a decision-evaluation assistant. Please help me solve the medicine question. Please answer the question step by step in a
XML format with the reasoning step enclosed with the tag <Think></Think> and the answer option enclosed with the tag
<Answer></Answer>. You must choose one correct option among A -D.
Here is the question: [Put The Questions Here]
Options: [Put the Options Here]
Answer:
"""  # ref: paper, fig 11


class MedQABaseline(Signature):
    """
    You are a decision-evaluation assistant. Please help me solve the medicine question.
    """

    problem = InputField(description="Medicine question to solve.")
    options = InputField(description="The answer options for the medicine question.")
    reasoning = OutputField(
        description="The step-by-step reasoning to solve the question."
    )
    answer = OutputField(
        description="The correct option among A-E. Just the letter, no other text."
    )


# RAG prompt & equivalent module/signature
RAG_PROMPT = """
You are a decision-evaluation assistant. Your task is to solve the given question based on the Reference Knowledge. There are
some knowledge to help you solve the problem:
<Reference>
<Knowledge>
[Put the Knowledge Here]
</Knowledge>
<Knowledge>
[Put the Knowledge Here]
</Knowledge>
<Knowledge>
[Put the Knowledge Here]
</Knowledge>
</Reference>
Now use the reference knowledge provided for guidance (but do not be strictly bound by them), please answer the question
step by step in a XML format with the reasoning step enclosed with the tag <Think></Think> and the answer option enclosed
with the tag <Answer></Answer>. You must choose one correct option among A -D.
Here is the question: [Put the Questions Here]
Options: [Put the Options Here]
Answer:
"""


if __name__ == "__main__":
    # LLM
    from llms import gemma_4b, azure_gpt4o

    dspy.settings.configure(lm=azure_gpt4o)

    solver = dspy.ChainOfThought(MedQABaseline)

    # testing
    # dataset = load_dataset("bigbio/med_qa")
    # test_data = dataset["test"]
    # print(json.dumps(test_data[0], indent=4))
    # print("-" * 80)

    # res = solver(problem=test_data[0]["question"], options=str(test_data[0]["options"]))
    # print(res)

    # evaluation
    evaluate_medqa(solver)
