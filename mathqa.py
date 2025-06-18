from datetime import datetime
from datasets import load_dataset
import dspy
from dspy import Signature, InputField, OutputField
import json
from tqdm import tqdm
import pandas as pd


def evaluate_mathqa(solver):
    test_data = list(
        load_dataset("allenai/math_qa", split="test")
    )  # .select(range(10))

    y_hat = []
    y_true = []
    for item in tqdm(test_data, total=len(test_data)):
        res = solver(problem=item["Problem"], options=item["options"])
        y_hat.append(res.answer)
        y_true.append(item["correct"])

    # save preds to a csv file, using pandas
    df = pd.DataFrame({"id": range(len(y_hat)), "prediction": y_hat, "truth": y_true})
    df.to_csv(
        f"preds/mathqa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )

    # calculate accuracy using pandas
    accuracy = (df["prediction"] == df["truth"]).mean()
    print(f"Accuracy: {accuracy:.2f}")


# Baseline prompt & equivalent signature by me.
BASE_PROMPT = """
Please help me solve the numerical analysis question. Please answer the question step by step in a XML format with the
reasoning step enclosed with the tag <Think></Think> and the answer option enclosed with the tag <Answer></Answer>. You
must choose one correct option among A-D.
Here is the question: [Put the Questions Here]
Options: [Put the Options Here]
Answer:
"""  # ref: paper, fig 14


class MathQABaseline(Signature):
    """
    Please help me solve the numerical analysis question.
    """

    problem = InputField(description="Numerical analysis question to solve.")
    options = InputField(description="The answer options for the math problem.")
    reasoning = OutputField(
        description="The step-by-step reasoning to solve the problem."
    )
    answer = OutputField(
        description="The correct option among a-d. Just the letter, no other text."
    )


if __name__ == "__main__":
    # testing
    # dataset = load_dataset("allenai/math_qa")
    # test_data = dataset["test"]
    # print(json.dumps(test_data[0], indent=4))
    # print('-' * 80)

    # res = solver(problem=test_data[0]["Problem"],
    #     options=test_data[0]["options"])
    # print(res)

    # evaluate Gemma 4B on MathQA
    from llms import gemma_4b, azure_gpt4o

    dspy.settings.configure(lm=azure_gpt4o)

    solver = dspy.ChainOfThought(MathQABaseline)
    evaluate_mathqa(solver)
