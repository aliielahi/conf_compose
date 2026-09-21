"""Print one rendered prompt and one verification probe, to see what a model is actually asked and answers."""

import argparse

from conf_compose.confidence_estimators import SelfVerification, zero_shot_targets
from conf_compose.constants import TASKS
from conf_compose.data import get_task
from conf_compose.pipelines.inference import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="csqa", choices=sorted(TASKS))
    args = parser.parse_args()

    task = get_task(args.task)
    example = task.load("test", n=1)[0]
    llm = load_model(args.model)
    print(f"generation_prefix={getattr(llm, 'generation_prefix', '')!r} "
          f"chat_template_kwargs={getattr(llm, 'chat_template_kwargs', {})}")
    print("\n=== rendered prompt (tail) ===")
    print(repr(llm.render(None, [{"role": "user", "content": task.prompt(example)}])[-300:]))

    generation = llm.generate([task.prompt(example)], max_tokens=TASKS[args.task]["max_tokens"],
                              temperature=0.0, logprobs=True)[0]
    print(f"\n=== response (first 300 chars) ===\n{generation.text[:300]}")
    print(f"\nextracted answer: {task.extract_answer(generation.text)}")

    target = zero_shot_targets(task, [example], [generation])[0]
    estimator = SelfVerification(llm)
    value = estimator.estimate(task, [target])[0]
    print(f"\nverification: {value} | top tokens: {getattr(estimator, 'sample_tokens', 'label found')}")


if __name__ == "__main__":
    main()
