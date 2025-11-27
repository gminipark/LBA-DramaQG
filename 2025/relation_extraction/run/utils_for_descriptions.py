import pandas as pd

def gather_examples(df: pd.DataFrame, rel: str, k: int = 5, seed: int = 42):
    sub = df[df["relation"] == rel]
    if len(sub) == 0:
        return []
    take = min(len(sub), k)
    ex = sub.sample(n=take, random_state=seed)[["text", "subject", "object"]]
    examples = []
    for _, r in ex.iterrows():
        examples.append({
            "sentence": str(r["text"]),
            "subject": str(r["subject"]),
            "object": str(r["object"]),
        })
    return examples


def build_prompt_for_relation(relation_name: str, examples: list[dict]) -> str:
    example_lines = []
    for i, ex in enumerate(examples, 1):
        example_lines.append(
            f"{i}) Sentence: {ex['sentence']}\n"
            f"   Subject: {ex['subject']}\n"
            f"   Object: {ex['object']}"
        )
    examples_block = "\n".join(example_lines) if example_lines else "(no examples available)"

    prompt = (
        "You are an expert annotator for relation extraction tasks.\n"
        "A relation label and several examples belonging to that label are provided.\n"
        f"Relation label: {relation_name}\n\n" 
        f"Examples (up to 5):\n{examples_block}\n\n"
        "Each relation label represents the relationship between the subject and object within a sentence.\n"
        "Using the example samples above, explain the meaning of the relation label in five to ten sentences in English.\n"
        "Your description will be used to annotate the relation in new input sentences.\n"
        f"Your description of {relation_name}:"
    )
    return prompt