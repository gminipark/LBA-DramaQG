def build_prompt(relation_list, sentence: str, subject: str, obj: str, cluster_name:str) -> str:
    rel_list = ", ".join(relation_list)
    prompt = (
        "Your task is to evaluate whether the predicted relation between the subject and object in the given sentence is correct.\n"
        f"Given the possible relations: [{rel_list}].\n\n"
        "The relationship between the subject and object in the sentence must belong to one of the relations in this list.\n"
        "If you determine that the predicted relation is correct, respond with 'Correct classification'.\n"
        "If the prediction is incorrect and another relation from the list is more appropriate, respond with the correct relation label.\n"
        "Do not provide any explanation other than the judgment result. \n\n"
        f"Sentence: {sentence}\n"
        f"Subject: {subject}\n"
        f"Object: {obj}\n"
        f"predicted relation: {cluster_name}\n"
        "Your evaluation: "
    )
    return prompt

def call_gpt(prompt: str, client, model="gpt-5"):
    try:
        result = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "high"},
            text={"verbosity": "low"},
        )
        return result.output_text
    except Exception as e:
        print(f"[ERROR] API 호출 실패: {e}")
        return "ERROR"
