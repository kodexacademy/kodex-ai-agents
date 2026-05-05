import json
from llm_client import call_llm

# Load evaluation instructions
with open("resources/evaluation_instructions.txt", "r") as f:
    EVALUATION_INSTRUCTIONS = f.read()

# --------------------------
# evaluation after Step
# --------------------------
def evaluation_step(current_doc: str, step_name: str):
    """Ask LLM to inspect the corrected document and decide if next step is needed."""
    evaluation_instructions = EVALUATION_INSTRUCTIONS.replace("EVALUATION_STEP", step_name)
    messages = [
        {"role": "system", "content": evaluation_instructions},
        {"role": "user", "content": f"Inspect the document for {step_name}:\n{current_doc}\nIs it ready for next correction step?"}
    ]

    response = call_llm(messages)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {"proceed_next": True, "issues_found": ""}  # default to continue

    print(f"evaluation {step_name} Completed")
    return result


def process_evaluation_step(corrected_doc: str, step_name: str):
    evaluation_result = evaluation_step(corrected_doc, step_name)
    evaluation_result["stage"] = "evaluation"
    evaluation_result["step"] = step_name
    return evaluation_result