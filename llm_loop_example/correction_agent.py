import os
import shutil
import json
from llm_client import call_llm

# Load correction instructions
with open("resources/correction_instructions.txt", "r") as f:
    CORRECTION_INSTRUCTIONS = f.read()

# Corrected file path
corrected_file_path = "corrected_files"

# Delete if it exists
if os.path.exists(corrected_file_path) and os.path.isdir(corrected_file_path):
    shutil.rmtree(corrected_file_path)
    print(f"Deleted directory: {corrected_file_path}")

# Recreate the directory
os.makedirs(corrected_file_path, exist_ok=True)
print(f"Created directory: {corrected_file_path}")

# --------------------------
# Single Step Correction
# --------------------------
def correct_step(current_doc: str, step_name: str, feedback: str):
    """Run LLM to correct the document for a single step."""
    correction_instructions = CORRECTION_INSTRUCTIONS.replace("CORRECTION_STEP", step_name).replace("FEEDBACK", feedback)
    messages = [
        {"role": "system", "content": correction_instructions},
        {"role": "user", "content": f"Step: {step_name}\nCurrent document:\n{current_doc}"}
    ]

    response = call_llm(messages)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Retry once if JSON is invalid
        messages.append({"role": "user", "content": "Your output was invalid JSON. Retry using only JSON."})
        response = call_llm(messages)
        result = json.loads(response)

    corrected_doc = result.get("corrected_document", current_doc)
    reasoning = result.get("reasoning", "")
    done = result.get("done", False)

    print(f"{step_name} Completed\nReasoning: {reasoning}\nCorrected Document:\n{corrected_doc}")
    return {
        "step": step_name,
        "reasoning": reasoning,
        "corrected_document": corrected_doc,
        "done": done
    }


def process_correction_step(current_doc: str, step_index: int, step_name: str, feedback: str):
    # Step correction
    correction_result = correct_step(current_doc, step_name, feedback)

    # Save step output
    step_file = os.path.join(corrected_file_path, f"step_{step_index}_{step_name.replace(' ', '_')}.txt")
    with open(step_file, "w", encoding="utf-8") as f:
        f.write(correction_result["corrected_document"])
    print(f"📂 Saved step {step_index}_{step_name} output to {step_file}")

    return correction_result