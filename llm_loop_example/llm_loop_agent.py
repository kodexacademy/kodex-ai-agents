import os
import json
from correction_agent import process_correction_step, corrected_file_path
from evaluation_agent import process_evaluation_step

# Load correction steps
with open("resources/steps.json", "r") as f:
    STEPS = json.load(f)["steps"]

# Load input document
with open("resources/document.txt", "r") as f:
    raw_document = f.read()

# --------------------------
# Main Correction Loop
# --------------------------
def document_correction_loop(document: str, max_retry = 3):
    result_history = []
    current_doc = document

    for step_index, step_name in enumerate(STEPS):
        print(f"\n=== Step {step_index+1}: {step_name} ===")
        feedback = ""

        for retry in range(max_retry):
            # Correction proceeding
            correction_result = process_correction_step(current_doc, step_index+1, step_name, feedback)
            current_doc = correction_result["corrected_document"]
            result_history.append(correction_result)

            # Evaluation correction before proceeding
            inspection_result = process_evaluation_step(current_doc, step_name)
            result_history.append(inspection_result)
            proceed_next = inspection_result["proceed_next"]

            if proceed_next:
                break
            else:
                feedback = inspection_result["issues_found"]  # take out feedback to set in next iteration

    return result_history, current_doc

# --------------------------
# Run Agent
# --------------------------
if __name__ == "__main__":
    result_history, final_doc = document_correction_loop(raw_document)
    # print(result_history)

    print("\n--- Final Corrected Document Saved to final_doc.txt ---")
    final_doc_path = os.path.join(corrected_file_path, "final_doc.txt")
    with open(final_doc_path, "w", encoding="utf-8") as f:
        f.write(final_doc)
