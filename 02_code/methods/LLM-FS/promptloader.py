import os


class PromptLoader:
    def __init__(self, base_path="/home/niklasdonhauser/LLM-Few-Shot-OLAMA/prompt/"):
        self.base_path = base_path

    def load_prompt(self, task="tasd", prediction_type="label", aspects=[], examples=[], seed_examples=42,
                    input_example=None, polarities=["positive", "negative", "neutral"],
                    reasoning_step="reasoning",
                    load_llm_instruction=False):
        if task == "tasd":
            prompt_path = os.path.join(self.base_path, "prompt_de.txt")
        elif task == "acsa":
            prompt_path = os.path.join(self.base_path, "prompt_de_acsa.txt")

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file {prompt_path} not found.")

        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read()

        # Set Prompt Header
        prompt = prompt.replace("[[aspect_category]]", str(aspects)[1:-1])

        # Set Examples in Prompt
        if len(examples) == 0:
            prompt = prompt.replace("[[examples]]", "")
        elif prediction_type == "label":
            example_str = ""
            for example in examples:
                example_str += f'Text: {example["text"]}\nSentiment Elements: {str(example["tuple_list"])}\n'
            prompt = prompt.replace("[[examples]]", example_str)

        # Set Footer of Example to be predicted
        if prediction_type == "label":
            prompt += "Text: " + \
                input_example["text"] + "\nSentiment Elements:"

        return prompt
