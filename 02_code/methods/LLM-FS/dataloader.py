import json
import os


class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_data(self, data_type: str, cv=False, seed=42, name=None):

        file_path = os.path.join(self.base_path, name)
        if name is None:
            name = data_type

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    entry = json.loads(line.strip())
                    text = entry.get("text", "").strip()
                    # trainset
                    if data_type == "train":
                        # test data has only id, original_id, text
                        data.append({
                            "id": entry.get("id", f"{idx}_{name}"),
                            # "original_id": entry.get("original_id", None),
                            "text": text
                        })
                except json.JSONDecodeError as e:
                    print(f"Skipping line {idx} due to JSON error: {e}")
                    continue
        print("Data in loader:", data)
        return data


class ExamplesLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_examples(self, max_examples=None, seed=None):
        examples = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_examples is not None and idx >= max_examples:
                    break
                data = json.loads(line.strip())
                # Build tuple_list from labels
                tuple_list = [tuple(label) for label in data.get("labels", [])]

                example = {
                    "text": data.get("text"),
                    "tuple_list": tuple_list
                }
                examples.append(example)

        if seed is not None:
            import random
            random.seed(seed)
            random.shuffle(examples)

        return examples
