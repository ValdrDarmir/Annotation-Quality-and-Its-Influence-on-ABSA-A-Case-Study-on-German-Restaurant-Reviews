import re
import string


def exchange_phrases(asp_current, asp_exchange, text, returnFalseError=False):
    # create dict with phrases and their exchange
    phrases_dict = {}
    for idx, asp in enumerate(asp_current):
        if asp[0] != "NULL":
            phrases_dict[asp[0]] = asp_exchange[idx][0]
        if len(asp) == 4 and asp[3] != "NULL":
            phrases_dict[asp[3]] = asp_exchange[idx][3]

    phrases = [tup[2] for tup in asp_current if tup[2] != "NULL"]
    if len(asp_current[0]) == 4:
        phrases += [tup[3] for tup in asp_current if tup[3] != "NULL"]

    phrases = list(set(phrases))
    # sort phrases by length (prio 1) and alphabetically (prio 2)
    phrases.sort(key=lambda x: (len(x), x))
    phrases.reverse()

    for phrase in phrases:
        text = text.replace(phrase, phrases_dict[phrase])

    for gph in phrases_dict.values():
        if not (gph in text):
            print("Warning: The phrase " + gph + " is not in the text ", text)
            if returnFalseError:
                return False

    return text


def extract_array_from_string(predicted_label):
    match = re.search(r"\[(.*\])", predicted_label)
    if match:
        return match.group(0)
    else:
        return None


def to_pred_list(text, n_elements):
    text_original = text
    text = text[text.find("["):][1:]
    text = "[" + text
    text = text.replace("\n", "")

    text = re.sub(r"\[ *\( *[\'\"]", "", text)
    text = re.sub(r"[\"\'] *, *[\'\"]", "#####", text)
    text = re.sub(r"[\"\']\) *, *\([\"\']", "#####", text)
    text = re.sub(r"[\"\'] *\) *\]", "", text)
    matches = text.split("#####")
    matches = [match.strip() for match in matches]
    label = []
    for i in range(0, len(matches), n_elements):
        if len(matches[i: i + n_elements]) != n_elements:
            raise ValueError(f"Error in text: {text_original}")
        label.append(matches[i: i + n_elements])

    return label


def validate_label(predicted_label, input_text, unique_aspect_categories,
                   polarities=["positive", "negative", "neutral"], task="tasd",
                   is_string=True, allow_small_variations=False, check_unique_ac=True):
    """
    Validate predicted label structures for tasd, asqp, or acsa tasks.

    - tasd: (category, polarity, aspect phrase)
    - asqp: (category, polarity, aspect phrase, opinion term)
    - acsa: (category, polarity)
    """

    # --- Step 1: Parse input string if necessary ---
    if isinstance(predicted_label, str):
        extracted = extract_array_from_string(predicted_label)
        if extracted is None:
            return [False, "no list in prediction"]
        try:
            extracted = extracted.strip().replace("```", "").strip()
            if extracted.endswith("]]") and not extracted.startswith("[["):
                extracted = extracted[:-1]
            label = eval(extracted)
            if isinstance(label, tuple):
                label = [label]
        except Exception:
            return [False, "failed to eval string to list"]
    else:
        label = predicted_label

    # --- Step 2: Check structure ---
    if not isinstance(label, list):
        return [False, "not a list"]
    if len(label) < 1:
        return [False, "no tuple found"]
    for element in label:
        if not isinstance(element, tuple):
            return [False, "inner elements not of type tuple"]

    # --- Step 3: Check tuple length per task ---
    n_elements_task = {"tasd": 3, "asqp": 4, "acsa": 2}
    for aspect in label:
        if len(aspect) != n_elements_task[task]:
            return [False, f"tuple has not exactly {n_elements_task[task]} elements"]

        for idx, item in enumerate(aspect):
            if not isinstance(item, str):
                return [False, "element not of type string"]
            if len(item) < 1:
                return [False, "element string is empty string"]

            if idx == 1 and item not in polarities:
                return [False, f"item {item} not a sentiment"]

            if idx == 0 and check_unique_ac and item not in unique_aspect_categories:
                return [False, f"item {item} is not a correct aspect category"]

    # --- Step 4: Check if terms exist in input_text ---
    if task != "acsa":  # only tasd / asqp need phrase checks
        if allow_small_variations:
            new_label = []
            for tup in label:
                tup = list(tup)

                # Aspect phrase check
                if tup[2] != "NULL" and tup[2].lower() not in input_text.lower():
                    candidate = check_if_similar_phrase_exists(
                        tup[2], input_text)
                    if candidate is False:
                        return [False, "aspect term not in text"]
                    tup[2] = candidate
                elif tup[2] != "NULL":
                    start = input_text.lower().index(tup[2].lower())
                    end = start + len(tup[2])
                    tup[2] = input_text[start:end]

                # Opinion term check (only asqp)
                if task == "asqp":
                    if tup[3] != "NULL" and tup[3].lower() not in input_text.lower():
                        candidate = check_if_similar_phrase_exists(
                            tup[3], input_text)
                        if candidate is False:
                            return [False, "opinion term not in text"]
                        tup[3] = candidate
                    elif tup[3] != "NULL":
                        start = input_text.lower().index(tup[3].lower())
                        end = start + len(tup[3])
                        tup[3] = input_text[start:end]

                new_label.append(tuple(tup))
            label = new_label
        else:
            for tup in label:
                if tup[2] != "NULL" and tup[2] not in input_text:
                    return [False, "aspect term not in text"]
                if task == "asqp" and tup[3] != "NULL" and tup[3] not in input_text:
                    return [False, "opinion term not in text"]

    # --- Step 5: Strip whitespace and return ---
    label = [tuple([t.strip() for t in tup]) for tup in label]
    return [label]


def validate_reasoning(output):
    if len(output) == 0:
        return [False, "no text found"]

    return [True]


all_chars = string.ascii_letters + string.digits + \
    "äöüÄÖÜßàâçéèêëîïôûùüáéíóúüñàèéìíóòùáčďéěíňóřšťúůýžáéëïóöüабвгдеёжзийклмнопрстуфхцчшщыэюя" + "çğıöşü"


def check_if_similar_phrase_exists(phrase, text_gen):
    # Überprüfen, ob die Phrase eine ähnliche Phrase im Text mit einer maximalen Veränderung von 1 Zeichen hat
    for i in range(len(phrase)):
        for char in all_chars:  # Verwenden des langen Alphabets
            modified_phrase = phrase[:i] + char + phrase[i+1:]
            if modified_phrase in text_gen:
                return modified_phrase
    return False
