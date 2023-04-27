import spacy
from api import cache, cache_lock

nlp = spacy.load("en_core_web_sm")


def split_into_sentences(data, uuid):
    try:
        doc = nlp(data)
        sentences = [str(sentence) for sentence in doc.sents]

        response = {
            "data": {"input": data, "amount": len(sentences), "sentences": sentences}
        }

        with cache_lock:
            cache.update(uuid, "complete", response)

    except Exception as e:
        response = {
            "data": {
                "input": data,
            },
            "error": e,
        }

        with cache_lock:
            cache.update(uuid, "error", response)


def detect_part_of_speech(tags, data, uuid):
    # Split tags if many, and manipulate data (white space trim and uppercase)
    tags = [tag.upper().replace(" ", "") for tag in tags.split(",")]

    # All available tags in spaCy
    all_tags = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]

    if any(tag not in all_tags for tag in tags):
        with cache_lock:
            cache.update(
                uuid,
                "error",
                {tag: "Invalid POS tag" for tag in tags if tag not in all_tags},
            )

    try:
        doc = nlp(data)
        response = {
            tag: list(
                map(
                    lambda token: token.text,
                    filter(lambda token: token.pos_ == tag, doc),
                )
            )
            for tag in tags
        }

        with cache_lock:
            cache.update(uuid, "complete", response)

    except Exception as e:
        response = {
            "data": {
                "input": data,
            },
            "error": e,
        }

        with cache_lock:
            cache.update(uuid, "error", response)


def detect_named_entities(data, uuid):
    try:
        doc = nlp(data)
        response: dict = {}
        for entity in doc.ents:
            if entity.label_ not in response:
                response[entity.label_] = []
            response[entity.label_].append(entity.text)

        with cache_lock:
            cache.update(uuid, "complete", response)

    except Exception as e:
        response = {
            "data": {
                "input": data,
            },
            "error": e,
        }

        with cache_lock:
            cache.update(uuid, "error", response)
