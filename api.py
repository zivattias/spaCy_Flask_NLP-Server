import spacy
import json
import time
from flask import Flask, request, Response
from uuid import uuid4
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from .cache import Cache

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)

cache = Cache()
cache_lock = Lock()


def _generate_id() -> str:
    return str(uuid4())


@app.route("/")
def index():
    return "<p>Natural Language Processing server, powered by spaCy"


@app.post("/sentences")
def sents():
    data = request.form.get("data")
    if not data:
        error_msg = {"data": "Missing required field"}
        return Response(
            json.dumps(error_msg), status=400, content_type="application/json"
        )

    uuid = _generate_id()

    with cache_lock:
        cache.add(uuid)

    def split_into_sentences(data, uuid):
        time.sleep(10)
        try:
            doc = nlp(data)
            sentences = [str(sentence) for sentence in doc.sents]

            response = {
                "data": {
                    "input": data,
                    "amount": len(sentences),
                    "sentences": sentences,
                }
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

    executor = ThreadPoolExecutor()
    executor.submit(split_into_sentences, data, uuid)

    return Response(
        json.dumps({uuid: cache.get(uuid)}), status=200, content_type="application/json"
    )


@app.post("/pos")
def pos():
    tags, data = request.args.get("tags"), request.form.get("data")
    if not tags or not data:
        error_msg = {"tags/data": "Missing required field(s)"}
        return Response(
            json.dumps(error_msg), status=400, content_type="application/json"
        )

    uuid = _generate_id()

    with cache_lock:
        cache.add(uuid)

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

    executor = ThreadPoolExecutor()
    executor.submit(detect_part_of_speech, tags, data, uuid)

    return Response(
        json.dumps({uuid: cache.get(uuid)}), status=200, content_type="application/json"
    )


@app.post("/ents")
def ents():
    data = request.form.get("data")
    if not data:
        return Response(
            json.dumps({"data": "Missing required field"}),
            status=400,
            content_type="application/json",
        )

    uuid = _generate_id()

    with cache_lock:
        cache.add(uuid)

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

    executor = ThreadPoolExecutor()
    executor.submit(detect_named_entities, data, uuid)

    return Response(
        json.dumps({uuid: cache.get(uuid)}), status=200, content_type="application/json"
    )


@app.get("/cache")
def get_cache():
    return Response(
        json.dumps(cache.get_all()), status=200, content_type="application/json"
    )


@app.get("/tasks/<task_id>/status")
def get_task_status(task_id: str):
    if task_id not in cache._cache:
        return Response(
            {"task_id": "Wrong task ID"}, status=400, content_type="application/json"
        )
    return Response(
        json.dumps(cache.get(task_id)["status"]),
        status=200,
        content_type="application/json",
    )


@app.get("/tasks/<task_id>/result")
def get_task_result(task_id: str):
    if task_id not in cache._cache:
        return Response(
            {"task_id": "Wrong task ID"}, status=400, content_type="application/json"
        )
    if cache._cache[task_id]["status"] != "complete":
        return Response(
            {"task_id": "Task is still processing"},
            status=400,
            content_type="application/json",
        )
    return Response(
        json.dumps(cache.get(task_id)["result"]),
        status=200,
        content_type="application/json",
    )


if __name__ == "__main__":
    app.run(debug=True)
