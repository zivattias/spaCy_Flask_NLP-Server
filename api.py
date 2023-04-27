import spacy
import json
from flask import Flask, request, Response
from cache_ import Cache
from uuid import uuid4
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from nlp import split_into_sentences, detect_part_of_speech, detect_named_entities

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

    with ThreadPoolExecutor() as executor:
        executor.submit(split_into_sentences, data, uuid)

    return Response(
        json.dumps(
            cache.get(uuid),
        ),
        status=200,
        content_type="application/json",
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

    with ThreadPoolExecutor() as executor:
        executor.submit(detect_part_of_speech, tags, data, uuid)

    return Response(
        json.dumps(cache.get(uuid)), status=200, content_type="application/json"
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

    with ThreadPoolExecutor() as executor:
        executor.submit(detect_named_entities, data, uuid)

    return Response(
        json.dumps(cache.get(uuid)), status=200, content_type="application/json"
    )


if __name__ == "__main__":
    app.run(debug=True)
