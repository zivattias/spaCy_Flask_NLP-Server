import spacy
import json
from flask import Flask, request, Response

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Natural Language Processing server, powered by spaCy"


@app.post("/sentences")
def split_into_sentences():
    data = request.form.get("data")
    if not data:
        error_msg = {"data": "Missing required field"}
        return Response(
            json.dumps(error_msg), status=400, content_type="application/json"
        )
    doc = nlp(data)
    sentences = [str(sentence) for sentence in doc.sents]

    response = {
        "data": {"input": data, "amount": len(sentences), "sentences": sentences}
    }
    return Response(
        json.dumps(
            response,
        ),
        status=200,
        content_type="application/json",
    )


@app.post("/pos")
def detect_part_of_speech():
    tags, data = request.args.get("tags"), request.form.get("data")
    if not tags or not data:
        error_msg = {"tags OR data": "Missing required field(s)"}
        return Response(
            json.dumps(error_msg), status=400, content_type="application/json"
        )

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
        return Response(
            json.dumps({tag: "Invalid POS tag" for tag in tags if tag not in all_tags}),
            status=400,
            content_type="application/json",
        )

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
    return Response(json.dumps(response), status=200, content_type="application/json")


@app.post("/ents")
def detect_named_entities():
    data = request.form.get("data")
    if not data:
        return Response(
            json.dumps({"data": "Missing required field"}),
            status=400,
            content_type="application/json",
        )

    doc = nlp(data)
    response: dict[str, list] = {}
    for entity in doc.ents:
        if entity.label_ not in response:
            response[entity.label_] = []
        response[entity.label_].append(entity.text)

    return Response(json.dumps(response), status=200, content_type="application/json")


if __name__ == "__main__":
    app.run(debug=True)
