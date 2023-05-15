import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pre-trained model for semantic similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')


topics = [
    "Present simple",
    "Future: Present simple",
    "Present continuous and present simple",
    "Present simple (incl. questions and denials)",
    "Questions in the present simple",
    "Future (future tense): will, going to, present continuous en present simple",
    "Present continuous",
    "Past simple (be/have)",
    "Past simple (irregular verbs)",
    "Past simple (past tense)",
    "Past simple (past tense) incl. questions and denials",
    "Future (future tense): going to",
    "Can (kunnen)",
    "Can and could",
    "Modals: can / can't en could / couldn't",
    "Tag questions (short questions) - basis + can / can't",
    "Questions with to do",
    "Questions and denials with did",
    "Questions with to be & have got",
    "Interrogative pronouns",
    "Negations (negations) with to be (zijn)",
    "Imperative",
    "Personal pronouns (object) (object pronoun)",
    "Personal Pronouns (Subject) and Possessive Pronouns",
    "Personal pronouns",
    "Some and any (basis)",
    "Some and any",
    "Interrogative pronouns",
    "Degrees of comparison: one, two or more syllables",
    "Degrees of comparison: -er/-est, more / most and irregular",
    "Degrees of comparison, incl. good/bad and (not) as ... as",
    "Articles: a, an, the",
    "Ordinals",
    "Demonstrative pronouns: these and those",
    "Demonstrative pronouns: this, that, these, those"
]


def find_closest_matches(input_text, k=3):

    input_embedding = model.encode([input_text])
    topic_embeddings = model.encode(topics)
    similarity_scores = cosine_similarity(input_embedding, topic_embeddings)[0]

    sorted_and_filtered = list(filter(lambda s: s[0] > 0.5, sorted(zip(similarity_scores, topics), reverse=True)))

    if sorted_and_filtered[0][1].lower() == input_text.lower():
        closest_match_topics = sorted_and_filtered[:1]
    else:
        closest_match_topics = sorted_and_filtered[:k]

    closest_match_topics = [topic for _, topic in closest_match_topics]

    return closest_match_topics


@app.route('/book/generate', methods=['POST'])
def closest_matches():
    # obj = json.loads(request.json)
    print(request)
    input_topics = request.json["topics"]
    input_topics_list = input_topics.split(",")
    topics = []
    for topic in input_topics_list:
        closest_topics = find_closest_matches(topic)
        topics.extend(closest_topics)
        print(closest_topics)

    topics = list(set(topics))
    response = {
        "topics": topics
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
