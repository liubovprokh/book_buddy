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
    "Present simple (incl. vragen en ontkenningen)",
    "Questions in the present simple",
    "Future (toekomende tijd): will, going to, present continuous en present simple",
    "Present continuous",
    "Past simple (be/have)",
    "Past simple (onregelmatige werkwoorden)",
    "Past simple (verleden tijd)",
    "Past simple (verleden tijd) incl. vragen en ontkenningen",
    "Future (toekomende tijd): going to",
    "Can (kunnen)",
    "Can and could",
    "Modals: can / can't en could / couldn't",
    "Tag questions (korte vragen) - basis + can / can't",
    "Vragen (questions) met to do",
    "Vragen en ontkenningen met did",
    "Vragen met to be & have got",
    "Vragende voornaamwoorden",
    "Ontkenningen (negations) met to be (zijn)",
    "Gebiedende wijs (imperative)",
    "Persoonlijke voornaamwoorden (voorwerp) (object pronoun)",
    "Persoonlijke voornaamwoorden (onderwerp) en bezittelijke voornaamwoorden",
    "Persoonlijke voornaamwoorden",
    "Some and any (basis)",
    "Some and any",
    "Vragende voornaamwoorden",
    "Trappen van vergelijking: één, twee of meer lettergrepen",
    "Trappen van vergelijking: -er/-est, more / most en onregelmatig",
    "Trappen van vergelijking, incl. good/bad en (not) as ... as",
    "Lidwoorden (articles): a, an, the",
    "Ordinals (rangtelwoorden)",
    "Aanwijzende voornaamwoorden: these en those",
    "Aanwijzende voornaamwoorden (Demonstrative pronouns): this, that, these, those"
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
    input_text = request.json['input_text']

    closest_topics = find_closest_matches(input_text)

    response = {
        'closest_topics': closest_topics,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
