# ==================================================
# INSTALLATION
# ==================================================
#
# Required:
# pip install numpy scikit-learn
#
# Optional embedding backend:
# pip install sentence-transformers
#
# All policies and source labels below are fictional.
# ==================================================

import csv
import json
import re
import time

from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# ==================================================
# CONFIGURATION
# ==================================================

OUTPUT_DIRECTORY = Path("retrieval_results")

ENABLE_EMBEDDINGS = False

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 48

OVERLAP_SIZE = 16

TOP_K = 3


# ==================================================
# NEW SYNTHETIC DATASET
# ==================================================

POLICIES = {

    "library": (
        "Students may borrow four library books for fourteen days. "
        "Renewals are allowed only when no other reader has reserved the book."
    ),

    "wifi": (
        "Connect to CampusNet using your university username and password. "
        "If access fails, forget the saved network and sign in again."
    ),

    "laboratory": (
        "Students must complete safety training before entering the laboratory. "
        "A supervisor must approve any laboratory access outside scheduled hours."
    ),

    "leave": (
        "An absence request must be submitted through the student portal. "
        "Medical absence requests must include a certificate from a clinician."
    ),

    "staff": (
        "Permanent staff may request remote work through their department manager. "
        "Approval must be recorded before remote work begins."
    ),

    "consultant": (
        "External consultants may work remotely only when their service agreement "
        "explicitly permits it. Permanent-staff approval does not authorize consultants."
    ),

}


SOURCE_NAMES = {

    key: f"fictional_{key}_guide.txt"

    for key in POLICIES

}


def build_documents():

    documents = {}

    introduction = (
        "This fictional guide describes one administrative category. "
        "Readers should check the named category before applying a rule."
    )

    background = (
        "Administrative reviews record routine requests and maintain "
        "clear responsibilities for the people involved."
    )

    closing = (
        "This training material is not an official university policy. "
        "Questions outside its scope should be referred to a human adviser."
    )

    long_sentence = (
        "Administrative "
        + " ".join(["review"] * 60)
        + "."
    )

    for category, policy in POLICIES.items():

        documents[category] = " ".join(

            [
                introduction,

                background,

                policy,

                long_sentence,

                closing,

                closing,

                closing,

            ]

        )

    return documents


DOCUMENTS = build_documents()


QUESTION_DATA = [

    (
        "How many library books may a student borrow?",
        "library",
        "dev"
    ),

    (
        "How do I connect to CampusNet?",
        "wifi",
        "dev"
    ),

    (
        "What training is required before laboratory entry?",
        "laboratory",
        "dev"
    ),

    (
        "Can an external consultant work remotely?",
        "consultant",
        "dev"
    ),

    (
        "How long can I keep borrowed books?",
        "library",
        "test"
    ),

    (
        "The campus wireless network will not connect. What can I try?",
        "wifi",
        "test"
    ),

    (
        "Who approves laboratory access outside scheduled hours?",
        "laboratory",
        "test"
    ),

    (
        "What document supports a medical absence request?",
        "leave",
        "test"
    ),

    (
        "Who approves remote work for permanent staff?",
        "staff",
        "test"
    ),

    (
        "Does staff approval allow a consultant to work remotely?",
        "consultant",
        "test"
    ),

    (
        "What will the weather be next month?",
        None,
        "test"
    ),

    (
        "Which team won yesterday's cricket match?",
        None,
        "test"
    ),

]


QUERIES = [

    {
        "query": question,

        "doc": category,

        "evidence": POLICIES.get(category),

        "split": split,

    }

    for question, category, split in QUESTION_DATA

]


FAQS = {

    "How many books can I borrow?": {

        "answer": "Students may borrow four books.",

        "source": SOURCE_NAMES["library"],

    },

    "How do I connect to campus Wi-Fi?": {

        "answer": "Use CampusNet with your university credentials.",

        "source": SOURCE_NAMES["wifi"],

    },

    "What is required before laboratory entry?": {

        "answer": "Complete safety training before entering.",

        "source": SOURCE_NAMES["laboratory"],

    },

    "What evidence supports medical absence?": {

        "answer": "Include a certificate from a clinician.",

        "source": SOURCE_NAMES["leave"],

    },

}


# ==================================================
# FILE AND DISPLAY HELPERS
# ==================================================

def heading(title):

    print("\n" + "=" * 72)

    print(f"\n{title}\n")


def write_json(filename, data):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = OUTPUT_DIRECTORY / filename

    destination.write_text(

        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"

    )


def write_csv(filename, records):

    if not records:

        return

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = OUTPUT_DIRECTORY / filename

    with destination.open(

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(records[0])

        )

        writer.writeheader()

        writer.writerows(records)


# ==================================================
# TASK 1: TOKENIZATION
# ==================================================

def tokenization_demo(model=None):

    heading("TASK 1: TOKENIZATION COMPARISON")

    examples = [

        "Campus Wi-Fi: available!",

        "Students shouldn't share passwords.",

        "support@campus.example",

        "LAB-204",

        "",

        "مجھے لائبریری کی معلومات چاہییں",

    ]

    for text in examples:

        whitespace_tokens = text.split()

        regex_tokens = re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE
        )

        print(f"Input: {text!r}\n")

        print(
            f"Whitespace tokens: {whitespace_tokens}"
        )

        print(
            f"Whitespace count: {len(whitespace_tokens)}\n"
        )

        print(f"Regex tokens: {regex_tokens}")

        print(f"Regex count: {len(regex_tokens)}\n")

        if model is not None:

            model_tokens = model.tokenizer.tokenize(text)

            print(f"Model tokens: {model_tokens}")

            print(f"Model token count: {len(model_tokens)}\n")

        print("-" * 60 + "\n")


# ==================================================
# TASK 1: FAQ MATCHING
# ==================================================

class FAQMatcher:

    def __init__(self, faqs):

        self.faqs = faqs

        self.questions = list(faqs)

        self.normalized_questions = {

            self.normalize(question): question

            for question in self.questions

        }

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        self.matrix = self.vectorizer.fit_transform(
            self.questions
        )

    @staticmethod
    def normalize(text):

        return " ".join(text.casefold().split())

    def match(self, query):

        normalized = self.normalize(query)

        if not normalized:

            return {

                "status": "empty",

                "message": "Please enter a question.",

            }

        exact_question = self.normalized_questions.get(
            normalized
        )

        if exact_question is not None:

            return {

                "status": "exact",

                **self.faqs[exact_question],

            }

        query_vector = self.vectorizer.transform([query])

        scores = (
            query_vector @ self.matrix.T
        ).toarray()[0]

        ranked = np.argsort(
            -scores,
            kind="stable"
        )

        candidates = [

            {
                "question": self.questions[index],

                "score": float(scores[index]),

            }

            for index in ranked[:TOP_K]

            if scores[index] > 0

        ]

        return {

            "status": "confirm" if candidates else "unmatched",

            "candidates": candidates,

        }


def faq_demo():

    heading("TASK 1: FAQ MATCHING")

    matcher = FAQMatcher(FAQS)

    examples = [

        "How many books can I borrow?",

        "  HOW MANY BOOKS CAN I BORROW?  ",

        "medical absence certificate",

        "",

        "ocean volcano geology",

    ]

    for query in examples:

        print(f"Question: {query!r}\n")

        print(

            json.dumps(

                matcher.match(query),

                ensure_ascii=False,

                indent=2

            )

        )

        print("\n" + "-" * 60 + "\n")

    print(
        "Nonexact matches are suggestions requiring confirmation, "
        "not automatically approved answers."
    )


# ==================================================
# TASK 2: CHUNKING WITH CHARACTER OFFSETS
# ==================================================

def word_windows(text, size=CHUNK_SIZE, overlap=0, offset=0):

    if size <= 0:

        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= size:

        raise ValueError(
            "Overlap must be between zero and size minus one."
        )

    words = list(re.finditer(r"\S+", text))

    spans = []

    step = size - overlap

    for start_index in range(0, len(words), step):

        stop_index = min(
            start_index + size,
            len(words)
        )

        spans.append(

            (
                offset + words[start_index].start(),

                offset + words[stop_index - 1].end(),

            )

        )

        if stop_index == len(words):

            break

    return spans


def sentence_chunks(text, size=CHUNK_SIZE):

    if size <= 0:

        raise ValueError("Sentence budget must be positive.")

    spans = []

    pending_start = None

    pending_end = None

    pending_words = 0

    sentence_pattern = r"\S.*?(?:[.!?](?=\s|$)|$)"

    matches = re.finditer(
        sentence_pattern,
        text,
        flags=re.DOTALL
    )

    for match in matches:

        start, end = match.span()

        word_count = len(match.group().split())

        if word_count > size:

            if pending_start is not None:

                spans.append(
                    (pending_start, pending_end)
                )

            spans.extend(

                word_windows(

                    match.group(),

                    size=size,

                    offset=start

                )

            )

            pending_start = None

            pending_end = None

            pending_words = 0

            continue

        if (
            pending_start is not None
            and pending_words + word_count <= size
        ):

            pending_end = end

            pending_words += word_count

            continue

        if pending_start is not None:

            spans.append(
                (pending_start, pending_end)
            )

        pending_start = start

        pending_end = end

        pending_words = word_count

    if pending_start is not None:

        spans.append(
            (pending_start, pending_end)
        )

    return spans


def build_chunks(documents, strategy):

    strategies = {

        "fixed": lambda text: word_windows(text),

        "overlap": lambda text: word_windows(

            text,

            overlap=OVERLAP_SIZE

        ),

        "sentence": sentence_chunks,

    }

    if strategy not in strategies:

        raise ValueError(f"Unknown strategy: {strategy}")

    records = []

    for category, text in documents.items():

        spans = strategies[strategy](text)

        for index, (start, end) in enumerate(spans):

            records.append(

                {
                    "id": f"{category}-{strategy}-{index}",

                    "doc": category,

                    "source": SOURCE_NAMES[category],

                    "start": start,

                    "end": end,

                    "text": text[start:end],

                }

            )

    return records


def validate_chunks(records):

    for record in records:

        original = DOCUMENTS[record["doc"]]

        assert record["text"] == original[
            record["start"]:record["end"]
        ]

        assert len(record["text"].split()) <= CHUNK_SIZE


def chunking_tests():

    assert word_windows("") == []

    assert sentence_chunks("") == []

    assert word_windows("small text") == [(0, 10)]

    assert sentence_chunks("small text") == [(0, 10)]

    long_text = " ".join(["word"] * 110) + "."

    assert len(sentence_chunks(long_text)) == 3

    try:

        word_windows(
            "one two",
            size=2,
            overlap=2
        )

    except ValueError:

        pass

    else:

        raise AssertionError("Invalid overlap was accepted.")


# ==================================================
# OPTIONAL EMBEDDING MODEL
# ==================================================

def load_embedding_model():

    if not ENABLE_EMBEDDINGS:

        print("Embedding comparison disabled; using TF-IDF.")

        return None

    started = time.perf_counter()

    try:

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(

            EMBEDDING_MODEL,

            device="cpu"

        )

    except (ImportError, OSError, RuntimeError) as error:

        print(f"Embedding backend could not load: {error}")

        print("Continuing with TF-IDF only.")

        return None

    elapsed = time.perf_counter() - started

    print(f"Embedding model load time: {elapsed:.3f} seconds")

    return model


def audit_model_tokens(model, texts):

    if model is None:

        return None

    longest = 0

    for text in texts:

        encoded = model.tokenizer(
            text,
            truncation=False
        )

        token_count = len(encoded["input_ids"])

        longest = max(longest, token_count)

        if token_count > model.max_seq_length:

            raise ValueError(
                "Input exceeds the embedding model token limit."
            )

    return longest


# ==================================================
# TASK 3: REUSABLE RETRIEVAL INDEX
# ==================================================

class RetrievalIndex:

    def __init__(self, records, backend="tfidf", model=None):

        self.records = records

        self.backend = backend

        self.model = model

        texts = [
            record["text"]
            for record in records
        ]

        started = time.perf_counter()

        if backend == "tfidf":

            self.vectorizer = TfidfVectorizer(

                ngram_range=(1, 2),

                sublinear_tf=True

            )

            self.matrix = self.vectorizer.fit_transform(texts)

        elif backend == "embedding":

            if model is None:

                raise ValueError("No embedding model is available.")

            audit_model_tokens(model, texts)

            self.matrix = model.encode(

                texts,

                normalize_embeddings=True,

                show_progress_bar=False

            )

        else:

            raise ValueError(f"Unknown backend: {backend}")

        self.index_seconds = time.perf_counter() - started

    def search(self, queries):

        started = time.perf_counter()

        if self.backend == "tfidf":

            query_vectors = self.vectorizer.transform(queries)

        else:

            audit_model_tokens(self.model, queries)

            query_vectors = self.model.encode(

                queries,

                normalize_embeddings=True,

                show_progress_bar=False

            )

        encoding_seconds = time.perf_counter() - started

        started = time.perf_counter()

        similarities = query_vectors @ self.matrix.T

        if self.backend == "tfidf":

            similarities = similarities.toarray()

        search_seconds = time.perf_counter() - started

        return (

            similarities,

            encoding_seconds,

            search_seconds,

        )


# ==================================================
# RETRIEVAL EVALUATION
# ==================================================

def evaluate(index, questions, strategy, split):

    query_texts = [

        item["query"]

        for item in questions

    ]

    scores, encoding_seconds, search_seconds = index.search(
        query_texts
    )

    hit_values = []

    reciprocal_ranks = []

    logs = []

    unsupported_count = 0

    for question, row in zip(questions, scores):

        ranked_indices = np.argsort(
            -row,
            kind="stable"
        )[:TOP_K]

        first_relevant_rank = None

        if question["doc"] is None:

            unsupported_count += 1

        for rank, index_number in enumerate(
            ranked_indices,
            start=1
        ):

            record = index.records[index_number]

            relevant = bool(

                question["doc"] == record["doc"]

                and question["evidence"]

                and question["evidence"] in record["text"]

            )

            if relevant and first_relevant_rank is None:

                first_relevant_rank = rank

            logs.append(

                {
                    "strategy": strategy,

                    "backend": index.backend,

                    "split": split,

                    "query": question["query"],

                    "rank": rank,

                    "score": float(row[index_number]),

                    "relevant": relevant,

                    **record,

                }

            )

        if question["doc"] is not None:

            hit_values.append(
                int(first_relevant_rank is not None)
            )

            reciprocal_ranks.append(

                1 / first_relevant_rank

                if first_relevant_rank is not None

                else 0.0

            )

    metrics = {

        "strategy": strategy,

        "backend": index.backend,

        "split": split,

        "hit_at_3": (
            float(np.mean(hit_values))
            if hit_values
            else None
        ),

        "mrr_at_3": (
            float(np.mean(reciprocal_ranks))
            if reciprocal_ranks
            else None
        ),

        "answerable_queries": len(hit_values),

        "unsupported_queries": unsupported_count,

        "chunks": len(index.records),

        "index_seconds": index.index_seconds,

        "query_encoding_seconds": encoding_seconds,

        "search_seconds": search_seconds,

    }

    return metrics, logs


# ==================================================
# EVIDENCE-ONLY CHATBOT
# ==================================================

def start_chat(index):

    heading("INTERACTIVE EVIDENCE BROWSER")

    print("All documents are fictional training material.")

    print("Retrieved passages are candidates, not verified answers.")

    print("Type 'exit' to stop.\n")

    while True:

        try:

            question = input("Question: ").strip()

            if question.casefold() in {"exit", "quit"}:

                break

            if not question:

                print("\nPlease enter a question.\n")

                continue

            scores, _, _ = index.search([question])

            ranked_indices = np.argsort(

                -scores[0],

                kind="stable"

            )[:TOP_K]

            for number, record_index in enumerate(
                ranked_indices,
                start=1
            ):

                record = index.records[record_index]

                print(f"\nCandidate {number}")

                print(f"Source: {record['source']}")

                print(
                    f"Similarity: "
                    f"{float(scores[0, record_index]):.4f}"
                )

                print(
                    f"Characters: "
                    f"{record['start']}–{record['end']}"
                )

                print(f"Preview: {record['text'][:220]}...\n")

            selection = input(

                "Choose a candidate number to inspect, "

                "or press Enter if none apply: "

            ).strip()

            allowed = {

                str(number)

                for number in range(
                    1,
                    len(ranked_indices) + 1
                )

            }

            if selection not in allowed:

                print(

                    "\nNo evidence selected. "

                    "Rephrase or contact a human adviser.\n"

                )

                continue

            selected_index = ranked_indices[
                int(selection) - 1
            ]

            record = index.records[selected_index]

            print("\nCandidate evidence only:\n")

            print(record["text"])

            print(f"\nSource: {record['source']}")

            print(
                f"Character range: "
                f"{record['start']}–{record['end']}\n"
            )

            print(
                "Selecting a passage does not verify that "
                "it answers the question.\n"
            )

        except (EOFError, KeyboardInterrupt):

            print("\nChat stopped.")

            break

        except ValueError as error:

            print(f"\nInput error: {error}\n")


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    write_json("documents.json", DOCUMENTS)

    write_json("questions.json", QUERIES)

    model = load_embedding_model()

    tokenization_demo(model)

    faq_demo()

    chunking_tests()

    heading("TASK 2: CHUNKING COMPARISON")

    chunk_sets = {}

    for strategy in ("fixed", "overlap", "sentence"):

        records = build_chunks(
            DOCUMENTS,
            strategy
        )

        validate_chunks(records)

        chunk_sets[strategy] = records

        sizes = [
            len(record["text"].split())
            for record in records
        ]

        original_words = sum(

            len(text.split())

            for text in DOCUMENTS.values()

        )

        print(f"Strategy: {strategy}\n")

        print(f"Chunk count: {len(records)}\n")

        print(f"Chunk-size distribution: {dict(Counter(sizes))}\n")

        print(
            f"Extra word occurrences: "
            f"{sum(sizes) - original_words}\n"
        )

        print(
            "Maximum model token count:",
            audit_model_tokens(
                model,
                [record["text"] for record in records]
            ),
            "\n"
        )

        for category, evidence in POLICIES.items():

            coverage = any(

                record["doc"] == category

                and evidence in record["text"]

                for record in records

            )

            print(
                f"Complete evidence coverage "
                f"for {category}: {coverage}"
            )

        print("\n" + "-" * 60 + "\n")

    development_questions = [

        question

        for question in QUERIES

        if question["split"] == "dev"

    ]

    test_questions = [

        question

        for question in QUERIES

        if question["split"] == "test"

    ]

    backends = ["tfidf"]

    if model is not None:

        backends.append("embedding")

    indexes = {}

    all_metrics = []

    all_logs = []

    heading("TASK 3: DEVELOPMENT EVALUATION")

    for strategy, records in chunk_sets.items():

        for backend in backends:

            index = RetrievalIndex(
                records,
                backend,
                model
            )

            indexes[(strategy, backend)] = index

            metrics, logs = evaluate(

                index,

                development_questions,

                strategy,

                "dev"

            )

            all_metrics.append(metrics)

            all_logs.extend(logs)

            print(json.dumps(metrics, indent=2))

            print()

    selected = max(

        all_metrics,

        key=lambda item: (

            item["mrr_at_3"],

            item["hit_at_3"],

            -item["chunks"],

        )

    ).copy()

    # Selection is saved before any test evaluation.

    write_json(
        "selected_configuration.json",
        selected
    )

    heading("TEST EVALUATION")

    for (strategy, backend), index in indexes.items():

        metrics, logs = evaluate(

            index,

            test_questions,

            strategy,

            "test"

        )

        all_metrics.append(metrics)

        all_logs.extend(logs)

        print(json.dumps(metrics, indent=2))

        print()

    write_json("metrics.json", all_metrics)

    write_csv("retrieval_log.csv", all_logs)

    write_json("chunks.json", chunk_sets)

    heading("SELECTED CONFIGURATION")

    print(json.dumps(selected, indent=2))

    heading("TOP-RANKED NONRELEVANT EXAMPLES")

    problematic = [

        record

        for record in all_logs

        if record["rank"] == 1

        and not record["relevant"]

    ]

    for record in problematic[:3]:

        print(f"Query: {record['query']}")

        print(f"Retrieved source: {record['source']}")

        print(f"Retrieved text: {record['text']}\n")

    print(

        "Evaluation note: Hit@3 and MRR@3 use only answerable "

        "queries. Unsupported queries remain in the retrieval log. "

        "Relevance requires the complete supplied evidence passage "

        "inside one retrieved chunk.\n"

    )

    selected_index = indexes[

        (
            selected["strategy"],

            selected["backend"],

        )

    ]

    start_chat(selected_index)


if __name__ == "__main__":

    main()