import re
# from .features import get_stanza_pipeline
from preprocessing import clean_text
from features import get_stanza_pipeline

# Initialize Stanza pipeline for English
# stanza.download('en')
nlp = get_stanza_pipeline()

def has_main_clause(sentence: str) -> bool:
    """
    Check if a sentence contains a main clause (explicit subject + main verb).
    CHAT Rule 1: utterance should (as much as possible) contain a main clause.
    """
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_subject = False
        has_verb_root = False
        for word in sent.words:
            if word.deprel in ("nsubj", "nsubj:pass"):
                has_subject = True
            if word.deprel == "root" and word.upos in ("VERB", "AUX"):
                has_verb_root = True
        if has_subject and has_verb_root:
            return True
    return False


# ----------------------------
# 2. Imperative detection
# ----------------------------
def is_imperative(sentence: str) -> bool:
    """
    Check if a sentence is an imperative: verb with no explicit subject.
    We keep this STRICT to avoid misclassifying things like
    'had decided to bring...' or 'in order to find him'.
    """
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_subject = False
        root_verb = None

        for word in sent.words:
            if word.deprel in ("nsubj", "nsubj:pass"):
                has_subject = True
            if word.deprel == "root" and word.upos == "VERB":
                root_verb = word

        if root_verb is None:
            continue
        if has_subject:
            continue

        # Prefer explicit morphological marking for imperatives.
        feats = root_verb.feats or ""
        if "Mood=Imp" in feats:
            return True

        # Optionally: very conservative fallback – comment OUT if you want
        # to be extra strict.
        #
        # # Fallback heuristic: verb is first content word (e.g., "Look here.")
        # content_words = [w for w in sent.words if w.upos not in ("PUNCT",)]
        # if content_words and content_words[0].id == root_verb.id:
        #     return True

    return False


# ----------------------------
# 3. Coordinating conjunctions ONLY (for splitting)
# ----------------------------
COORD_CONJ_LEXICON = {"and", "but", "or", "so", "yet", "nor"}

def get_coordinating_conjunctions(text: str):
    """
    Extract coordinating conjunctions (CCONJ) from the text.
    CHAT Rule 4: only conjoined CLAUSES via coordinating conjunctions
    are split into separate utterances.
    Subordinating conjunctions like 'if', 'because', 'although' should NOT
    trigger utterance splits.
    """
    doc = nlp(text)
    conjunctions = set()

    for sent in doc.sentences:
        for word in sent.words:
            # Prefer UPOS=CCONJ, but also back off to lexicon.
            if word.upos == "CCONJ" or word.text.lower() in COORD_CONJ_LEXICON:
                conjunctions.add(word.text.lower())

    if not conjunctions:
        # Fallback to common English coordinators only (no subordinators!)
        conjunctions = {"and", "but", "or", "so"}

    return list(conjunctions)


# ----------------------------
# 4. Utterance segmentation
# ----------------------------
def segment_utterances(text: str):
    """
    Segment text into utterances according to CHAT-like principles:

    1. Prefer utterances with a main clause (subject + verb).
    2. Allow subjectless imperatives as utterances.
    3. Keep main clause + its dependent clauses/adjuncts together.
    4. Split conjoined main clauses introduced by COORDINATING conjunctions
       (and, but, or, so...), but NOT subordinating ones (if, because...).
    5. Allow incomplete adult utterances (ellipsis, trailing off, etc.).
    """

    terminators = r"[.!?]"
    raw_sentences = re.split(f"({terminators})", text)

    utterances = []
    current_utterance = ""
    pending_conjunction = ""

    # Only use COORDINATING conjunctions for splitting.
    coord_conjs = get_coordinating_conjunctions(text)
    conj_pattern = (
        r"\b("
        + "|".join(map(re.escape, coord_conjs))
        + r"(?:\s+"
        + "|".join(map(re.escape, coord_conjs))
        + r")*)\b"
    )

    # Process in (sentence, punctuation) pairs
    for i in range(0, len(raw_sentences) - 1, 2):
        sentence = raw_sentences[i].strip()
        terminator = raw_sentences[i + 1]

        if not sentence:
            continue

        # Split by COORDINATING conjunctions, keep them in output.
        parts = re.split(conj_pattern, sentence, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p and p.strip()]

        for j, part in enumerate(parts):
            if not part:
                continue

            # Is this part just a coordinating conjunction (or a chain of them)?
            conj_only_pattern = (
                r"\b(?:"
                + "|".join(map(re.escape, coord_conjs))
                + r")(?:\s+(?:"
                + "|".join(map(re.escape, coord_conjs))
                + r"))*\b"
            )
            is_conjunction = bool(
                re.fullmatch(conj_only_pattern, part, flags=re.IGNORECASE)
            )

            has_main = has_main_clause(part)
            is_imp = is_imperative(part)

            # --- Case 1: pure conjunction, no clause yet ---
            if is_conjunction and not (has_main or is_imp):
                # Store to attach to next clause
                pending_conjunction = part
                continue

            # --- Case 2: main clause or imperative ---
            if has_main or is_imp:
                # Attach a pending 'and/but/or/so' if any
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""

                # If this follows a conjunction and has its own clause,
                # start a NEW utterance (Rule 4: conjoined clauses).
                if (
                    j > 0
                    and bool(
                        re.fullmatch(
                            conj_only_pattern, parts[j - 1], flags=re.IGNORECASE
                        )
                    )
                    and (has_main or is_imp)
                ):
                    if current_utterance:
                        utterances.append(current_utterance + terminator)
                    current_utterance = part
                else:
                    # Otherwise, extend the current utterance
                    current_utterance = (current_utterance + " " + part).strip()

            # --- Case 3: dependent clause / phrase (no main clause, no imperative) ---
            else:
                # This is a dependent clause, adjunct, or phrase.
                # Rule 3: keep it with the current main clause.
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""
                current_utterance = (current_utterance + " " + part).strip()

        # After finishing this punctuation-delimited sentence, close the utterance.
        if current_utterance:
            utterances.append(current_utterance + terminator)
            current_utterance = ""

    # Handle any leftover utterance fragment at the end
    if current_utterance:
        if pending_conjunction:
            current_utterance = f"{pending_conjunction} {current_utterance}".strip()
        utterances.append(current_utterance + (terminator if terminator else "."))

    # Final pass: merge standalone conjunction utterances like "And."
    final_utterances = []
    i = 0
    conj_only_with_punct_pattern = (
        r"\b(?:"
        + "|".join(map(re.escape, coord_conjs))
        + r")(?:\s+(?:"
        + "|".join(map(re.escape, coord_conjs))
        + r"))*\b[.!?]"
    )

    while i < len(utterances):
        utt = utterances[i].strip()

        if re.fullmatch(conj_only_with_punct_pattern, utt, flags=re.IGNORECASE):
            # Just 'And.' or 'But.' → merge with next utterance if possible
            if i + 1 < len(utterances):
                next_utt = utterances[i + 1].strip()
                combined = f"{utt[:-1]} {next_utt}".strip()  # drop punctuation on conj
                final_utterances.append(combined)
                i += 2
            else:
                final_utterances.append(utt)
                i += 1
        else:
            final_utterances.append(utt)
            i += 1

    return final_utterances
# Sample dialogue
dialogue = "As I remember the story, Cinderella came to live with the, the, the mean old lady and her two daughters, and um she, they, she became the housemaid, if you will, because they, for whatever reason, they didn't want her around. And um she cleaned the house and did all the stuff, and one day found out about a ball that was happening, and wished that she could go, but um she wasn't terribly Welcome there. So she, um, I don't even remember the story. She finally did get to go. Her fairy godmother came by, tapped her on the head, she turned her into a little princess, and she went to the, to the ball, met the, the prince, um, realized that she was going to turn back at midnight, so she ran out, dropped her slipper, uh, her was in the silver slipper, but she dropped her slipper, and the prince was so enamored with her, he sent people out to find her, they found her. He tried on the shoe. First, the, the bad sisters wanted the shoe, none of them fit. She tried it on, it fit. He fell in love and they lived happily ever after. That's the short version."
cleaned_dialogue = clean_text(dialogue)
# Segment the dialogue
segmented = segment_utterances(cleaned_dialogue)

print("Segmented Utterances:", len(segmented))
print("-------------------------")

# Output in CHAT-like format
for i, utterance in enumerate(segmented, 1):
    # print(f"*PAR: {utterance} %utt{i}")
    print(i, " ", utterance)