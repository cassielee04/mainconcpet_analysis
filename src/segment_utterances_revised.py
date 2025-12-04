import re
# from .features import get_stanza_pipeline
from preprocessing import clean_text
from features import get_stanza_pipeline
from nouns_verbs_per_utterance import calculate_nouns_per_utterance, calculate_verbs_per_utterance

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
dialogue = "Once upon a time, there was a man who lost his wife and had decided to bring in a new wife who happened to have two children of her own. When they were still small, he introduced his daughter to his new wife, who would then be her stepmother and her two new stepsisters. As they aged and grew up, Cinderella became less of a daughter and more of just a maid working for. The stepmother and the stepdaughters. Apparently papa had died and could no longer See to the welfare of his own child. As they continued to Move on in life The stepmother. Kept Cinderella. In the dark, or in the basement, or at least not in keeping with the rest of the household. Cinderella made friends with the mice and the dogs and the horses that were part of her milieu. And had become her real family. Even the birds that sang to her would come in and keep her company. One day The king decided his son needed to marry and so proclaimed that the prince would have to find a wife and would throw a major Ball in order for the prince to find an acceptable. Me. Everybody in the Kingdom was sent an invitation to this ball. But Cinderella's stepmother. was not going to allow Cinderella even a dress in which to accompany the rest of the family. Cinderella found a book in a chest apparently left by her parents, and was able to make a dress with the help of her animal friends. And put together a lovely gown, and when it was time to leave for the ball, her stepsisters and stepmother. tore the dress apart and refused to allow Cinderella to accompany them. But off they went anyway. In her Despair Cinderella is introduced to her fairy godmother, someone that she had never known existed before. And was given an opportunity to attend the ball in a beautiful gown. Pulled by Marvelous white horses, which were enchanted mice who were her friends, and a pumpkin that was made into a beautiful carriage, and a dog that was made into a footman so that she could. Arrive at the ball in true style. Once at the bally. was the prince's only Attraction. He paid no attention to anyone but Cinderella, but at the stroke of midnight, as her Godmother had said, She had to leave the ball so as not to Disturbed the enchantment under the eyes of any onlookers. She arrived back home just in time, but she had lost her glass slipper on the steps of the palace. In the next days. The prince, having found the shoe. Went looking for the person whose foot would fit that shoe, and after having traversed the entire kingdom looking for the one woman whose foot would fit in that shoe, he comes across the stepsisters and stepmother. And at last, Cinderella is allowed to try on the shoe, which fits perfectly. And leads to happily ever after."
cleaned_dialogue = clean_text(dialogue)
# Segment the dialogue
segmented = segment_utterances(cleaned_dialogue)

print("Segmented Utterances:", len(segmented))
print("-------------------------")
print("verbs per utterance:", calculate_verbs_per_utterance(segmented, nlp))

# Output in CHAT-like format
for i, utterance in enumerate(segmented, 1):
    # print(f"*PAR: {utterance} %utt{i}")
    print(i, " ", utterance)