import re
from features import get_stanza_pipeline

# Initialize Stanza pipeline for English
# stanza.download('en')
nlp = get_stanza_pipeline()

def has_main_clause(sentence):
    """Check if a sentence contains a main clause (subject + verb)."""
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_subject = False
        has_verb = False
        for word in sent.words:
            if word.deprel in ['nsubj', 'nsubj:pass']:
                has_subject = True
            if word.deprel == 'root' and word.upos in ['VERB', 'AUX']:
                has_verb = True
        if has_subject and has_verb:
            return True
    return False
 
def is_imperative(sentence):
    """Check if a sentence is an imperative (verb with no explicit subject)."""
    doc = nlp(sentence)
    for sent in doc.sentences:
        has_verb = False
        has_subject = False
        for word in sent.words:
            if word.deprel == 'root' and word.upos == 'VERB':
                has_verb = True
            if word.deprel in ['nsubj', 'nsubj:pass']:
                has_subject = True
        if has_verb and not has_subject:
            return True
    return False

def get_conjunctions(sentence):
    """Extract conjunctions (CCONJ or SCONJ) from the sentence using Stanza."""
    doc = nlp(sentence)
    conjunctions = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['CCONJ', 'SCONJ']:
                conjunctions.append(word.text.lower())
    return list(set(conjunctions))  # Remove duplicates


# [Previous functions: has_main_clause, is_imperative, get_conjunctions remain unchanged]

def segment_utterances(text):
    """Segment text into utterances, preserving repeated conjunctions."""
    # Define utterance terminators
    terminators = r'[.!?]'
    
    # Split text into potential utterances based on terminators
    raw_sentences = re.split(f'({terminators})', text)
    utterances = []
    current_utterance = ""
    pending_conjunction = ""
    
    # Get conjunctions for the entire text
    text_conjunctions = get_conjunctions(text)
    if not text_conjunctions:
        text_conjunctions = ['and', 'but', 'although', 'or', 'so']  # Fallback list
    
    # Create regex pattern for conjunctions (single or repeated), preserving them
    conj_pattern = r'\b(' + '|'.join(text_conjunctions) + r'(?:\s+' + '|'.join(text_conjunctions) + r')*)\b'
    
    for i in range(0, len(raw_sentences) - 1, 2):
        sentence = raw_sentences[i].strip()
        terminator = raw_sentences[i + 1]
        
        if not sentence:
            continue
            
        # Split by conjunctions, keeping them in the output
        parts = re.split(conj_pattern, sentence, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]  # Remove empty strings
        
        for j, part in enumerate(parts):
            # Skip if part is empty
            if not part:  
                continue
                
            # Check if the part is a conjunction (or repeated conjunctions)
            is_conjunction = bool(re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b', part, re.IGNORECASE))
            
            # Check for main clause or imperative
            has_main = has_main_clause(part)
            is_imp = is_imperative(part)
            
            if is_conjunction and not (has_main or is_imp):
                # Store the conjunction(s) to prepend to the next valid clause
                pending_conjunction = part
                continue
                
            if has_main or is_imp:
                # If there's a pending conjunction, prepend it
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""
                
                # If it follows a conjunction and is a main clause or imperative, start new utterance
                if j > 0 and bool(re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b', parts[j-1], re.IGNORECASE)) and (has_main or is_imp):
                    if current_utterance:
                        utterances.append(current_utterance + terminator)
                    current_utterance = part
                else:
                    # Combine with current utterance
                    current_utterance = (current_utterance + " " + part).strip()
            else:
                # Append to current utterance if it's a dependent clause/phrase
                if pending_conjunction:
                    part = f"{pending_conjunction} {part}".strip()
                    pending_conjunction = ""
                current_utterance = (current_utterance + " " + part).strip()
                
        # Append the current utterance with its terminator
        if current_utterance:
            utterances.append(current_utterance + terminator)
            current_utterance = ""
    
    # Handle any remaining utterance
    if current_utterance:
        if pending_conjunction:
            current_utterance = f"{pending_conjunction} {current_utterance}".strip()
        utterances.append(current_utterance + (terminator if terminator else '.'))
    
    # Final cleanup: merge standalone conjunctions
    final_utterances = []
    i = 0
    while i < len(utterances):
        utterance = utterances[i].strip()
        # Check if the utterance is just a conjunction (or repeated conjunctions)
        if re.fullmatch(r'\b(?:' + '|'.join(text_conjunctions) + r')(?:\s+(?:' + '|'.join(text_conjunctions) + r'))*\b[.!?]', utterance, re.IGNORECASE):
            if i + 1 < len(utterances):
                # Merge with the next utterance
                next_utterance = utterances[i + 1].strip()
                combined = f"{utterance[:-1]} {next_utterance}".strip()
                final_utterances.append(combined)
                i += 2
            else:
                # If it's the last utterance, append as is
                final_utterances.append(utterance)
                i += 1
        else:
            final_utterances.append(utterance)
            i += 1
    
    return final_utterances
# Sample dialogue
# dialogue = "There were 3 sisters, wasn't? Yeah, uh, 3 sisters and. Cinderella was uh. Cinderella was, she had uh she washed type of thing too and uh. I know what I wanna say, but I mean. Oh I know. She was invited to uh. To the ball type of thing too and uh Cinderella was and she had to go uh. Be there at 12 o'clock um I I. At 12 o'clock and uh. She had to be there at. She had to go, she had to, she had to go at 3 at 1 o'clock at 12 o'clock. And uh. Her, her godmother, uh. She had pumpkin. Her chariot was a pumpkin and then uh her her grandmother. Her godmother, uh, snapped or whatever. And uh she she uh had a chariot and she lost her slipper and then uh. He danced with her type of thing too and uh she lost her slipper and uh. He was um. I can't remember."

# # Segment the dialogue
# segmented = segment_utterances(dialogue)

# print("Segmented Utterances:", len(segmented))


# Output in CHAT-like format
# for i, utterance in enumerate(segmented, 1):
#     print(f"*PAR: {utterance} %utt{i}")