import nltk
nltk.data.path.append('/projectnb/skiran/Cassie/Code/nltk_data')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import yaml
from pathlib import Path
from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm")

# Core lexicon from Appendix C of Dalton et al. (2020)
config_path = Path(__file__).resolve().parent / "../config" / "story_config.yml"
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

def is_similar(word, ciu_vocab, threshold=0.80):
    synsets = wordnet.synsets(word)
    for ciu_word in ciu_vocab:
        ciu_synsets = wordnet.synsets(ciu_word)
        for s1 in synsets:
            for s2 in ciu_synsets:
                sim = s1.wup_similarity(s2)
                if sim and sim >= threshold:
                    # print(f"Word '{word}' is similar to CIU word '{ciu_word}' with similarity {sim}")
                    return True
    return False

def calculate_cinderella_ciu(transcript, duration_seconds, total_words_count, narrative_type):
    if narrative_type == 'cinderella':
        ciu_vocab = cfg["stories"][0]["ciu"]

    lemmatizer = WordNetLemmatizer()
    total_words = len(transcript)
    matched_cius = []

    for word in transcript:
        lemma = lemmatizer.lemmatize(word.lower())
        if lemma in ciu_vocab or is_similar(lemma, ciu_vocab):
            matched_cius.append(lemma)

    total_cius = len(matched_cius)
    duration_minutes = duration_seconds / 60

    return {
        "Cinderella_Total_CIUs": total_cius,
        "Cinderella_CIU_Ratio": round(total_cius / total_words_count, 2) if total_words_count > 0 else 0,
        "Cinderella_CIUs_per_Minute": round(total_cius / duration_minutes, 2) if duration_minutes > 0 else 0
    }

def get_ciu_nouns(narrative_type):
    return cfg["stories"][0]["ciu_nouns"] if narrative_type == 'cinderella' else []

def count_ciu_nouns(narrative_type, utterance):
    if narrative_type == 'cinderella':
        ciu_nouns = cfg["stories"][0]["ciu_nouns"]
    
    else:
        return 0

    lemmatizer = WordNetLemmatizer()
    matched_cius = []
    tokens = word_tokenize(utterance)
    for word in tokens:
        if word == 'like':
            return len(matched_cius), matched_cius
        lemma = lemmatizer.lemmatize(word.lower())
        if lemma in ciu_nouns or is_similar(lemma, ciu_nouns):
            matched_cius.append(lemma)
    return len(matched_cius), matched_cius

# print(count_ciu_nouns("cinderella", "Hi Cinderella came.."))
