
from nltk import pos_tag, word_tokenize


def pos_tag_sentence(sentence):
    sen_tokens = word_tokenize(sentence)
    sen_pos_tags = pos_tag(sen_tokens)
    pos_tags_only = [pos_tag_pair[1] for pos_tag_pair in sen_pos_tags]
    return " ".join(pos_tags_only)
