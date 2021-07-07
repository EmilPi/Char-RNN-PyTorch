from collections import Counter

from vars import REPLACE_CHAR


def replace_unknown_chars(text, chars, replace_char=REPLACE_CHAR):
    text_chars = ''.join(set(text))
    for ch in text_chars:
        if ch not in chars:
            text = text.replace(ch, replace_char)
    return text


def replace_rare_chars(text, replace_char=REPLACE_CHAR):
    chars_counter = Counter(text)
    chars = [ch for ch, count in chars_counter.items() if count > 2]
    text = replace_unknown_chars(text, chars)
    chars += '\x01'
    chars = ''.join(sorted(chars))
    return text, chars
