"""Translate Chinese (Mandarin) SoulX-Singer metadata to English using Qwen LLM.

Reads metadata.json, translates text, generates English phonemes, writes output.

Usage:
    python qwen_translate.py --metadata example/transcriptions/music/metadata.json \
                             --output example/transcriptions/music/metadata_en.json
"""

import argparse
import json
import os
import re

from openai import OpenAI
from g2p_en import G2p

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

g2p = G2p()


def word_to_phoneme(word):
    """Convert an English word to SoulX-Singer phoneme format: en_PH1-PH2-..."""
    phonemes = g2p(word.lower())
    # Keep only actual phoneme tokens (uppercase letters, optional trailing digit)
    phonemes = [p for p in phonemes if re.match(r'^[A-Z]+\d?$', p)]
    if not phonemes:
        return "<UNK>"
    return "en_" + "-".join(phonemes)


def extract_phrases(text_tokens, note_types):
    """Extract phrases (contiguous non-<SP> token groups) with their positions."""
    phrases = []
    i = 0
    n = len(text_tokens)
    while i < n:
        if text_tokens[i] != "<SP>":
            start = i
            ptokens, ptypes = [], []
            while i < n and text_tokens[i] != "<SP>":
                ptokens.append(text_tokens[i])
                ptypes.append(note_types[i])
                i += 1
            phrases.append((start, ptokens, ptypes))
        else:
            i += 1
    return phrases


def build_prompt(phrase_info_list):
    """Build the LLM translation prompt."""
    phrase_lines = []
    for i, info in enumerate(phrase_info_list):
        phrase_lines.append(
            f'  Phrase {i+1} (EXACTLY {info["word_count"]} words): "{info["chinese"]}"'
        )

    output_lines = "\n".join(
        f'Phrase {i+1}: <exactly {info["word_count"]} English words>'
        for i, info in enumerate(phrase_info_list)
    )

    return f"""You are a professional song lyric translator (Chinese Mandarin to English).

Translate each Chinese phrase into natural, singable English.

CRITICAL RULES:
- Each phrase MUST have EXACTLY the specified number of English words.
- Count contractions like "you're" or "can't" as ONE word.
- Do NOT include any punctuation.
- Keep translations natural, poetic, and suitable for singing.
- Maintain the emotional meaning of the original lyrics.

Chinese phrases to translate:
{chr(10).join(phrase_lines)}

Output ONLY the translations, one per line, in this exact format:
{output_lines}"""


def parse_response(response, num_phrases):
    """Parse LLM response into per-phrase word lists."""
    translations = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        m = re.match(r'Phrase\s+(\d+)\s*[:：]\s*(.+)', line)
        if m:
            idx = int(m.group(1)) - 1
            text = m.group(2).strip()
            # Remove any stray punctuation the LLM might have added
            text = re.sub(r'[^\w\s\'-]', '', text).strip()
            words = text.split()
            if 0 <= idx < num_phrases:
                translations[idx] = words
    return translations


def translate_segment(seg):
    """Translate one metadata segment from Chinese to English."""
    text_tokens = seg["text"].split()
    note_types = list(map(int, seg["note_type"].split()))

    phrases = extract_phrases(text_tokens, note_types)

    # Build per-phrase info
    phrase_info_list = []
    for start, ptokens, ptypes in phrases:
        # Unique word slots = non-tied note positions (note_type != 3)
        unique_count = sum(1 for t in ptypes if t != 3)
        # Clean Chinese text: only the unique (non-tied) characters
        clean_chars = [tok for tok, typ in zip(ptokens, ptypes) if typ != 3]
        phrase_info_list.append({
            'start': start,
            'tokens': ptokens,
            'types': ptypes,
            'chinese': "".join(clean_chars),
            'word_count': unique_count,
        })

    prompt = build_prompt(phrase_info_list)

    print(f"\n{'='*60}")
    print(f"Segment: {seg['index']}")
    print(f"Original: {seg['text']}")
    print(f"{'='*60}")
    print(f"Prompt:\n{prompt}\n")

    # Call Qwen LLM
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
        stream=True,
    )

    response = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            response += delta.content

    print(f"LLM response:\n{response}\n")

    translations = parse_response(response, len(phrase_info_list))

    # Apply translations to text and phoneme tokens
    new_text = list(text_tokens)
    new_phoneme = seg["phoneme"].split()

    for pidx, info in enumerate(phrase_info_list):
        if pidx not in translations:
            print(f"  WARNING: No translation for phrase {pidx+1}, keeping original")
            continue

        en_words = translations[pidx]
        expected = info['word_count']

        if len(en_words) != expected:
            print(f"  WARNING: Phrase {pidx+1} needs {expected} words, got {len(en_words)}: {en_words}")
            if len(en_words) < expected:
                en_words += [en_words[-1]] * (expected - len(en_words))
            else:
                en_words = en_words[:expected]

        print(f"  Phrase {pidx+1}: {' '.join(en_words)} ({len(en_words)} words)")

        word_idx = 0
        start = info['start']
        for j, (tok, ntype) in enumerate(zip(info['tokens'], info['types'])):
            pos = start + j
            if ntype == 3:
                # Tied note: repeat previous word and phoneme
                new_text[pos] = new_text[pos - 1]
                new_phoneme[pos] = new_phoneme[pos - 1]
            else:
                w = en_words[word_idx]
                new_text[pos] = w
                new_phoneme[pos] = word_to_phoneme(w)
                word_idx += 1

    seg["text"] = " ".join(new_text)
    seg["phoneme"] = " ".join(new_phoneme)
    seg["language"] = "English"
    return seg


def main():
    parser = argparse.ArgumentParser(
        description="Translate Chinese SoulX-Singer metadata to English"
    )
    parser.add_argument("--metadata", required=True, help="Input metadata.json path")
    parser.add_argument("--output", required=True, help="Output translated metadata path")
    args = parser.parse_args()

    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} segment(s) from {args.metadata}")

    for seg in metadata:
        translate_segment(seg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Translated metadata saved to {args.output}")


if __name__ == "__main__":
    main()
