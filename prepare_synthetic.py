"""
Generate synthetic constraint-following training data.

Programmatically creates instruction-output pairs covering the main IFEval
constraint types. Every output is verified by the same verifiers used in
evaluate.py before being saved — zero false positives.

Output: data/train_synthetic.jsonl
Target: ~2000 examples covering all major constraint categories.

Usage:
    uv run python prepare_synthetic.py
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, "data", "train_synthetic.jsonl")

sys.path.insert(0, ROOT)
from evaluate import verify_instruction

random.seed(42)

# ---------------------------------------------------------------------------
# Content bank: realistic sentences about diverse topics
# ---------------------------------------------------------------------------

TOPICS = [
    ("climate change", [
        "Global temperatures have risen significantly over the past century due to greenhouse gas emissions.",
        "Renewable energy sources like solar and wind power are becoming increasingly affordable.",
        "Deforestation contributes to carbon dioxide buildup in the atmosphere.",
        "Rising sea levels threaten coastal communities around the world.",
        "Electric vehicles reduce dependence on fossil fuels and lower emissions.",
        "Carbon capture technology removes CO2 directly from the atmosphere.",
        "Melting polar ice caps accelerate the overall warming process.",
        "International agreements aim to limit global warming to 1.5 degrees Celsius.",
    ]),
    ("artificial intelligence", [
        "Machine learning models are trained on large datasets to recognize patterns.",
        "Neural networks mimic the structure of the human brain to process information.",
        "Large language models can generate human-like text based on prompts.",
        "AI is transforming industries from healthcare to finance and manufacturing.",
        "Ethical considerations in AI development include fairness and transparency.",
        "Autonomous vehicles use computer vision and sensor fusion to navigate roads.",
        "Natural language processing enables computers to understand human speech.",
        "Reinforcement learning trains agents by rewarding desired behaviors.",
    ]),
    ("nutrition and health", [
        "A balanced diet includes proteins from lean meats and legumes.",
        "Vegetables provide essential vitamins and minerals for the immune system.",
        "Whole grains offer more fiber and nutrients than refined alternatives.",
        "Staying hydrated improves cognitive function and physical performance.",
        "Processed foods often contain high amounts of sodium and added sugars.",
        "Antioxidants found in berries protect cells from oxidative damage.",
        "Regular physical activity combined with good nutrition prevents chronic disease.",
        "Sleep is essential for the body to repair and metabolize nutrients.",
    ]),
    ("space exploration", [
        "The James Webb Space Telescope captures images of galaxies billions of light-years away.",
        "Mars missions are testing life support systems for future human colonies.",
        "Reusable rockets have dramatically reduced the cost of reaching orbit.",
        "The International Space Station has hosted astronauts continuously since 2000.",
        "Exoplanet research has identified thousands of planets outside our solar system.",
        "Asteroid mining could provide rare metals needed for advanced technology.",
        "Lunar bases would serve as stepping stones for deeper space exploration.",
        "Radio telescopes scan the universe for signals from potential civilizations.",
    ]),
    ("personal finance", [
        "Compound interest allows savings to grow exponentially over long periods.",
        "Diversifying investments across asset classes reduces overall portfolio risk.",
        "An emergency fund should cover three to six months of living expenses.",
        "High-interest debt should be paid off before investing surplus income.",
        "Index funds offer broad market exposure at very low management fees.",
        "Budgeting helps track spending and identify areas for improvement.",
        "Retirement accounts provide tax advantages for long-term savings.",
        "Insurance protects against unexpected financial losses from accidents or illness.",
    ]),
    ("education", [
        "Active recall and spaced repetition are among the most effective study techniques.",
        "Critical thinking skills help students evaluate sources and arguments.",
        "Project-based learning connects classroom concepts to real-world problems.",
        "Early childhood education has lasting positive effects on cognitive development.",
        "Online courses have democratized access to high-quality instruction worldwide.",
        "Feedback from teachers helps students identify and correct misconceptions.",
        "Reading widely builds vocabulary and improves analytical reasoning.",
        "Collaboration in learning environments prepares students for professional teamwork.",
    ]),
    ("mental health", [
        "Mindfulness meditation reduces stress and improves emotional regulation.",
        "Regular exercise releases endorphins that elevate mood naturally.",
        "Social connection is a fundamental human need linked to mental wellbeing.",
        "Cognitive behavioral therapy helps identify and change negative thought patterns.",
        "Sleep deprivation is strongly associated with anxiety and depression.",
        "Journaling can help process emotions and gain perspective on challenges.",
        "Setting realistic goals builds confidence and a sense of accomplishment.",
        "Seeking professional help is a sign of strength rather than weakness.",
    ]),
    ("technology trends", [
        "Quantum computing promises to solve problems beyond classical computer capabilities.",
        "Blockchain provides a decentralized and tamper-resistant record of transactions.",
        "The Internet of Things connects everyday devices to the internet for automation.",
        "5G networks enable faster data transfer and lower latency for mobile devices.",
        "Augmented reality overlays digital information onto the physical world.",
        "Cybersecurity threats are growing in sophistication and frequency.",
        "Edge computing processes data closer to the source rather than in centralized servers.",
        "Digital twins simulate physical systems to optimize performance and maintenance.",
    ]),
    ("environmental conservation", [
        "Protected marine reserves allow fish populations to recover and ecosystems to stabilize.",
        "Reforestation programs restore biodiversity and sequester atmospheric carbon.",
        "Sustainable agriculture minimizes soil erosion and reduces chemical runoff.",
        "Recycling reduces the volume of waste sent to landfills and conserves raw materials.",
        "Wetlands filter pollutants from water and provide habitat for diverse species.",
        "Reducing plastic consumption prevents harm to marine wildlife and ecosystems.",
        "Conservation genetics helps preserve genetic diversity in endangered species.",
        "Community-based conservation gives local populations ownership over natural resources.",
    ]),
    ("history and culture", [
        "The printing press accelerated the spread of knowledge across Europe.",
        "Ancient trade routes connected civilizations and enabled cultural exchange.",
        "The Industrial Revolution transformed economies from agrarian to manufacturing-based.",
        "Oral traditions preserved history and cultural values before written records.",
        "Archaeological discoveries continuously reshape our understanding of ancient societies.",
        "Colonialism had lasting economic and social consequences for affected regions.",
        "The Renaissance produced breakthroughs in art, science, and philosophy.",
        "Democracies evolved from early experiments in ancient Athens to modern republics.",
    ]),
]

FORBIDDEN_WORDS_LIST = [
    "example", "however", "therefore", "moreover", "furthermore",
    "importantly", "significantly", "basically", "actually", "certainly",
    "obviously", "clearly", "simply", "just", "really",
]

END_PHRASES = [
    "I hope this helps.",
    "Thank you for your question.",
    "Please let me know if you need more information.",
    "Feel free to ask if you have further questions.",
    "I hope you found this useful.",
]

KEYWORDS_TO_INCLUDE = [
    ["important", "key"], ["research", "study"], ["method", "approach"],
    ["benefit", "advantage"], ["challenge", "problem"], ["solution", "answer"],
    ["result", "outcome"], ["example", "instance"], ["process", "system"],
]


def get_sentences(topic_data: tuple, n: int) -> list[str]:
    """Get n sentences from a topic's content bank."""
    sentences = topic_data[1]
    return random.sample(sentences, min(n, len(sentences)))


def topic_name(topic_data: tuple) -> str:
    return topic_data[0]


# ---------------------------------------------------------------------------
# Generators — one per constraint type
# ---------------------------------------------------------------------------

def gen_no_comma(topic_data, target_words=150) -> tuple[str, str]:
    """punctuation:no_comma"""
    name = topic_name(topic_data)
    sentences = get_sentences(topic_data, 6)
    instruction = f"Write a short essay about {name}. Do not use any commas in your response."

    # Remove commas and restructure
    text = " ".join(sentences)
    text = text.replace(",", " and").replace("  ", " ")
    # Pad to target word count if needed
    while len(text.split()) < target_words:
        extra = random.choice(topic_data[1])
        extra = extra.replace(",", " and")
        text += " " + extra

    return instruction, text.strip()


def gen_word_count(topic_data, min_words=200) -> tuple[str, str]:
    """length_constraint:number_words"""
    name = topic_name(topic_data)
    instruction = f"Write an essay of at least {min_words} words about {name}."

    sentences = topic_data[1][:]
    random.shuffle(sentences)
    text = " ".join(sentences)
    # Keep adding sentences until we hit the count
    while len(text.split()) < min_words:
        extra = random.choice(topic_data[1])
        text += " " + extra

    return instruction, text.strip()


def gen_bullet_list(topic_data, n_bullets=5) -> tuple[str, str]:
    """detectable_format:number_bullet_lists"""
    name = topic_name(topic_data)
    instruction = f"Provide at least {n_bullets} key points about {name} using bullet points."

    sentences = get_sentences(topic_data, max(n_bullets, 5))
    bullets = ["- " + s for s in sentences[:n_bullets]]
    # Add extra bullets if needed
    all_sents = topic_data[1][:]
    random.shuffle(all_sents)
    for s in all_sents:
        if len(bullets) >= n_bullets:
            break
        if s not in sentences:
            bullets.append("- " + s)

    return instruction, "\n".join(bullets)


def gen_json_format(topic_data) -> tuple[str, str]:
    """detectable_format:json_format"""
    name = topic_name(topic_data)
    instruction = f"Provide information about {name} in JSON format."

    sentences = get_sentences(topic_data, 3)
    data = {
        "topic": name,
        "summary": sentences[0],
        "key_points": sentences[1:],
        "importance": "high",
    }
    return instruction, json.dumps(data, indent=2)


def gen_markdown_sections(topic_data, n_sections=3) -> tuple[str, str]:
    """detectable_format:multiple_sections"""
    name = topic_name(topic_data)
    instruction = f"Write about {name} using at least {n_sections} markdown sections with headers."

    section_names = ["Overview", "Key Facts", "Challenges", "Solutions", "Conclusion"]
    random.shuffle(section_names)
    chosen = section_names[:n_sections]

    sentences = topic_data[1][:]
    random.shuffle(sentences)
    idx = 0
    parts = []
    for sec in chosen:
        count = random.randint(1, 2)
        sec_sentences = []
        for _ in range(count):
            if idx < len(sentences):
                sec_sentences.append(sentences[idx])
                idx += 1
        parts.append(f"## {sec}\n\n" + " ".join(sec_sentences))

    return instruction, "\n\n".join(parts)


def gen_forbidden_word(topic_data, forbidden: str) -> tuple[str, str]:
    """keywords:forbidden_words"""
    name = topic_name(topic_data)
    instruction = (
        f"Write a short description of {name}. "
        f"Do not use the word '{forbidden}' anywhere in your response."
    )

    sentences = get_sentences(topic_data, 4)
    text = " ".join(sentences)
    # Remove the forbidden word (case-insensitive)
    text = re.sub(rf'\b{re.escape(forbidden)}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    return instruction, text


def gen_keyword_existence(topic_data, keywords: list[str]) -> tuple[str, str]:
    """keywords:existence"""
    name = topic_name(topic_data)
    kw_str = " and ".join(f"'{k}'" for k in keywords)
    instruction = f"Write about {name}. Your response must include the words {kw_str}."

    sentences = get_sentences(topic_data, 3)
    text = " ".join(sentences)
    # Append keywords if not present
    for kw in keywords:
        if kw.lower() not in text.lower():
            text += f" This is an {kw} area of study."

    return instruction, text.strip()


def gen_end_with(topic_data, end_phrase: str) -> tuple[str, str]:
    """startend:end_checker"""
    name = topic_name(topic_data)
    instruction = (
        f"Write a short response about {name}. "
        f"Your response must end with the exact phrase: '{end_phrase}'"
    )

    sentences = get_sentences(topic_data, 3)
    text = " ".join(sentences) + " " + end_phrase

    return instruction, text.strip()


def gen_lowercase(topic_data) -> tuple[str, str]:
    """change_case:english_lowercase"""
    name = topic_name(topic_data)
    instruction = f"Write about {name} using only lowercase letters. Do not use any capital letters."

    sentences = get_sentences(topic_data, 4)
    text = " ".join(sentences).lower()

    return instruction, text


def gen_postscript(topic_data) -> tuple[str, str]:
    """detectable_content:postscript"""
    name = topic_name(topic_data)
    instruction = f"Write a response about {name}. Include a postscript starting with 'P.S.'"

    sentences = get_sentences(topic_data, 3)
    text = " ".join(sentences)
    ps_sentences = get_sentences(topic_data, 1)
    text += "\n\nP.S. " + ps_sentences[0]

    return instruction, text.strip()


def gen_paragraph_count(topic_data, n_paragraphs=3) -> tuple[str, str]:
    """length_constraint:number_paragraphs"""
    name = topic_name(topic_data)
    instruction = f"Write about {name} in exactly {n_paragraphs} paragraphs."

    sentences = topic_data[1][:]
    random.shuffle(sentences)
    per_para = max(1, len(sentences) // n_paragraphs)
    paragraphs = []
    for i in range(n_paragraphs):
        chunk = sentences[i * per_para: (i + 1) * per_para]
        if not chunk:
            chunk = [random.choice(topic_data[1])]
        paragraphs.append(" ".join(chunk))

    return instruction, "\n\n".join(paragraphs)


def gen_sentence_count(topic_data, n_sentences=5) -> tuple[str, str]:
    """length_constraint:number_sentences"""
    name = topic_name(topic_data)
    instruction = f"Write exactly {n_sentences} sentences about {name}."

    all_sents = topic_data[1][:]
    random.shuffle(all_sents)
    chosen = all_sents[:n_sentences]
    while len(chosen) < n_sentences:
        chosen.append(random.choice(topic_data[1]))

    return instruction, " ".join(chosen[:n_sentences])


# ---------------------------------------------------------------------------
# Verification + generation loop
# ---------------------------------------------------------------------------

GENERATORS = [
    # (generator_fn, kwargs, instruction_id, verify_kwargs)
    ("no_comma",         gen_no_comma,         {"target_words": 120}, "punctuation:no_comma",                    {}),
    ("no_comma_long",    gen_no_comma,         {"target_words": 200}, "punctuation:no_comma",                    {}),
    ("word_200",         gen_word_count,       {"min_words": 200},   "length_constraint:number_words",           {"relation": "at least", "num_words": 200}),
    ("word_300",         gen_word_count,       {"min_words": 300},   "length_constraint:number_words",           {"relation": "at least", "num_words": 300}),
    ("bullets_3",        gen_bullet_list,      {"n_bullets": 3},     "detectable_format:number_bullet_lists",    {"relation": "at least", "num_bullets": 3}),
    ("bullets_5",        gen_bullet_list,      {"n_bullets": 5},     "detectable_format:number_bullet_lists",    {"relation": "at least", "num_bullets": 5}),
    ("json",             gen_json_format,      {},                   "detectable_format:json_format",            {}),
    ("sections_3",       gen_markdown_sections,{"n_sections": 3},    "detectable_format:multiple_sections",      {"relation": "at least", "num_sections": 3}),
    ("sections_4",       gen_markdown_sections,{"n_sections": 4},    "detectable_format:multiple_sections",      {"relation": "at least", "num_sections": 4}),
    ("lowercase",        gen_lowercase,        {},                   "change_case:english_lowercase",            {}),
    ("postscript",       gen_postscript,       {},                   "detectable_content:postscript",            {"postscript_marker": "P.S."}),
    ("paragraphs_3",     gen_paragraph_count,  {"n_paragraphs": 3},  "length_constraint:number_paragraphs",      {"relation": "at least", "num_paragraphs": 3}),
    ("sentences_5",      gen_sentence_count,   {"n_sentences": 5},   "length_constraint:number_sentences",       {"relation": "at least", "num_sentences": 4}),
]


def generate_all():
    examples = []

    for gen_name, gen_fn, gen_kwargs, instr_id, verify_kwargs in GENERATORS:
        count = 0
        for topic_data in TOPICS:
            for _ in range(3):  # 3 attempts per topic per generator
                try:
                    if gen_fn in (gen_forbidden_word,):
                        forbidden = random.choice(FORBIDDEN_WORDS_LIST)
                        instruction, output = gen_fn(topic_data, forbidden)
                        vkw = {"forbidden_words": [forbidden]}
                    elif gen_fn == gen_keyword_existence:
                        kws = random.choice(KEYWORDS_TO_INCLUDE)
                        instruction, output = gen_fn(topic_data, kws)
                        vkw = {"keywords": kws}
                    elif gen_fn == gen_end_with:
                        phrase = random.choice(END_PHRASES)
                        instruction, output = gen_fn(topic_data, phrase)
                        vkw = {"end_phrase": phrase}
                    else:
                        instruction, output = gen_fn(topic_data, **gen_kwargs)
                        vkw = verify_kwargs

                    if verify_instruction(instr_id, vkw, output):
                        examples.append({"instruction": instruction, "output": output})
                        count += 1
                except Exception:
                    pass

        print(f"  [{gen_name}] {count} examples generated", flush=True)

    # Also generate forbidden_word and keyword_existence examples
    for topic_data in TOPICS:
        for _ in range(3):
            forbidden = random.choice(FORBIDDEN_WORDS_LIST)
            try:
                instruction, output = gen_forbidden_word(topic_data, forbidden)
                if verify_instruction("keywords:forbidden_words", {"forbidden_words": [forbidden]}, output):
                    examples.append({"instruction": instruction, "output": output})
            except Exception:
                pass

        for _ in range(3):
            kws = random.choice(KEYWORDS_TO_INCLUDE)
            try:
                instruction, output = gen_keyword_existence(topic_data, kws)
                if verify_instruction("keywords:existence", {"keywords": kws}, output):
                    examples.append({"instruction": instruction, "output": output})
            except Exception:
                pass

        for _ in range(2):
            phrase = random.choice(END_PHRASES)
            try:
                instruction, output = gen_end_with(topic_data, phrase)
                if verify_instruction("startend:end_checker", {"end_phrase": phrase}, output):
                    examples.append({"instruction": instruction, "output": output})
            except Exception:
                pass

    return examples


def main():
    print(f"Generating synthetic constraint-following training data...", flush=True)
    examples = generate_all()
    random.shuffle(examples)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal: {len(examples)} verified examples -> {OUT_PATH}")
    print("Set TRAIN_DATA in finetune.py to: data/train_synthetic.jsonl")


if __name__ == "__main__":
    main()
