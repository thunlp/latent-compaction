"""Fixed, task-independent source prompts for document-level teacher rollouts."""

REQUESTS = (
    "Summarize the document's main subject and structure.",
    "Describe the writing style used in the document.",
    "What broad themes appear in the document?",
    "Give a concise overview of the document.",
    "Explain how the document is organized.",
    "Describe the kinds of statements found in the document.",
    "What is the general tone of the document?",
    "Paraphrase the document's introductory idea.",
    "What recurring language patterns appear in the document?",
    "Describe the document without quoting it directly.",
    "What would be a suitable title for this document?",
    "Give a high-level outline of the document.",
    "What kind of reader is this document written for?",
    "Explain the document's overall purpose.",
    "What stylistic features are most noticeable?",
    "Write one sentence characterizing the document.",
    "Identify the document's main topics.",
    "What factual details are emphasized in the document?",
    "Which parts of the document are most distinctive?",
    "Describe how ideas progress through the document.",
    "What concepts recur throughout the text?",
    "State the document's central claims.",
    "What examples does the document use?",
    "Describe the document's vocabulary and sentence patterns.",
)

INSTRUCTIONS = (
    "Answer briefly.",
    "Use one sentence.",
    "Be precise and concise.",
    "Focus only on information stated in the text.",
    "Do not introduce outside information.",
    "Use at most two sentences.",
    "Give a compact answer.",
    "Emphasize concrete textual evidence.",
    "Preserve important proper nouns.",
    "Avoid speculation.",
    "Answer in plain language.",
    "Prioritize the most important point.",
    "Mention one supporting detail.",
    "Use a short noun phrase when possible.",
    "Distinguish content from writing style.",
    "State only what can be inferred directly.",
    "Keep the answer self-contained.",
    "Use a neutral tone.",
    "Retain specific terminology from the text.",
    "Answer as accurately as possible in a compact form.",
)

SOURCE_PROMPTS = tuple(
    f"{request} {instruction}\nAnswer:"
    for instruction in INSTRUCTIONS
    for request in REQUESTS
)

assert len(SOURCE_PROMPTS) == len(set(SOURCE_PROMPTS)) == 480
