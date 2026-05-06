"""
Pre-retrieval intent guard for non-legal queries.

Why it exists:
  Without this, a query like "who are you?" passes the retrieval gate
  (BM25 finds chunks containing "you" or "person", dense finds Article 4
  ("controller means the natural or legal person…")), and the LLM dutifully
  answers as if the question were about GDPR. Users (rightly) call that
  hallucination because the answer doesn't address their question.

Approach:
  Cheap regex classifier first (zero LLM cost) — catches the obvious 95% of
  greetings, thanks, identity, help, and goodbye messages. Only legal-looking
  queries fall through to the RAG pipeline.
"""

from __future__ import annotations

import random
import re
from typing import Optional

# Lowercase, punctuation-stripped patterns. We match exact phrases or
# whole-word starts so we don't over-trigger on phrases that legitimately
# include words like "hi" inside a longer question.
GREETING_RE = re.compile(
    r"^("
    r"hi|hello|hey|hiya|yo|hola|greetings|hi there|hello there|hey there|"
    r"good (morning|afternoon|evening|day)|sup|whats up|what's up|howdy"
    r")[\s!.\?]*$"
)

THANKS_RE = re.compile(
    r"^(thanks|thank you|thx|ty|thank u|thanks!|thanks a lot|much appreciated|cheers)[\s!.\?]*$"
)

IDENTITY_RE = re.compile(
    r"^("
    r"who are you|what are you|who r u|whats your name|what'?s your name|"
    r"introduce yourself|tell me about yourself|are you (a )?bot|are you (an? )?ai|"
    r"what is lexai|whats lexai"
    r")[\s!.\?]*$"
)

HELP_RE = re.compile(
    r"^("
    r"what can you do|how (do|can) i use (this|you)|help|how does this work|"
    r"what do you do|how to use|usage|instructions"
    r")[\s!.\?]*$"
)

GOODBYE_RE = re.compile(
    r"^(bye|goodbye|see you|see ya|later|cya|farewell|good night|gn)[\s!.\?]*$"
)


def _normalise(text: str) -> str:
    return text.strip().lower()


def classify(question: str) -> Optional[str]:
    """Return one of greeting/thanks/identity/help/goodbye, or None for legal queries."""
    q = _normalise(question)
    if not q:
        return None
    if GREETING_RE.match(q):
        return "greeting"
    if THANKS_RE.match(q):
        return "thanks"
    if IDENTITY_RE.match(q):
        return "identity"
    if HELP_RE.match(q):
        return "help"
    if GOODBYE_RE.match(q):
        return "goodbye"
    return None


# Multiple variants per intent so the assistant doesn't sound like
# a recording. random.choice picks one per call.
_RESPONSES = {
    "greeting": [
        "Hi there! 👋 I'm **LexAI** — your legal research assistant. Upload a PDF and ask me anything about it.",
        "Hello! I'm here to help you research legal documents — regulations, contracts, case law. What would you like to know?",
        "Hey! 😊 Drop me a legal question or upload a document and I'll dig in.",
        "Hi! I can answer questions grounded in any legal PDF you've uploaded. What's on your mind?",
    ],
    "thanks": [
        "You're welcome! Let me know if you'd like me to dig into anything else.",
        "Happy to help. Any other questions?",
        "Anytime! 👍",
        "Glad I could help — ask me anything else any time.",
    ],
    "identity": [
        "I'm **LexAI**, a legal research assistant.\n\nI retrieve and cite passages from PDFs you upload — I won't guess or make things up. Every claim I make includes an inline citation pointing to the source document, page, and section.\n\nUpload a PDF to get started, or ask me about something I've already indexed.",
        "I'm **LexAI** — a Retrieval-Augmented Generation assistant tuned for legal documents.\n\nWhat I can do:\n- Answer questions grounded strictly in PDFs you upload (with citations)\n- Refuse politely when a question is outside the documents (no hallucinations)\n- Run multi-step research as an agent (Tab → **Agent**) using both your docs and the live web\n- Evaluate my own accuracy on a test set you create (Tab → **Evaluation**)",
    ],
    "help": [
        "Here's how to use me:\n\n1. **Upload a PDF** in the right-hand panel of the *RAG Pipeline* tab\n2. **Ask a question** — I'll find relevant passages and cite them inline\n3. Need multi-step research? Try the **Agent** tab — it can also search the web\n4. Want to measure accuracy? The **Evaluation** tab runs a RAGAS-style suite\n\nYou can also paste your **OpenAI API key** in *Settings* if you haven't yet.",
        "Quick tour:\n\n- **RAG Pipeline** — chat with your PDFs, get cited answers\n- **Evaluation** — auto-generate or hand-write test questions and measure RAG quality\n- **Agent** — multi-step research with optional human-in-the-loop\n- **Settings** — API keys, vector backend, chunking parameters",
    ],
    "goodbye": [
        "Goodbye! Come back any time. 👋",
        "See you later — your conversations are saved in the sidebar.",
        "Bye! Sessions persist, so you can pick up right where you left off.",
    ],
}


def respond(intent: str) -> str:
    return random.choice(_RESPONSES[intent])
