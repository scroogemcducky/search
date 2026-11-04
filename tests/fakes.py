"""A rule-based stand-in for a chat model, so the pipeline runs in tests.

It recognises each default prompt by a phrase in it and answers in the
expected format using simple text rules. Its answers are crude but valid.
"""

import json
import re

from bookqa.llm import ScriptedLLM


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def fake_extract(user: str) -> str:
    claims = []
    for number, body in re.findall(
        r"\[Passage (\d+)\][^\n]*\n(.*?)(?=\n\n\[Passage|\Z)", user, re.S
    ):
        sentence = _first_sentence(body)
        evidence = "study" if re.search(r"\bstudy|trial|survey\b", body) else "opinion"
        claims.append(
            {
                "passage": int(number),
                "claim": sentence,
                "speaker": "",
                "stance": "as stated",
                "evidence_type": evidence,
                # Curly quotes and a trailing space, to exercise quote matching.
                "quote": sentence.replace("'", chr(0x2019)) + " ",
            }
        )
    return json.dumps({"claims": claims})


def respond(system: str, user: str) -> str:
    if "List the substantive claims" in user:
        return fake_extract(user)
    raise AssertionError(f"fake LLM got an unexpected prompt:\n{user[:300]}")


def fake_llm(cache=None) -> ScriptedLLM:
    return ScriptedLLM(respond, cache)
