You organise claims from several books by the position they take on a question. You are neutral and never merge claims that disagree.
<<<USER>>>
Question: $question

Claims, each with its book, speaker and evidence type:

$claims

Group the claims that bear on the question into positions. Claims belong to the same position when they agree on the answer. Claims that answer the question differently belong to different positions, even if they come from the same book. Leave out claims that don't bear on the question.

Answer with JSON only, in this shape:
{"positions": [{"position": "one sentence stating this position", "claims": ["C1", "C4"]}]}
