gpt-synthetic-data
===================

Simple, Azure OpenAI–backed synthetic data generator with checkpointing.

How It Works
------------
- Configure in `generate.py` (`__main__`):
  - `NUM_QUESTIONS`, `PARALLELISM`, `FLUSH_EVERY_N`, paths
  - `PARALLELISM`, `FLUSH_EVERY_N`, paths
  - `AZURE_DEPLOYMENT`, `TEMPERATURE`
- Processes topics in parallel; for each topic:
  1) Generates questions, then 2) generates answers for each question.
- Appends Q/A JSONL records after every `FLUSH_EVERY_N` generations.
- Marks a topic complete only after all its Q/A pairs are generated.
- Resumes by skipping topics listed in `output/covered_topics.txt`.
- All timestamps are in IST with second precision.

JSON-only Questions
-------------------
- The question step enforces a strict JSON object and, when supported by your Azure model/version,
  uses `response_format={"type": "json_object"}`.
- Expected response shape (only): `{ "questions": ["question 1?", "question 2?", ...] }`.
- If the model returns a wrong shape or non-string items, those are skipped and counted; the run does not halt.
- A running total of malformed questions is printed at the end.

Progress Logging
----------------
- After each file write (flush), the script logs how many topics are left.

Where To Edit
-------------
- `topics.py` — edit `TOPICS`.
- `prompts.py` — edit `QUESTION_PROMPT` and `ANSWER_PROMPT`.
- `generate.py` — tweak `NUM_QUESTIONS`, `NUM_OF_VIEWPOINTS`, `TARGET_LEN_OF_ANS`,
  `PARALLELISM`, `FLUSH_EVERY_N`, Azure settings.

Template Variables
------------------
- `QUESTION_PROMPT` supports `{NUM_QUESTIONS}`, `{TOPIC}` and is formatted per-topic.
- `ANSWER_PROMPT` supports `{NUM_OF_VIEWPOINTS}`, `{QUESTION}`, `{TARGET_LEN_OF_ANS}` and is
  formatted per-question; the topic is also appended for extra context.

Azure setup
-----------
Set environment variables before running:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (e.g., `2025-01-01-preview`)
- `AZURE_OPENAI_API_BASE` (e.g., `https://<your-endpoint>.openai.azure.com`)
Also set `AZURE_DEPLOYMENT` in `generate.py` to your chat deployment name.

Run
---
- From this folder: `python3 generate.py`

Output
------
- `output/synthetic_data.jsonl` — JSON lines; schema per line:
  - `{ "instruction": <question>, "context": "", "response": <answer>, "topic": <topic>, "generated_at_ist": <timestamp> }`
- `output/covered_topics.txt` — One topic per line; used for resume.

Note
----
- If interrupted mid-topic, some Q/A pairs for that topic may have been flushed; the topic will not be marked complete and will be regenerated on the next run, which can produce duplicates. Deduping is out of scope for this script.


Data experiments pending
- remove area specific instructions from prompt
- experiment combining output from another model
