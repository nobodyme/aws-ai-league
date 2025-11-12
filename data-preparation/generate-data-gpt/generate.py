import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Tuple

from zoneinfo import ZoneInfo
from openai import AzureOpenAI


# ---- Time helpers (IST, down to seconds) ----
def ist_now_str() -> str:
    return datetime.now(tz=ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")


# ---- Progress helpers ----
def read_completed_topics(progress_file: str) -> List[str]:
    if not os.path.exists(progress_file):
        return []
    completed: List[str] = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            topic = line.strip()
            if topic:
                completed.append(topic)
    return completed


def flush_results(output_file: str, results_buffer: List[Dict[str, Any]]) -> int:
    if not results_buffer:
        return 0
    with open(output_file, "a", encoding="utf-8") as out:
        for item in results_buffer:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    n = len(results_buffer)
    results_buffer.clear()
    return n


def run(
    topics: List[str],
    question_prompt: str,
    answer_prompt: str,
    num_questions: int,
    target_len_of_ans: int,
    parallelism: int,
    flush_every_n: int,
    output_dir: str,
    output_filename: str,
    progress_filename: str,
    azure_deployment: str,
    temperature: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)
    progress_file = os.path.join(output_dir, progress_filename)

    completed = set(read_completed_topics(progress_file))
    remaining = [t for t in topics if t not in completed]

    print(f"[{ist_now_str()}] Starting generation.")
    print(
        f"[{ist_now_str()}] Total topics: {len(topics)} | Completed: {len(completed)} | Remaining: {len(remaining)}"
    )

    if not remaining:
        print(f"[{ist_now_str()}] Nothing to do. All topics covered.")
        return

    # Azure client (mirrors azure-openai.py pattern)
    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
    )

    # Shared buffers and locks
    results_buffer: List[Dict[str, Any]] = []
    buffer_lock = threading.Lock()
    flush_counter = 0

    progress_lock = threading.Lock()
    stats_lock = threading.Lock()

    remaining_total = len(remaining)
    completed_count = 0
    bad_question_count = 0  # counts items that failed format (or missing)

    def add_result(record: Dict[str, Any]):
        nonlocal flush_counter
        with buffer_lock:
            results_buffer.append(record)
            flush_counter += 1
            if flush_counter >= flush_every_n:
                written = flush_results(output_file, results_buffer)
                topics_left = max(0, remaining_total - completed_count)
                print(f"[{ist_now_str()}] Flushed {written} Q/A pairs to disk. Topics left: {topics_left}")
                flush_counter = 0

    def mark_topic_complete(topic: str):
        nonlocal completed_count
        with progress_lock:
            with open(progress_file, "a", encoding="utf-8") as prog:
                prog.write(topic + "\n")
            completed_count += 1
            topics_left = max(0, remaining_total - completed_count)
            print(f"[{ist_now_str()}] Marked topic complete: {topic}. Topics left: {topics_left}")

    def azure_chat(system_prompt: str, user_prompt: str, response_format: Dict[str, Any] = None) -> str:
        try:
            completion = client.chat.completions.create(
                model=azure_deployment,  # deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                **({"response_format": response_format} if response_format else {}),
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[{ist_now_str()}] Azure call failed: {e}. Retrying without response_format if set...")
            if response_format:
                completion = client.chat.completions.create(
                    model=azure_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return completion.choices[0].message.content or ""
            raise

    def parse_questions(text: str) -> Tuple[List[str], int, bool]:
        """Parse strict shape and count bad items.

        Returns (questions, bad_item_count, shape_ok).
        Does not raise for content issues; JSON errors will raise and be caught by caller.
        """
        data = json.loads(text)
        if not isinstance(data, dict) or "questions" not in data or not isinstance(data["questions"], list):
            return [], 0, False
        out: List[str] = []
        bad = 0
        for it in data["questions"]:
            if isinstance(it, str):
                s = it.strip()
                if not s:
                    bad += 1
                    continue
                if not s.endswith("?"):
                    s += "?"
                out.append(s)
            else:
                bad += 1
        return out, bad, True

    def process_topic(topic: str):
        nonlocal bad_question_count
        try:
            # 1) Generate questions
            sys_prompt_q = (
                "You are an expert question generator. Output must be STRICT JSON. "
                "Respond with exactly this shape and nothing else: {\"questions\": [\"...\"]}. "
                "No prose, no markdown, no code fences."
            )
            qp_formatted = question_prompt.format(NUM_QUESTIONS=num_questions, TOPIC=topic)
            user_prompt_q = (
                f"{qp_formatted}\n\n"
                f"Return exactly a JSON object: {{\"questions\": [<{num_questions} strings>]}}."
            )
            q_text = azure_chat(sys_prompt_q, user_prompt_q, response_format={"type": "json_object"})
            questions: List[str] = []
            bad_items = 0
            try:
                parsed, bad_items, shape_ok = parse_questions(q_text)
                questions = parsed[: num_questions]
                # Count missing questions (model returned fewer than requested)
                missing = max(0, num_questions - len(questions)) if shape_ok else num_questions
                with stats_lock:
                    bad_question_count += (bad_items + missing)
                if not shape_ok:
                    print(f"[{ist_now_str()}] WARNING topic '{topic}': invalid JSON shape for questions. Counted {missing} malformed.")
            except Exception as e:
                # JSON parse failure: count all as malformed and continue without halting
                with stats_lock:
                    bad_question_count += num_questions
                print(f"[{ist_now_str()}] WARNING topic '{topic}': failed to parse questions JSON ({e}). Counted {num_questions} malformed.")

            if not questions:
                print(f"[{ist_now_str()}] No valid questions parsed for topic: {topic}")
                return  # do not mark complete; will retry next run

            # 2) For each question, generate answer and emit JSONL
            for q in questions:
                sys_prompt_a = "You craft precise, helpful answers to the given question"
                ap_formatted = answer_prompt.format(
                    TARGET_LEN_OF_ANS=target_len_of_ans,
                    QUESTION=q,
                )
                a_text = azure_chat(sys_prompt_a, ap_formatted)
                record = {
                    "instruction": q,
                    "context": "",
                    "response": a_text.strip(),
                }
                add_result(record)

            # 3) Mark topic complete after processing all questions
            mark_topic_complete(topic)

        except Exception as e:
            print(f"[{ist_now_str()}] ERROR in topic '{topic}': {e}")

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(parallelism))) as pool:
            for t in remaining:
                pool.submit(process_topic, t)
            pool.shutdown(wait=True)
    except KeyboardInterrupt:
        print(f"[{ist_now_str()}] KeyboardInterrupt received. Flushing buffers before exit...")
    finally:
        # Final flush of any pending results
        with buffer_lock:
            if results_buffer:
                written = flush_results(output_file, results_buffer)
                topics_left = max(0, remaining_total - completed_count)
                print(f"[{ist_now_str()}] Final flush: {written} Q/A pairs. Topics left: {topics_left}")

        with stats_lock:
            print(f"[{ist_now_str()}] Summary: malformed questions counted = {bad_question_count}")

        print(f"[{ist_now_str()}] Done.")


if __name__ == "__main__":
    # ---- User-controlled variables (hardcoded) ----
    from topics import TOPICS
    from prompts import QUESTION_PROMPT, ANSWER_PROMPT

    NUM_QUESTIONS: int = 8  # number of questions per topic
    TARGET_LEN_OF_ANS: int = 100  # approximate words per answer

    PARALLELISM: int = 4  # e.g., 2 or 5
    FLUSH_EVERY_N: int = NUM_QUESTIONS * PARALLELISM

    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")
    OUTPUT_FILENAME: str = "synthetic_data.jsonl"  # JSONL with one object per line
    PROGRESS_FILENAME: str = "covered_topics.txt"  # one topic per line

    # Azure OpenAI settings
    AZURE_DEPLOYMENT: str = "gpt-4o"  # set to your Azure deployment name
    TEMPERATURE: float = 0.3

    run(
        topics=TOPICS,
        question_prompt=QUESTION_PROMPT,
        answer_prompt=ANSWER_PROMPT,
        num_questions=NUM_QUESTIONS,
        parallelism=PARALLELISM,
        target_len_of_ans=TARGET_LEN_OF_ANS,
        flush_every_n=FLUSH_EVERY_N,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        progress_filename=PROGRESS_FILENAME,
        azure_deployment=AZURE_DEPLOYMENT,
        temperature=TEMPERATURE,
    )
