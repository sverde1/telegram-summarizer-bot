"""Prompts, parsing, backends and the Codex command line."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from summarizer import config, netjail, summarize

from helpers import SUMMARY


def test_parse_accepts_fences_and_checks_keys():
    assert summarize._parse("```json\n" + json.dumps(SUMMARY) + "\n```", summarize.SCHEMA) == SUMMARY
    with pytest.raises(summarize.SummaryError) as e:
        summarize._parse('{"title": "x"}', summarize.SCHEMA)
    assert str(e.value) == summarize.AI_FAILED and "missing" in e.value.detail
    with pytest.raises(summarize.SummaryError) as e:
        summarize._parse("not json", summarize.SCHEMA)
    assert "invalid JSON" in e.value.detail


def test_schemas_are_strict():
    for schema in (summarize.SCHEMA, summarize.FIRST_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_format_transcript_marks_time():
    text = summarize.format_transcript([(0, "a"), (5, "b"), (12, "c"), (75, "d")])
    assert text == "[0:00] a b \n[0:12] c \n[1:15] d"


def test_available_backends(codex_home, monkeypatch):
    monkeypatch.setattr(summarize.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    assert summarize.available_backends() == ["codex", "claude-code"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert summarize.available_backends() == ["codex", "claude-code", "openai-api"]


def test_codex_model_list_follows_codex(codex_home):
    assert [m["id"] for m in summarize.list_models("codex")] == ["gpt-a"]


def test_labels_and_defaults():
    assert summarize.default_model("codex") == "gpt-test"
    assert summarize.llm_label("codex") == "Codex (gpt-test)"
    assert summarize.llm_label("claude-code", "claude-x") == "Claude Code (claude-x)"
    with pytest.raises(summarize.SummaryError):
        summarize.conversation("nope")


def test_codex_runs_sandboxed_and_resumes_by_session_id(codex_home, monkeypatch):
    calls = []

    def fake_run(cmd, *, timeout, input=None, **kwargs):
        """Plays Codex: writes the answer where -o points (via the bwrap mount) and reports a session id."""
        calls.append(cmd)
        out_dir = cmd[cmd.index("/job/out") - 1]
        schema_keys = SUMMARY | ({"needs_frames": False, "frame_moments": []} if len(calls) == 1 else {})
        (Path(out_dir) / "result.json").write_text(json.dumps(schema_keys))
        return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"thread.started","thread_id":"S-1"}\n',
                                           stderr="")

    monkeypatch.setattr(summarize.proc, "run", fake_run)
    conv = summarize.conversation("codex")
    conv.start({"title": "t", "uploader": "u", "upload_date": "", "duration": 1, "description": ""},
               "youtube", "text", "captions", "en", [])
    conv.add_frames([])
    first, second = calls
    assert first[1:3] == ["-I", netjail.__file__] and "--clearenv" in first and "--share-net" not in first
    jail_args = first[3:first.index("--", 3)]
    assert not any("/home" in a for a in jail_args) and "/etc/resolv.conf" not in jail_args
    assert first[first.index("codex"):][:2] == ["codex", "exec"] and "-m" in first and "gpt-test" in first
    assert second[second.index("codex"):][:3] == ["codex", "exec", "resume"] and "S-1" in second
    assert "--last" not in second


def test_codex_runs_in_the_network_jail_or_not_at_all(monkeypatch, tmp_path):
    jailed = summarize._bwrap(tmp_path)
    assert jailed[1:3] == ["-I", netjail.__file__] and jailed[-1] == "--" and "--share-net" not in jailed
    monkeypatch.setattr(netjail, "_state", {"ready": False})
    with pytest.raises(summarize.SummaryError) as e:
        summarize._bwrap(tmp_path)
    assert "network jail" in e.value.detail
    monkeypatch.setattr(config, "NETJAIL", False)  # the owner's opt-out: the shared network
    shared = summarize._bwrap(tmp_path)
    assert shared[:4] == ["bwrap", "--unshare-all", "--share-net", "--die-with-parent"]
    assert "/etc/resolv.conf" in shared


def test_codex_errors_come_from_its_json_events():
    out = ('{"type":"thread.started","thread_id":"S"}\n'
           '{"type":"error","message":"stream disconnected before completion"}\n'
           '{"type":"turn.failed","error":{"message":"stream disconnected before completion"}}\n')
    assert summarize._codex_error(out) == "stream disconnected before completion"
    assert summarize._codex_error("not json\n") == ""


def test_the_prompt_ignores_ads_in_descriptions():
    assert "affiliate or shop links" in summarize.SYSTEM and "mention a sponsorship only in the caveats" in \
        summarize.SYSTEM


def _stalling_codex(codex_home, stalls: set[int], seen: list):
    """A fake proc.run playing Codex whose calls numbered in `stalls` hang (ProcStalled) after appending to the
    session file, as a real stalled turn does; the others answer. `seen` gets (call, session file, idle)."""
    calls = []
    shutil.rmtree(Path(config.CODEX_HOME) / "sessions", ignore_errors=True)  # the Codex home outlives one test

    def fake_run(cmd, *, timeout, input=None, idle_timeout=None, **kwargs):
        """Writes the session file and the answer like Codex, or stalls."""
        calls.append(cmd)
        n = len(calls)
        session = Path(config.CODEX_HOME) / "sessions" / "2026" / "rollout-x-S-1.jsonl"
        session.parent.mkdir(parents=True, exist_ok=True)
        seen.append((n, session.read_text() if session.exists() else None, idle_timeout))
        with session.open("a") as f:
            f.write(f"call {n}\n")
        if n in stalls:
            raise summarize.proc.ProcStalled("codex printed nothing",
                                             stdout='{"type":"thread.started","thread_id":"S-1"}\n')
        out_dir = cmd[cmd.index("/job/out") - 1]
        resumed = "resume" in cmd
        (Path(out_dir) / "result.json").write_text(json.dumps(
            SUMMARY | ({} if resumed else {"needs_frames": False, "frame_moments": []})))
        return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"thread.started","thread_id":"S-1"}\n', stderr="")

    return fake_run


def _start(conv):
    """Turn 1 of a small video summary."""
    return conv.start({"title": "t", "uploader": "u", "upload_date": "", "duration": 1, "description": ""},
                      "youtube", "text", "captions", "en", [])


def test_a_stalled_codex_turn_two_is_resumed_from_the_session_as_it_was(codex_home, monkeypatch):
    seen = []
    monkeypatch.setattr(summarize.proc, "run", _stalling_codex(codex_home, {2}, seen))
    conv = summarize.conversation("codex")
    _start(conv)
    assert conv.add_frames([])["summary"] == SUMMARY["summary"]
    # The retry (call 3) sees the session as turn 1 left it, without the stalled attempt's frames prompt.
    assert [s for _, s, _ in seen] == [None, "call 1\n", "call 1\n"]
    assert all(idle >= config.CODEX_STALL_SEC for _, _, idle in seen)


def test_a_stalled_codex_turn_one_runs_again_fresh(codex_home, monkeypatch):
    seen = []
    monkeypatch.setattr(summarize.proc, "run", _stalling_codex(codex_home, {1}, seen))
    conv = summarize.conversation("codex")
    assert _start(conv)["summary"] == SUMMARY["summary"]
    assert [s for _, s, _ in seen] == [None, None]  # the stalled attempt's session was deleted


def test_codex_stalling_twice_fails_with_try_again(codex_home, monkeypatch):
    monkeypatch.setattr(summarize.proc, "run", _stalling_codex(codex_home, {1, 2}, []))
    with pytest.raises(summarize.SummaryError) as e:
        _start(summarize.conversation("codex"))
    assert str(e.value) == summarize.AI_FAILED and "stalled twice" in e.value.detail


def test_codex_stall_limit_grows_with_the_request_and_can_be_off(monkeypatch):
    monkeypatch.setattr(config, "CODEX_STALL_SEC", 150)
    assert summarize._codex_stall_limit(1000, 0) == 150
    assert summarize._codex_stall_limit(config.BOOK_CHUNK_CHARS, 0) > 150  # a book chunk is quiet for longer
    monkeypatch.setattr(config, "CODEX_STALL_SEC", 0)
    assert summarize._codex_stall_limit(1000, 0) is None
