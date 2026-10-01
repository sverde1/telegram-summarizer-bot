"""Prompts, parsing, backends and the Codex command line."""
import json
import subprocess
from pathlib import Path

import pytest

from summarizer import config, summarize

from helpers import SUMMARY


def test_parse_accepts_fences_and_checks_keys():
    assert summarize._parse("```json\n" + json.dumps(SUMMARY) + "\n```", summarize.SCHEMA) == SUMMARY
    with pytest.raises(summarize.SummaryError, match="missing"):
        summarize._parse('{"title": "x"}', summarize.SCHEMA)
    with pytest.raises(summarize.SummaryError, match="invalid JSON"):
        summarize._parse("not json", summarize.SCHEMA)


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

    def fake_run(cmd, input=None, **kwargs):
        """Plays Codex: writes the answer where -o points (via the bwrap mount) and reports a session id."""
        calls.append(cmd)
        out_dir = cmd[cmd.index("/job/out") - 1]
        schema_keys = SUMMARY | ({"needs_frames": False, "frame_moments": []} if len(calls) == 1 else {})
        (Path(out_dir) / "result.json").write_text(json.dumps(schema_keys))
        return subprocess.CompletedProcess(cmd, 0, stdout='{"type":"thread.started","thread_id":"S-1"}\n',
                                           stderr="")

    monkeypatch.setattr(summarize.subprocess, "run", fake_run)
    conv = summarize.conversation("codex")
    conv.start({"title": "t", "uploader": "u", "upload_date": "", "duration": 1, "description": ""},
               "youtube", "text", "captions", "en", [])
    conv.add_frames([])
    first, second = calls
    assert first[0] == "bwrap" and "--clearenv" in first and "--unshare-all" in first
    assert not any("/home" in a for a in first[first.index("bwrap"):first.index("codex")])
    assert first[first.index("codex"):][:2] == ["codex", "exec"] and "-m" in first and "gpt-test" in first
    assert second[second.index("codex"):][:3] == ["codex", "exec", "resume"] and "S-1" in second
    assert "--last" not in second
