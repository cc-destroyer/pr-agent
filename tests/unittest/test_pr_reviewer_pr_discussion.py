"""/review can read the PR's conversation, so answered findings are not raised again."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from jinja2 import Environment, StrictUndefined, select_autoescape

from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

_ENABLED_KEY = "pr_reviewer.include_pr_discussion"
_BUDGET_KEY = "pr_reviewer.max_pr_discussion_chars"


@pytest.fixture(autouse=True)
def _restore_discussion_settings():
    snapshot = snapshot_settings((_ENABLED_KEY, _BUDGET_KEY))
    yield
    restore_settings(snapshot)


def _comment(author, body):
    return SimpleNamespace(user=SimpleNamespace(login=author), body=body)


def _reviewer(comments=None, error=None):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = MagicMock()
    reviewer.git_provider.is_supported.return_value = True
    if error is not None:
        reviewer.git_provider.get_issue_comments.side_effect = error
    else:
        reviewer.git_provider.get_issue_comments.return_value = comments or []
    return reviewer


def test_is_off_by_default_and_reads_no_comments():
    reviewer = _reviewer([_comment("dev", "Not applicable.")])

    assert reviewer._get_pr_discussion() == ""
    reviewer.git_provider.get_issue_comments.assert_not_called()


def test_lists_comments_oldest_first_with_authors_and_without_hidden_markers():
    get_settings().set(_ENABLED_KEY, True)
    reviewer = _reviewer([
        _comment("PR-Agent", "<!-- pr-agent-review-state:v1\n{}\n-->\nPossible Crash: division by capacity"),
        _comment("dev", "  Doesn't apply: capacity is validated at start-up.  "),
        _comment("dev", "<!-- only a marker -->"),
        SimpleNamespace(user=None, body=None),
        _comment("dev", "/review"),
    ])

    assert reviewer._get_pr_discussion() == (
        "PR-Agent:\nPossible Crash: division by capacity"
        "\n\n-----\n\n"
        "dev:\nDoesn't apply: capacity is validated at start-up."
        "\n\n-----\n\n"
        "dev:\n/review"
    )


def test_accepts_the_setting_as_an_environment_string():
    get_settings().set(_ENABLED_KEY, "true")

    assert _reviewer([_comment("dev", "Fixed in abc123.")])._get_pr_discussion() == "dev:\nFixed in abc123."


def test_keeps_the_newest_comments_that_fit_the_budget():
    get_settings().set(_ENABLED_KEY, True)
    get_settings().set(_BUDGET_KEY, len("b:\n" + "2" * 10) + len("c:\n" + "3" * 10))
    reviewer = _reviewer([
        _comment("a", "1" * 10),
        _comment("b", "2" * 10),
        _comment("c", "3" * 10),
    ])

    assert reviewer._get_pr_discussion() == "b:\n2222222222\n\n-----\n\nc:\n3333333333"


def test_clips_a_newest_comment_larger_than_the_whole_budget():
    get_settings().set(_ENABLED_KEY, True)
    get_settings().set(_BUDGET_KEY, 8)

    assert _reviewer([_comment("a", "x" * 50)])._get_pr_discussion() == "a:\nxxxxx"


@pytest.mark.parametrize("budget", [0, -1, "not a number"])
def test_a_non_positive_or_invalid_budget_disables_it(budget):
    get_settings().set(_ENABLED_KEY, True)
    get_settings().set(_BUDGET_KEY, budget)
    reviewer = _reviewer([_comment("dev", "Not applicable.")])

    assert reviewer._get_pr_discussion() == ""
    reviewer.git_provider.get_issue_comments.assert_not_called()


def test_a_provider_without_issue_comments_gets_no_discussion():
    get_settings().set(_ENABLED_KEY, True)
    reviewer = _reviewer([_comment("dev", "Not applicable.")])
    reviewer.git_provider.is_supported.return_value = False

    assert reviewer._get_pr_discussion() == ""
    reviewer.git_provider.get_issue_comments.assert_not_called()


def test_a_failure_to_read_comments_does_not_fail_the_review():
    get_settings().set(_ENABLED_KEY, True)

    assert _reviewer(error=RuntimeError("Failed to get comments"))._get_pr_discussion() == ""


def _build_reviewer(monkeypatch, comments):
    """Run the real ``PRReviewer.__init__`` so ``self.vars`` is the shipped dict."""
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_languages.return_value = {}
    provider.get_files.return_value = []
    provider.get_pr_description.return_value = ("desc", [])
    provider.get_issue_comments.return_value = comments

    monkeypatch.setattr(pr_reviewer_module, "get_git_provider_with_context", lambda pr_url: provider)
    monkeypatch.setattr(pr_reviewer_module, "get_main_pr_language", lambda languages, files: "Python")
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", MagicMock())

    return PRReviewer(
        "https://example/pr/1",
        ai_handler=lambda: SimpleNamespace(main_pr_language=None),
    )


def _render(reviewer):
    environment = Environment(
        autoescape=select_autoescape(default_for_string=False),
        undefined=StrictUndefined,
    )
    prompt = get_settings().pr_review_prompt
    return (
        environment.from_string(prompt.system).render(reviewer.vars),
        environment.from_string(prompt.user).render(reviewer.vars),
    )


def test_prompt_carries_the_discussion_and_the_rule_for_it(monkeypatch):
    get_settings().set(_ENABLED_KEY, True)
    reviewer = _build_reviewer(monkeypatch, [_comment("dev", "Capacity is validated at start-up.")])

    system, user = _render(reviewer)

    assert "Do not raise a concern again when the discussion shows it was fixed" in system
    assert (
        "Earlier discussion on this PR, oldest first:\n======\ndev:\nCapacity is validated at start-up.\n======"
        in user
    )
    assert user.index("Earlier discussion on this PR") < user.index("The PR code diff:")


def test_prompt_is_unchanged_when_disabled(monkeypatch):
    reviewer = _build_reviewer(monkeypatch, [_comment("dev", "Capacity is validated at start-up.")])

    system, user = _render(reviewer)

    assert reviewer.vars["pr_discussion"] == ""
    assert "earlier discussion" not in system
    assert "Earlier discussion on this PR" not in user
    assert "When confidence is limited" in system
    assert "guessing.\n\nConstructing comments:" in system
