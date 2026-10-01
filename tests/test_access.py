"""Access states and labels."""
import access
from summarizer import db

from conftest import ADMIN_ID


def test_states():
    assert access.state(ADMIN_ID) == "admin"
    assert access.state(42) is None
    access.set_state(42, "pending", "Bob", None)
    assert access.state(42) == "pending"
    access.set_state(42, "allowed")
    assert access.state(42) == "allowed"
    access.set_state(42, None)
    assert access.state(42) is None


def test_stale_admin_row_grants_nothing():
    db.set_user(43, "admin")  # e.g. someone removed from ADMIN_USER_IDS
    assert access.state(43) is None
    assert not access.is_admin(43)


def test_label():
    assert access.label(7, {"name": "Ana", "username": "ana"}) == "Ana (@ana, 7)"
    assert access.label(7, {"name": "Ana"}) == "Ana (7)"
    assert access.label(7, None) == "? (7)"
