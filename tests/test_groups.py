"""The bot works in private chats only."""
import bot
import access

from conftest import ADMIN_ID, callback_update, msg_update, my_chat_member_update, send


async def test_commands_and_links_in_groups_are_ignored(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/history", chat_type="group"))
    await send(app, msg_update(ADMIN_ID, "/users", chat_type="supergroup"))
    await send(app, msg_update(ADMIN_ID, "https://youtu.be/abcdefghijk", chat_type="group"))
    assert telegram.texts() == [] and bot.queue.qsize() == 0


async def test_buttons_pressed_in_groups_are_ignored(app, telegram):
    access.set_state(50, "pending")
    await send(app, callback_update(ADMIN_ID, "allow:50", chat_type="group"))
    await send(app, callback_update(ADMIN_ID, "llm:home", chat_type="group"))
    assert access.state(50) == "pending" and telegram.sent("editMessageText") == []


async def test_bot_leaves_a_group_and_tells_the_admins(app, telegram):
    await send(app, my_chat_member_update(by_uid=50, chat_type="group"))
    assert telegram.sent("leaveChat") == [{"chat_id": -100}]
    note = [d for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID][0]["text"]
    assert "Eve (50)" in note and "Family chat" in note


async def test_private_chat_membership_changes_need_no_reaction(app, telegram):
    await send(app, my_chat_member_update(by_uid=50, chat_type="private"))
    await send(app, my_chat_member_update(by_uid=50, chat_type="group", status="left"))
    assert telegram.calls == [] or all(m == "getMe" for m, _ in telegram.calls)
