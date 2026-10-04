"""Command and button handlers: access, settings (/models, /limit, /units, /ocrlang), /history,
cancel, OCR approval and 🔊.
"""
import asyncio
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import ContextTypes

import access
from summarizer import config, db, ocr, summarize, tts
from tgbot import jobs, lifecycle, limits, menus, prefs, render, state, texts

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


MAX_PENDING = 10  # open access requests; more is a flood of throwaway accounts, not family and friends


PENDING_REPLY_EVERY = 600  # seconds between "still waiting for approval" replies to the same user


async def guard(update: Update, ctx: ContextTypes.DEFAULT_TYPE, request: bool = False) -> bool:
    """Checks whether the user may use the bot, and handles everyone who may not.

    Strangers only get an access request sent to the admins when they explicitly ask with /start, so
    random messages to the bot never ping the admins. Blocked users are ignored silently.

    Args:
        update: The incoming update.
        ctx: Handler context (used to message the admins).
        request: True for /start: file an access request for an unknown user.

    Returns:
        True if the user is an admin or allowed; False otherwise (they've been told why, if appropriate).
    """
    user = update.effective_user
    if not user:
        return False
    st = access.state(user.id)
    if st in ("admin", "allowed"):
        return True
    msg = update.effective_message
    if not access.ADMINS:  # setup mode: tell the owner their id
        if msg:
            await msg.reply_text(f"Setup: your Telegram user id is {user.id}. Add ADMIN_USER_IDS={user.id} "
                                 "to .env and restart the bot.")
        return False
    # Below, everyone who can't use the bot is answered sparingly: each reply costs a Telegram call on the
    # single update-processing path, so a spammer would otherwise slow the bot down for everyone.
    if st == "blocked":
        return False  # silently, and without a log line per message
    if st == "pending":
        # Already asked (admins were notified once); remind them at most every PENDING_REPLY_EVERY seconds.
        now = time.monotonic()
        if msg and now - state.pending_replied.get(user.id, -PENDING_REPLY_EVERY) >= PENDING_REPLY_EVERY:
            state.pending_replied[user.id] = now
            await msg.reply_text("⏳ Your access request is waiting for the admin's approval.")
        return False
    if not request:
        return False  # strangers only get an answer to /start, the one command their menu shows
    if len(access.all_users()["pending"]) >= MAX_PENDING:
        log.warning("access request from %s refused: %d requests already pending", user.id, MAX_PENDING)
        if msg:
            await msg.reply_text("🔒 This bot isn't accepting new access requests right now. Please try later.")
        return False
    info = access.set_state(user.id, "pending", user.full_name, user.username)
    log.warning("access request from %s", access.label(user.id, info))
    buttons = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Allow", callback_data=f"allow:{user.id}"),
                                     InlineKeyboardButton("❌ Deny", callback_data=f"block:{user.id}")]])
    for admin in access.ADMINS:
        try:
            await ctx.bot.send_message(admin, f"🔔 Access request from {access.label(user.id, info)}",
                                       reply_markup=buttons)
        except (BadRequest, Forbidden) as e:
            log.error("couldn't notify admin %s: %s", admin, e)
    if msg:
        await msg.reply_text("🔒 This is a private bot. I've asked the admin to give you access; "
                             "you'll get a message when it's approved.")
    return False


async def on_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /users (admins only): lists admins and users, with action buttons per user.

    Sends one message for the admins, then one per non-empty group (allowed, pending, blocked) so each
    group gets its own buttons. Non-admins get no reply at all, so the command's existence isn't revealed.
    """
    user = update.effective_user
    if not access.is_admin(user.id):
        return
    db.touch_user(user.id, user.full_name, user.username)  # admins have no /start request to record it
    users = access.all_users()

    def llm(u: dict) -> str:
        """Describes a user's AI choice, e.g. "Codex · gpt-6-sol" or "default AI".

        Args:
            u: A users-table row.

        Returns:
            Short text for the user list.
        """
        if not u.get("backend") and not u.get("model"):
            return "default AI"
        return f"{summarize.BACKEND_NAMES.get(u['backend'], u['backend'] or '')} · {u['model'] or 'default'}"

    admins = "\n".join(f"• {access.label(u['id'], u)}: {llm(u)} · no limit" for u in users["admin"])
    # Admins have a users row too (status "admin", for their settings) but aren't one of the managed
    # STATES; count only the managed ones, or the "no other users" hint would never show.
    others = sum(len(users[st]) for st in access.STATES)
    await update.message.reply_text(
        f"👑 Admins (set in .env)\n{admins}"
        + ("" if others else "\n\nNo other users yet. When someone sends the bot /start, "
                             "you'll get an access request here."))
    # "remove" doubles as Unblock: forgetting a blocked user lets them /start a new request.
    actions = {"allowed": [("🗑 Remove", "remove")], "pending": [("✅ Allow", "allow"), ("❌ Deny", "block")],
               "blocked": [("↩️ Unblock", "remove")]}
    glob = access.global_limit("daily")
    titles = {"allowed": f"✅ Allowed (daily limit: {glob or 'none'})", "pending": "⏳ Pending",
              "blocked": "⛔ Blocked"}
    for st in access.STATES:
        if not users[st]:
            continue
        rows = [[InlineKeyboardButton(f"{text}: {u['name'] or u['id']}", callback_data=f"{act}:{u['id']}")
                 for text, act in actions[st]] for u in users[st]]
        lines = "\n".join(f"• {access.label(u['id'], u)}"
                          + (f": {llm(u)} · {limits.limit_label(u['id'])}" if st == "allowed" else "")
                          for u in users[st])
        await update.message.reply_text(f"{titles[st]}\n{lines}", reply_markup=InlineKeyboardMarkup(rows))


def _llm_home(uid: int) -> tuple[str, InlineKeyboardMarkup]:
    """Builds step 1 of /models: the user's current AI and a button per installed provider.

    Args:
        uid: Telegram user id.

    Returns:
        Message text and its inline keyboard.
    """
    b, m, is_default = prefs.current_llm(uid)
    text = (f"🧠 You're using {summarize.BACKEND_NAMES[b]} · {m}" + (" (default)" if is_default else "")
            + "\n\nChoose a provider:")
    rows = []
    for backend in summarize.available_backends():
        name, billing = summarize.BACKENDS[backend]
        mark = "✓ " if backend == b else ""
        tag = " (default)" if backend == config.LLM_BACKEND else ""
        rows.append([InlineKeyboardButton(f"{mark}{name} ({billing}){tag}", callback_data=f"llm:b:{backend}")])
    if not is_default:
        rows.append([InlineKeyboardButton("↩️ Back to default", callback_data="llm:default")])
    return text, InlineKeyboardMarkup(rows)


async def llm_models(uid: int, backend: str) -> tuple[str, InlineKeyboardMarkup]:
    """Builds step 2 of /models: the models of one provider, the user's current one marked ✓.

    Args:
        uid: Telegram user id.
        backend: The provider whose models to list.

    Returns:
        Message text and its inline keyboard (an error text with only a Back button if listing fails).
    """
    b, m, _ = prefs.current_llm(uid)
    try:
        # The API backends fetch their list over the network: never on the event loop, which would
        # freeze every other chat until the provider answers.
        models = await asyncio.to_thread(summarize.list_models, backend)
    except summarize.SummaryError as e:
        return f"⚠️ {e}", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="llm:home")]])
    default = summarize.default_model(backend)
    lines = [f"🧠 {summarize.BACKEND_NAMES[backend]} models:", ""]
    rows = []
    for model in models:
        mark = "✓ " if (backend, model["id"]) == (b, m) else ""
        tag = " (default)" if model["id"] == default else ""
        lines.append(f"{mark}{model['id']}{tag}" + (f": {model['description']}" if model["description"] else ""))
        # Telegram rejects callback_data longer than 64 bytes.
        rows.append([InlineKeyboardButton(f"{mark}{model['name'] or model['id']}{tag}",
                                          callback_data=f"llm:m:{backend}:{model['id']}"[:64])])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="llm:home")])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


async def _set_llm(uid: int, backend: str | None, model: str | None) -> str:
    """Validates and stores a user's backend/model choice.

    The choice is re-validated here rather than trusted from the button, because callback data comes
    from the client and the model list can change between showing the buttons and the tap.

    Args:
        uid: Telegram user id.
        backend: The chosen backend, or None to go back to the defaults.
        model: The chosen model of that backend.

    Returns:
        The confirmation (or error) text to show the user.
    """
    if backend is None:
        db.set_user_llm(uid, None, None)
        b = config.LLM_BACKEND
        return f"✅ Back to the default: {summarize.BACKEND_NAMES[b]} · {summarize.default_model(b)}"
    if backend not in summarize.available_backends():
        return f"⚠️ {backend} isn't available on this bot."
    try:
        ids = [m["id"] for m in await asyncio.to_thread(summarize.list_models, backend)]
    except summarize.SummaryError as e:
        return f"⚠️ {e}"
    if model not in ids:
        return f"Unknown {summarize.BACKEND_NAMES[backend]} model “{model}”. Available: {', '.join(ids)}"
    db.set_user_llm(uid, backend, model)
    return (f"✅ Your summaries now use {summarize.BACKEND_NAMES[backend]} · {model}. "
            "Each video gets a summary written by this model (kept separately from other models' summaries).")


async def on_models(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /models: shows step 1 of the provider/model picker."""
    if await guard(update, ctx):
        text, buttons = _llm_home(update.effective_user.id)
        await update.message.reply_text(text, reply_markup=buttons)


def _private(update: Update) -> bool:
    """Whether the update comes from a private chat with the bot.

    The bot is for private chats only: in a group, summaries, /history and /users would be shown to every
    member. Message handlers filter on this; button handlers call it (buttons have no chat-type filter).
    """
    return bool(update.effective_chat and update.effective_chat.type == "private")


async def button(update: Update, *, admin_only: bool = False,
                  refusal: str | None = "This isn't available.") -> tuple | None:
    """The checks every inline button needs, since callback data can be forged and buttons outlive access.

    Private chat, and the user still allowed (or an admin, for admin_only); otherwise the tap is answered with
    `refusal` and None is returned. Ownership is checked per button (see _owns).

    Returns:
        (callback query, user id, data split at ":") or None.
    """
    q = update.callback_query
    uid = q.from_user.id
    allowed = access.is_admin(uid) if admin_only else access.state(uid) in ("admin", "allowed")
    if not _private(update) or not allowed:
        await q.answer(refusal)
        return None
    return q, uid, (q.data or "").split(":")


def owns(uid: int, owner: int) -> bool:
    """Whether a user may act on something of `owner`'s: their own, or anything for an admin."""
    return uid == owner or access.is_admin(uid)


async def on_my_chat_member(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Leaves any group or channel the bot is added to, and tells the admins who added it.

    BotFather's "Allow Groups" setting should prevent this (see README); this is the fallback if it's on.
    """
    change = update.my_chat_member
    chat = change.chat
    if chat.type == "private" or change.new_chat_member.status in ("left", "kicked"):
        return  # private chats are normal, and leaving needs no reaction
    who = change.from_user
    try:
        await ctx.bot.leave_chat(chat.id)
    except TelegramError as e:
        log.error("couldn't leave chat %s: %s", chat.id, e)
    log.warning("added to %s %r by %s; left", chat.type, chat.title, who and who.id)
    text = (f"⚠️ {access.label(who.id, {'name': who.full_name, 'username': who.username}) if who else 'Someone'} "
            f"added the bot to the {chat.type} “{chat.title or chat.id}”. I left it: the bot only works in "
            "private chats.")
    for admin in access.ADMINS:
        try:
            await ctx.bot.send_message(admin, text)
        except TelegramError as e:
            log.warning("couldn't notify admin %s: %s", admin, e)


async def on_llm_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles taps in the /models picker by editing the same message in place.

    Callback data: `llm:home`, `llm:default`, `llm:b:<backend>` (open a provider's models) or
    `llm:m:<backend>:<model>` (choose a model).
    """
    # Buttons can outlive access (e.g. the user was removed after /models was shown): re-check on every tap.
    if not (checked := await button(update, refusal=None)):
        return
    q, uid, _ = checked
    parts = (q.data or "").split(":", 3)  # llm:home | llm:default | llm:b:<backend> | llm:m:<backend>:<model>
    # Callback data comes from the client: anything malformed or naming an unknown provider just shows the
    # first step again instead of raising.
    action = parts[1] if len(parts) > 1 else ""
    if action == "b" and len(parts) == 3 and parts[2] in summarize.BACKENDS:
        text, buttons = await llm_models(uid, parts[2])
        await q.answer()
    elif action == "m" and len(parts) == 4 and parts[2] in summarize.BACKENDS:
        reply = await _set_llm(uid, parts[2], parts[3])
        await q.answer(reply[:200])  # Telegram caps callback answer (toast) text at 200 characters
        text, buttons = await llm_models(uid, parts[2])
    elif action == "default":
        reply = await _set_llm(uid, None, None)
        await q.answer(reply[:200])
        text, buttons = _llm_home(uid)
    else:
        text, buttons = _llm_home(uid)
        await q.answer()
    try:
        await q.edit_message_text(text, reply_markup=buttons)
    except BadRequest:  # unchanged
        pass


async def on_limit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /limit: users see their own usage; admins see and change the limits.

    Admin forms, each also with a limit's word first for that limit (`/limit ocr 5`, `/limit ask 20`; see
    limits.LIMIT_UI): `/limit` (show),
    `/limit 50` (everyone), `/limit <user id> 200` (one user), `/limit <user id> default` (remove the
    override), `0` meaning no limit.
    """
    if not await guard(update, ctx):
        return
    uid = update.effective_user.id
    if not access.is_admin(uid):
        await update.message.reply_text("\n".join(limits.usage_line(uid, k) for k in limits.LIMIT_UI))
        return
    args = list(ctx.args)
    words = {ui.word: k for k, ui in limits.LIMIT_UI.items() if ui.word}
    kind = words.get(args[0].lower(), "daily") if args else "daily"
    if kind != "daily":
        args = args[1:]
    others = ", ".join(f'"{w}"' for w in words)
    title = limits.LIMIT_UI[kind].title
    if not args:
        lines = []
        for k, ui in limits.LIMIT_UI.items():
            lines.append(f"📊 {ui.title} for everyone: {access.global_limit(k) or 'none'} {ui.what} per 24 h.")
            col = access.LIMITS[k].user_column
            lines += [f"  • {access.label(u['id'], u)}: {u[col] or 'no limit'}"
                      for u in access.all_users()["allowed"] if u.get(col) is not None]
        lines.append("\nChange it: /limit 50 · one user: /limit <user id> 200 · /limit <user id> default · "
                     f"0 = no limit. The same with {others} first for those limits: /limit ocr 5")
        await update.message.reply_text("\n".join(lines))
        return
    if len(args) == 1 and args[0].isdigit():
        db.set_setting(access.LIMITS[kind].setting, str(int(args[0])))
        await update.message.reply_text(f"✅ {title} for everyone: {int(args[0]) or 'none'}.")
        return
    if len(args) == 2 and args[0].isdigit() and (args[1].isdigit() or args[1] == "default"):
        target = int(args[0])
        value = None if args[1] == "default" else int(args[1])
        if access.is_admin(target) or not db.set_user_limit(target, kind, value):
            await update.message.reply_text("⚠️ No such user (admins have no limit).")
            return
        who = access.label(target, db.get_user(target))
        text = "back to the default" if value is None else (value or "no limit")
        await update.message.reply_text(f"✅ {title} for {who}: {text}.")
        return
    await update.message.reply_text("Usage: /limit · /limit 50 · /limit <user id> 200 · /limit <user id> default "
                                    f"(add {others} first for those limits)")


async def on_ocrlang(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /ocrlang (admins only): lists, adds or removes the languages OCR can read.

    `/ocrlang` lists the installed and available languages; `/ocrlang add <code>` downloads the model for
    the current engine (from fixed sources only) and adds it; `/ocrlang remove <code>` drops one.
    Non-admins get no reply, so the command's existence isn't revealed.
    """
    uid = update.effective_user.id
    if not access.is_admin(uid):
        return
    args = [a.lower() for a in ctx.args]
    if len(args) == 2 and args[0] in ("add", "remove"):
        if args[0] == "add" and args[1] in ocr.LANGUAGES and args[1] not in ocr.installed():
            await update.message.reply_text(f"⏳ Downloading the {ocr.LANGUAGES[args[1]][0]} model…")
        try:
            fn = ocr.add_language if args[0] == "add" else ocr.remove_language
            reply = await asyncio.to_thread(fn, args[1])  # a download takes a while: off the event loop
        except ocr.LanguageError as e:
            reply = f"⚠️ {e}"
        await update.message.reply_text(reply)
        return
    have = ocr.installed()
    others = ", ".join(f"{code} {name}" for code, (name, *_rest) in ocr.LANGUAGES.items() if code not in have)
    engine = {"tesseract": "Tesseract", "rapidocr": "RapidOCR"}[ocr.engine()]
    status = "" if ocr.available() else " (not installed!)"
    await update.message.reply_text(
        f"🔍 OCR engine: {engine}{status}\nInstalled: " + ", ".join(f"{ocr.LANGUAGES[c][0]} ({c})" for c in have)
        + f"\n\nAdd: /ocrlang add <code> · remove: /ocrlang remove <code>\nAvailable: {others}")


async def on_units(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /units: shows the user's units with buttons to change them."""
    if not await guard(update, ctx):
        return
    text, markup = menus.units_menu(update.effective_user.id)
    await update.message.reply_text(text, reply_markup=markup)


async def on_units_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /units buttons (`units:sys:<metric|imperial>`, `units:temp:<c|f>`).

    Callback data can be forged: the chat, the user's access and the value (from the fixed set) are checked.
    """
    if not (checked := await button(update)):
        return
    q, uid, parts = checked
    if len(parts) != 3 or parts[2] not in menus.UNIT_CHOICES.get(parts[1], {}):
        await q.answer("This isn't available.")
        return
    if parts[1] == "sys":
        db.set_user_units(uid, system=parts[2])
    else:
        db.set_user_units(uid, temperature=parts[2])
    text, markup = menus.units_menu(uid)
    await q.answer("Saved. Applies to new replies.")
    try:
        await q.edit_message_text(text, reply_markup=markup)
    except BadRequest:
        pass  # tapped the choice that was already set: the message is unchanged


async def on_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /history: recent requests, admins see everyone's (with who sent them), users only their own.

    Who submitted what is private, so only admins get other users' rows and the ⚡ cache-hit marker.
    """
    if not await guard(update, ctx):
        return
    uid = update.effective_user.id
    admin = access.is_admin(uid)
    rows = db.recent_requests(None if admin else uid, limit=20)
    if not rows:
        await update.message.reply_text("No requests yet.")
        return
    icons = {"done": "✅", "failed": "⚠️", "queued": "⏳", "processing": "⏳", "cancelled": "✖️",
             "waiting": "⏸"}
    lines = []
    for r in rows:
        when = time.strftime("%d.%m. %H:%M", time.localtime(r["created_at"]))
        line = f"{icons.get(r['status'], '•')} {when} "
        if admin:
            who = "you" if r["user_id"] == uid else (r["user_name"] or str(r["user_id"]))
            line += f"[{who}] "
        line += (r["title"] or r["url"])[:70]
        if r["kind"] != "summary":
            line += f" ({r['kind']})"
        if admin and r["cached"]:
            line += " ⚡"
        lines.append(line)
    head = "Recent requests (all users; ⚡ = from cache)" if admin else "Your recent requests"
    await update.message.reply_text(head + "\n\n" + "\n".join(lines), disable_web_page_preview=True)


async def on_cancel_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles "✖️ Don't wait, cancel" on a job waiting for memory (`cancel:<request id>`).

    Only the job's owner or an admin may cancel it; the data is checked, since callback data can be forged.
    """
    if not (checked := await button(update, refusal="This request isn't waiting any more.")):
        return
    q, uid, parts = checked
    job = state.jobs.get(int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else None
    if job is None or job.cancel_reason:
        await q.answer("This request isn't waiting any more.")
        return
    if not owns(uid, job.user_id):
        await q.answer("Only the person who sent this link can cancel it.")
        return
    await jobs.cancel_job(ctx.application, job, texts.CANCELLED)
    await q.answer("Cancelled.")


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the admins' user-management buttons (`allow:<id>`, `block:<id>`, `remove:<id>`).

    Used by both the access-request message and /users. After the change the user's command menu is
    re-synced, and a newly allowed user is told they're in.
    """
    # Only the admins get these buttons, but callback data can be forged: check on every tap.
    if not (checked := await button(update, admin_only=True, refusal="Admins only.")):
        return
    q = checked[0]
    action, _, uid_s = (q.data or "").partition(":")
    if action not in ("allow", "block", "remove") or not uid_s.isdigit():
        await q.answer()
        return
    uid = int(uid_s)
    # Admins come from .env; changing them here would be undone at the next restart anyway.
    if access.is_admin(uid):
        await q.answer("That's an admin (configured in .env).")
        return
    new = {"allow": "allowed", "block": "blocked", "remove": None}[action]
    info = access.set_state(uid, new) or {}
    who = access.label(uid, info)
    done = {"allow": f"✅ Allowed {who}", "block": f"⛔ Denied and blocked {who}",
            "remove": f"🗑 Removed {who}"}[action]
    log.info("admin %s: %s", q.from_user.id, done)
    if action in ("block", "remove"):
        await jobs.cancel_user_jobs(ctx.application, uid, texts.ACCESS_REMOVED)
    await q.answer(done[:200])  # Telegram caps callback answer (toast) text at 200 characters
    await q.edit_message_text(done)
    await lifecycle.sync_commands(ctx.bot, uid)
    if action == "allow":
        try:
            await ctx.bot.send_message(uid, "✅ You now have access. Send me a YouTube or TikTok link.\n\n"
                                       + texts.HELP)
        except (BadRequest, Forbidden) as e:
            log.warning("couldn't notify user %s: %s", uid, e)


async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start: the help text for allowed users, an access request for strangers.

    /start is the only way a stranger can file an access request (see `guard`).
    """
    user = update.effective_user
    db.touch_user(user.id, user.full_name, user.username)  # known users: refresh name (no-op otherwise)
    if await guard(update, ctx, request=True):
        admin = access.is_admin(update.effective_user.id)
        await update.message.reply_text(texts.HELP + (texts.ADMIN_HELP if admin else ""))


async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /help: the help text, plus the admin commands for admins."""
    if await guard(update, ctx):
        admin = access.is_admin(update.effective_user.id)
        await update.message.reply_text(texts.HELP + (texts.ADMIN_HELP if admin else ""))


async def on_ocr_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the user's OCR buttons (`ocr:<request id>:go|no|ok|ask`).

    go starts the OCR (continuing the request), no/ok drop it, ask sends the admins the details with
    Allow/Deny buttons (once per request). Everything is re-checked: chat, access, ownership, the request
    still waiting, the page cap, the OCR limit.
    """
    if not (checked := await button(update)):
        return
    q, uid, parts = checked
    req = db.get_request(int(parts[1])) if len(parts) == 3 and parts[1].isdigit() else None
    hold = db.get_ocr_hold(req["id"]) if req else None
    if not hold or not owns(uid, req["user_id"]):
        await q.answer("This isn't available.")
        return
    if req["status"] != "waiting":
        await q.answer("This request isn't waiting any more.")
        return
    action = parts[2]
    upload = db.get_upload(hold["upload_id"])
    if action in ("no", "ok"):
        db.update_request(req["id"], status="cancelled", error="OCR declined")
        await q.answer()
        await q.edit_message_text("✖️ Cancelled." if action == "no" else "OK, I won't read this scan.")
        return
    if action == "ask":
        if hold["asked"]:
            await q.answer("An admin has been asked already.")
            return
        db.update_ocr_hold(req["id"], asked=1)
        used, limit, _, _ = limits.limit_status(req["user_id"], "ocr")
        size = f" ({render.fmt_size(upload['size'])})" if upload and upload["size"] else ""
        details = (f"🙋 {access.label(req['user_id'], db.get_user(req['user_id']))} asks to read a long scan:\n"
                   f"📄 {upload['name'][:100] if upload else '?'}{size}\n"
                   f"Pages needing OCR: {hold['pages']} (limit {config.OCR_MAX_PAGES})\n"
                   f"Language: {render.ocr_names(hold['language'])}\n"
                   f"Estimated OCR time: about {render.fmt_eta(hold['seconds'])}, plus the summary\n"
                   f"Their OCR use today: {used}" + (f"/{limit}" if limit else ""))
        buttons = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Allow", callback_data=f"ocradm:{req['id']}:yes"),
                                         InlineKeyboardButton("❌ Deny", callback_data=f"ocradm:{req['id']}:no")]])
        for admin in access.ADMINS:
            try:
                await ctx.bot.send_message(admin, details, reply_markup=buttons)
            except TelegramError as e:
                log.warning("couldn't ask admin %s: %s", admin, e)
        await q.answer()
        await q.edit_message_text("🙋 I asked an admin; I'll let you know their answer.")
        return
    if action != "go":
        await q.answer("This isn't available.")
        return
    if hold["pages"] > config.OCR_MAX_PAGES and not hold["approved"] and not access.is_admin(uid):
        await q.answer("This needs an admin's approval first.", show_alert=True)
        return
    refusal = None if access.is_admin(req["user_id"]) else limits.ocr_refusal(req["user_id"])
    if refusal := refusal or limits.refusal(req["user_id"], new_request=False):
        await q.answer(refusal[:200], show_alert=True)
        return
    await q.answer()
    await q.edit_message_text(limits.queued_message())
    db.update_request(req["id"], status="queued")
    await jobs.start_job(req["user_id"], q.message.chat.id, q.message.message_id, req["url"], req["kind"],
                     request_id=req["id"], upload_id=hold["upload_id"], book_mode=hold["mode"],
                     chapter=hold["chapter"], ocr_ok=True, job_kind=state.JobKind.DOCUMENT)


async def on_ocr_admin_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles an admin's Allow/Deny on a long-scan request (`ocradm:<request id>:yes|no`)."""
    if not (checked := await button(update, admin_only=True, refusal="Only admins can do this.")):
        return
    q, _, parts = checked
    req = db.get_request(int(parts[1])) if len(parts) == 3 and parts[1].isdigit() else None
    hold = db.get_ocr_hold(req["id"]) if req else None
    if not hold or req["status"] != "waiting":
        await q.answer("Already handled.")
        await q.edit_message_reply_markup(None)
        return
    by = q.from_user.full_name
    if parts[2] == "yes":
        db.update_ocr_hold(req["id"], approved=1)
        await ctx.bot.send_message(
            req["user_id"], f"✅ An admin allowed reading all {hold['pages']} pages (about "
                            f"{render.fmt_eta(hold['seconds'])}, plus the summary). Start?",
            reply_markup=menus.ocr_buttons(req["id"], False))
        await q.edit_message_text(f"{q.message.text}\n\n✅ Allowed by {by}.")
    else:
        db.update_request(req["id"], status="cancelled", error="long OCR denied")
        await ctx.bot.send_message(req["user_id"], "❌ An admin declined reading this long scan.")
        await q.edit_message_text(f"{q.message.text}\n\n❌ Denied by {by}.")
    await q.answer()


async def on_voice_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles 🔊 Listen (`voice:<request id>`): queues a voice-message job for that summary.

    Callback data can be forged, so everything is re-checked: private chat, access, that the summary is the
    user's own (or the user is an admin) and was delivered. Listening doesn't count toward the daily limit;
    a voice message that has to be made counts toward TTS_DAILY_LIMIT (a reused one is free).
    """
    if not (checked := await button(update)):
        return
    q, uid, parts = checked
    req = db.get_request(int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else None
    if req is None or not owns(uid, req["user_id"]):
        await q.answer("This isn't available.")
        return
    spoken = db.get_spoken(req["id"])
    if not spoken or not tts.available():
        await q.answer(texts.VOICE_TOO_OLD if not spoken else "Voice messages aren't available right now.",
                       show_alert=True)
        return
    if any(j.voice_of == req["id"] and not j.cancel_reason for j in state.jobs.values()):
        await q.answer("Already on its way.")
        return
    key = tts.key(spoken["text"], spoken["lang"], spoken["voice"])
    if not access.is_admin(uid) and not db.get_voice(key, config.VOICE_CACHE_DAYS * limits.DAY):
        used, limit, _, frees_in = limits.limit_status(uid, "voice")
        if limit and used >= limit:
            await q.answer(f"You've made {limit} new voice messages today; more in about "
                           f"{render.fmt_until(frees_in or 0)}.", show_alert=True)
            return
    if refusal := limits.refusal(uid, new_request=False):
        await q.answer(refusal[:200], show_alert=True)
        return
    await q.answer()
    status = await ctx.bot.send_message(q.message.chat.id, limits.queued_message())
    job = await jobs.start_job(uid, q.message.chat.id, status.message_id, req["url"], "voice", voice_of=req["id"],
                           job_kind=state.JobKind.VOICE)
    db.update_request(job.request_id, platform="voice", video_id=key)  # what db.user_saw_video looks for
