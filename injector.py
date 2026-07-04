"""Text injection: set clipboard, paste at cursor. Clipboard intentionally keeps the
text afterward so the user can re-paste it into other apps (user requirement)."""
import time

import keyboard
import pyperclip


def _process_elevated(pid=None) -> bool:
    """True if the process (current if pid is None) runs with an elevated token."""
    import win32api
    import win32con
    import win32security
    if pid is None:
        handle = win32api.GetCurrentProcess()
    else:
        # QUERY_LIMITED_INFORMATION works across integrity levels
        handle = win32api.OpenProcess(0x1000, False, pid)
    try:
        token = win32security.OpenProcessToken(handle, win32con.TOKEN_QUERY)
        return bool(win32security.GetTokenInformation(token, win32security.TokenElevation))
    finally:
        if pid is not None:
            win32api.CloseHandle(handle)


def _foreground_pid() -> int:
    import win32gui
    import win32process
    hwnd = win32gui.GetForegroundWindow()
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return pid


def self_elevated() -> bool:
    """True if Vanni itself runs elevated. Best-effort (False if undeterminable)."""
    try:
        return _process_elevated(None)
    except Exception:
        return False


def is_foreground_elevated() -> bool:
    """True if the focused window's process runs elevated. A synthetic Ctrl+V is
    silently ignored by an elevated window when Vanni itself is not elevated, so
    this drives a proactive 'text is in your clipboard' warning. Best-effort:
    returns False if elevation can't be determined."""
    try:
        return _process_elevated(_foreground_pid())
    except Exception:
        return False


# settle time around the synthetic ctrl+c: lets physically-held hotkey modifiers
# lift, and gives the target app time to answer the copy. Tuned by smoke test.
_GRAB_SETTLE_S = 0.15


def grab_selection() -> str:
    """Copy the focused app's current selection and return it ('' if none).
    A sentinel clear beforehand means stale clipboard content can never be
    mistaken for a selection."""
    pyperclip.copy("")
    time.sleep(_GRAB_SETTLE_S)
    keyboard.send("ctrl+c")
    time.sleep(_GRAB_SETTLE_S)
    return pyperclip.paste()


def inject(text: str) -> bool:
    """Put text in clipboard and paste into the focused app. Returns False if the
    clipboard write didn't stick (paste itself can't be verified in arbitrary apps;
    elevated windows may silently ignore it — text stays in clipboard as fallback)."""
    if not text:
        return False
    pyperclip.copy(text)
    time.sleep(0.05)  # let the clipboard settle before pasting
    if pyperclip.paste() != text:
        return False
    keyboard.send("ctrl+v")
    return True


def notepad_roundtrip(sentinel: str) -> str:
    """Test helper: spawn Notepad, inject sentinel, read back its text, close."""
    import subprocess

    from pywinauto import Desktop

    # Win11 Notepad restores the user's previous tabs (may hold unsaved user
    # content) — never kill the process or dismiss dialogs blindly. Work in a
    # fresh tab and close only that tab afterwards.
    subprocess.Popen(["notepad.exe"])
    win = None
    for _ in range(20):
        time.sleep(0.5)
        try:
            win = Desktop(backend="uia").window(title_re=r".* - Notepad$", found_index=0)
            win.wait("ready", timeout=2)
            break
        except Exception:
            win = None
    if win is None:
        raise RuntimeError("Notepad window not found")
    win.set_focus()
    time.sleep(0.5)
    keyboard.send("ctrl+n")  # fresh empty tab
    time.sleep(0.7)
    inject(sentinel)
    time.sleep(0.8)  # let the paste land
    # Win11 Notepad's text area is a Document control; read the active tab via UIA
    doc = win.child_window(control_type="Document", found_index=0).wrapper_object()
    got = doc.iface_text.DocumentRange.GetText(-1)
    keyboard.send("ctrl+w")  # close only our tab
    time.sleep(0.7)
    try:  # our tab is dirty -> save prompt; decline for OUR tab only
        win.child_window(title="Don't save", control_type="Button").click_input()
    except Exception:
        pass
    return got


if __name__ == "__main__":
    s = f"vanni-sentinel-{int(time.time())}"
    got = notepad_roundtrip(s)
    print(f"sent: {s!r}\ngot:  {got!r}\nclipboard: {pyperclip.paste()!r}")
    assert s in got and pyperclip.paste() == s
    print("injection OK, clipboard retained")
