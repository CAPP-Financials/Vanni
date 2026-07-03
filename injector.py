"""Text injection: set clipboard, paste at cursor. Clipboard intentionally keeps the
text afterward so the user can re-paste it into other apps (user requirement)."""
import time

import keyboard
import pyperclip


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
