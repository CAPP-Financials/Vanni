; Vanni — Inno Setup installer
; Build: pyinstaller (Vanni.spec) first, then: iscc installer.iss
; Output: Output\VanniSetup.exe

#define AppName "Vanni"
#define AppVersion "1.2.0"
#define AppExe "Vanni.exe"

[Setup]
AppId={{7E1F3C7A-2B7D-4E5B-9C4A-3F00D1C7A210}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Vanni Contributors
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
InfoBeforeFile=INSTALL_NOTES.md
OutputBaseFilename=VanniSetup
Compression=lzma2
SolidCompression=yes
; the onedir bundle is ~2.1 GB (CUDA included) — disable the disk spanning default
DiskSpanning=no
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#AppExe}

[Tasks]
Name: "startup"; Description: "Start {#AppName} when Windows starts (runs in the background, ready to dictate)"
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
; exclude user-editable/runtime files that may sit in dist\Vanni from a portable run
Source: "dist\Vanni\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion; \
    Excludes: "config.toml,corrections.json,snippets.json,.setup_done,history\*"
; user-editable config lives next to the exe (see paths.py BASE); don't clobber edits on upgrade
Source: "config.toml"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "corrections.json"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "snippets.json"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "INSTALL_NOTES.md"; DestDir: "{app}"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; per-user autostart; removed automatically on uninstall
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExe}"""; \
    Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName} now (first launch downloads models — needs internet once)"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; runtime artifacts created next to the exe
Type: files; Name: "{app}\.setup_done"
