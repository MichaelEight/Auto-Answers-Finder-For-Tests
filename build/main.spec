# -*- mode: python ; coding: utf-8 -*-
import os

# Bundle the optional drag & drop library when it is installed
datas, binaries, hiddenimports = [], [], []
try:
    from PyInstaller.utils.hooks import collect_all
    d, b, h = collect_all('tkinterdnd2')
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

a = Analysis(
    [os.path.join(SPECPATH, '..', 'main.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
