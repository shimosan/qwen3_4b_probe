#!/usr/bin/env python3
r"""venv の二重括弧プロンプト ((name) ) を単一括弧 (name) に直す。

使い方:
    # Mac / Linux
    python3 fix_venv_prompt.py ~/.venvs/aidemo2026
    # Windows (パス区切りは \)
    python3 fix_venv_prompt.py C:\path\to\venv
    # 引数を省略すると、いま activate 中の venv ($VIRTUAL_ENV) を対象にする

    - 二重括弧は「見た目だけ」の問題で venv の動作には影響しない。直さず進めてもよい。
    - 正常な venv（既に単一括弧）には「変更なし」で終わり、何も壊さない（再実行も安全・冪等）。
    - Windows PowerShell 5.1 で実行ログ（日本語）が文字化けする場合は、事前に
      `$env:PYTHONIOENCODING = "utf-8"` を設定する（動作には影響しない）。

何が起きているか（cosmetic な見た目だけの問題・venv の動作は正常）:
    `python -m venv --prompt '(name)'` のように、pyvenv.cfg の `prompt` 値に
    すでに括弧が入っていると、各シェルの activation スクリプトがその値をさらに
    「( ... ) 」で包むため、プロンプトが ((name) ) と二重括弧になる。
    これは特定シェルの問題ではなく、pyvenv.cfg の prompt に括弧があることが原因:
      - bash/zsh/csh/fish: venv 作成時に `(name) ` を焼き付けるので二重括弧が固定される
      - Windows PowerShell (Activate.ps1): 起動のたびに pyvenv.cfg の prompt を読んで
        「($prompt) 」で包むので、prompt に括弧があると二重になる
    → したがって Windows も免疫ではなく、pyvenv.cfg を直せば PowerShell は自動で単一化する。

このスクリプトがすること（再実行しても安全・冪等）:
    1. 二重括弧でなければ何もしない（正常な venv は不変）。
    2. 二重括弧なら「本来のラベル」を復元し、
       - pyvenv.cfg の prompt を括弧なしに正規化（Activate.ps1 / activate.bat の実行時参照を直す）
       - bin/activate・activate.csh・activate.fish（と Scripts/ 配下の同名）の
         焼き付き値を単一括弧 `(name) ` に書き換える
    Activate.ps1 は pyvenv.cfg を実行時に読むため直接は書き換えない（cfg 正規化で直る）。
"""
from __future__ import annotations  # 3.7+ で "X | None" 型注釈を文字列化（3.9 以下でも動く）

import os
import re
import sys
from pathlib import Path

# activation スクリプトは venv/bin (POSIX) か venv/Scripts (Windows) に置かれる
ACTIVATE_DIRS = ("bin", "Scripts")


def clean_label(value: str) -> str:
    """ '(demo)' / '((demo)) ' / "'aidemo2026'" などから中身のラベルを取り出す。"""
    v = value.strip().strip('"').strip("'").strip()
    # 外側の括弧を、無くなるまで繰り返し剥がす（多重括弧に対応）
    while True:
        m = re.match(r"^\((.*)\)$", v.strip())
        if not m:
            break
        v = m.group(1).strip()
    return v


def find_activate_dir(venv: Path) -> Path | None:
    for name in ACTIVATE_DIRS:
        d = venv / name
        if (d / "activate").exists() or (d / "Activate.ps1").exists() or (d / "activate.bat").exists():
            return d
    return None


def read_cfg_prompt(cfg: Path) -> str | None:
    if not cfg.exists():
        return None
    for line in cfg.read_text().splitlines():
        m = re.match(r"\s*prompt\s*=\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None


def looks_doubled(adir: Path, cfg_prompt: str | None) -> bool:
    """この venv が二重括弧かどうかを判定する。"""
    # 1) pyvenv.cfg の prompt に括弧があれば二重化の原因（PowerShell 含め全シェルが影響）
    if cfg_prompt and "(" in cfg_prompt:
        return True
    # 2) 焼き付き型シェルの prompt 行に (( があるか
    for fname in ("activate", "activate.csh", "activate.fish", "activate.bat"):
        f = adir / fname
        if not f.exists():
            continue
        for line in f.read_text(errors="ignore").splitlines():
            if ("VIRTUAL_ENV_PROMPT" in line or "PS1=" in line
                    or "set prompt" in line or "set_color" in line) and "((" in line:
                return True
        # 3) form B: PS1 が (${VIRTUAL_ENV_PROMPT}) と参照し、その値が括弧付き
        if fname == "activate":
            text = f.read_text(errors="ignore")
            venv_prompt = re.search(r"VIRTUAL_ENV_PROMPT=\"?([^\"\n]*)\"?", text)
            if (re.search(r"PS1=\"\(\$\{VIRTUAL_ENV_PROMPT\}", text)
                    and venv_prompt and "(" in venv_prompt.group(1)):
                return True
    return False


def fix_activate_bash(text: str, label: str) -> tuple[str, bool]:
    out, changed = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("VIRTUAL_ENV_PROMPT="):
            new = f'{indent}VIRTUAL_ENV_PROMPT="({label}) "'
        elif (stripped.startswith("PS1=") and "${PS1:-}" in stripped
              and "_OLD_VIRTUAL_PS1" not in stripped):
            new = f'{indent}PS1="({label}) ${{PS1:-}}"'
        else:
            new = line
        changed = changed or (new != line)
        out.append(new)
    return "\n".join(out) + "\n", changed


def fix_activate_csh(text: str, label: str) -> tuple[str, bool]:
    out, changed = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("setenv VIRTUAL_ENV_PROMPT"):
            new = f'{indent}setenv VIRTUAL_ENV_PROMPT "({label}) "'
        elif stripped.startswith("set prompt"):
            new = f'{indent}set prompt = "({label}) $prompt"'
        else:
            new = line
        changed = changed or (new != line)
        out.append(new)
    return "\n".join(out) + "\n", changed


def fix_activate_fish(text: str, label: str) -> tuple[str, bool]:
    out, changed = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("set -gx VIRTUAL_ENV_PROMPT"):
            new = f'{indent}set -gx VIRTUAL_ENV_PROMPT "({label}) "'
        elif "printf" in line and "set_color" in line:
            # printf "%s%s%s" (set_color XXXXXX) "((demo)) " (set_color normal)
            new = re.sub(
                r'(\(set_color [^)]*\)\s+)"[^"]*"(\s+\(set_color normal\))',
                lambda m: f'{m.group(1)}"({label}) "{m.group(2)}',
                line,
            )
        else:
            new = line
        changed = changed or (new != line)
        out.append(new)
    return "\n".join(out) + "\n", changed


def fix_activate_bat(text: str, label: str) -> tuple[str, bool]:
    # activate.bat: set "VIRTUAL_ENV_PROMPT=((demo)) " / set "PROMPT=((demo)) %PROMPT%"
    new = re.sub(
        r'(set\s+"?VIRTUAL_ENV_PROMPT=)[^"\r\n]*("?)',
        rf'\1({label}) \2',
        text,
    )
    new = re.sub(
        r'(set\s+"?PROMPT=)\(+\s*[^%]*?\s*\)+\s*(%PROMPT%)',
        rf'\1({label}) \2',
        new,
    )
    return new, new != text


def fix_cfg(cfg: Path, label: str) -> bool:
    lines = cfg.read_text().splitlines()
    changed = False
    new_lines = []
    seen = False
    for line in lines:
        if re.match(r"\s*prompt\s*=", line):
            seen = True
            repl = f"prompt = '{label}'"
            changed = changed or (repl != line)
            new_lines.append(repl)
        else:
            new_lines.append(line)
    if not seen:
        new_lines.append(f"prompt = '{label}'")
        changed = True
    if changed:
        cfg.write_text("\n".join(new_lines) + "\n")
    return changed


FIXERS = {
    "activate": fix_activate_bash,
    "activate.csh": fix_activate_csh,
    "activate.fish": fix_activate_fish,
    "activate.bat": fix_activate_bat,
}


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VIRTUAL_ENV")
    if not arg:
        sys.exit(
            "対象 venv を指定してください:\n"
            "  python3 fix_venv_prompt.py <venv_path>\n"
            "  （または venv を activate した状態で、引数なしで実行）"
        )

    venv = Path(arg).expanduser().resolve()
    adir = find_activate_dir(venv)
    cfg = venv / "pyvenv.cfg"
    if adir is None:
        sys.exit(f"activation スクリプトが見つかりません: {venv}/(bin|Scripts)\n"
                 "（venv のパスを確認してください）")

    cfg_prompt = read_cfg_prompt(cfg)
    if not looks_doubled(adir, cfg_prompt):
        print(f"OK: {venv} は既に単一括弧です（変更なし）")
        return

    # ラベル復元の優先順: pyvenv.cfg の prompt → bin/activate の VIRTUAL_ENV_PROMPT → dir 名
    label = None
    if cfg_prompt:
        label = clean_label(cfg_prompt)
    if not label:
        act = adir / "activate"
        if act.exists():
            m = re.search(r"VIRTUAL_ENV_PROMPT=\"?([^\"\n]*)\"?", act.read_text())
            if m:
                label = clean_label(m.group(1))
    if not label:
        label = venv.name

    fixed = []
    for fname, fixer in FIXERS.items():
        f = adir / fname
        if not f.exists():
            continue
        new, changed = fixer(f.read_text(), label)
        if changed:
            f.write_text(new)
            fixed.append(fname)

    if cfg.exists() and fix_cfg(cfg, label):
        fixed.append("pyvenv.cfg")

    print(f"fixed: {venv} -> ({label})")
    if fixed:
        print("  更新: " + ", ".join(fixed))
    print("  ※ Activate.ps1 は pyvenv.cfg を実行時に読むため直接編集せず、cfg 正規化で単一化します。")
    print(f"→ 新しいターミナルを開く、または  deactivate; source {adir / 'activate'}  で確認")


if __name__ == "__main__":
    main()
