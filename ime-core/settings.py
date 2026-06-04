"""
IME Settings dialog — mode selection, Ollama/DeepSeek configuration.
"""

import tkinter as tk
from tkinter import ttk

from config import load_config, save_config, VALID_MODES

FONT_LABEL = ("Microsoft YaHei", 11)
FONT_ENTRY = ("Consolas", 11)
FONT_TITLE = ("Microsoft YaHei", 12, "bold")


MODE_LABELS = {
    "map_only": "仅拼音映射（离线，最快）",
    "map_ollama": "映射 + Ollama 本地模型",
    "map_deepseek": "映射 + DeepSeek 远程",
    "map_ollama_deepseek": "映射 + Ollama + DeepSeek（最智能）",
}


class SettingsDialog:
    """Modal settings dialog. Blocks until closed."""

    def __init__(self, parent):
        self._result = None  # dict or None (cancelled)

        self._win = tk.Toplevel(parent)
        self._win.title("IME 设置")
        self._win.geometry("520x430")
        self._win.resizable(False, False)
        self._win.transient(parent)
        self._win.grab_set()

        # Load current config
        self._cfg = load_config()

        self._build_ui()
        self._win.wait_window()

    def _build_ui(self):
        pad = {"padx": 16, "pady": (12, 4)}

        # ── Title ──
        tk.Label(self._win, text="输入法引擎设置", font=FONT_TITLE).pack(
            **pad, anchor="w"
        )

        # ── Mode ──
        mode_frame = tk.Frame(self._win)
        mode_frame.pack(fill="x", **pad)
        tk.Label(mode_frame, text="处理模式:", font=FONT_LABEL).pack(anchor="w")
        self._mode_var = tk.StringVar(value=self._cfg.get("mode", "map_ollama_deepseek"))
        for key, label in MODE_LABELS.items():
            tk.Radiobutton(
                mode_frame, text=label, variable=self._mode_var,
                value=key, font=FONT_LABEL, anchor="w",
            ).pack(fill="x", padx=(12, 0), pady=1)

        # ── Separator ──
        ttk.Separator(self._win, orient="horizontal").pack(fill="x", padx=16, pady=8)

        # ── Ollama section ──
        self._ollama_frame = tk.LabelFrame(self._win, text="Ollama (本地模型)", font=FONT_LABEL)
        self._ollama_frame.pack(fill="x", padx=16, pady=4)

        row = 0
        ollama_cfg = self._cfg.get("ollama", {})
        tk.Label(self._ollama_frame, text="地址:", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        self._ollama_endpoint = tk.Entry(
            self._ollama_frame, font=FONT_ENTRY, width=35,
        )
        self._ollama_endpoint.insert(0, ollama_cfg.get("endpoint", "http://localhost:11434/v1"))
        self._ollama_endpoint.grid(row=row, column=1, padx=8, pady=3)
        row += 1

        tk.Label(self._ollama_frame, text="模型:", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        self._ollama_model = tk.Entry(
            self._ollama_frame, font=FONT_ENTRY, width=35,
        )
        self._ollama_model.insert(0, ollama_cfg.get("model", "qwen2.5:1.5b"))
        self._ollama_model.grid(row=row, column=1, padx=8, pady=3)
        row += 1

        tk.Label(self._ollama_frame, text="超时(秒):", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        self._ollama_timeout = tk.Entry(
            self._ollama_frame, font=FONT_ENTRY, width=10,
        )
        self._ollama_timeout.insert(0, str(ollama_cfg.get("timeout", 5)))
        self._ollama_timeout.grid(row=row, column=1, sticky="w", padx=8, pady=3)

        # ── DeepSeek section ──
        self._ds_frame = tk.LabelFrame(self._win, text="DeepSeek (远程)", font=FONT_LABEL)
        self._ds_frame.pack(fill="x", padx=16, pady=4)

        row = 0
        llm_cfg = self._cfg.get("llm", {})
        tk.Label(self._ds_frame, text="地址:", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        self._ds_endpoint = tk.Entry(
            self._ds_frame, font=FONT_ENTRY, width=35,
        )
        self._ds_endpoint.insert(0, llm_cfg.get("endpoint", "https://api.deepseek.com/v1"))
        self._ds_endpoint.grid(row=row, column=1, padx=8, pady=3)
        row += 1

        tk.Label(self._ds_frame, text="模型:", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        self._ds_model = tk.Entry(
            self._ds_frame, font=FONT_ENTRY, width=35,
        )
        self._ds_model.insert(0, llm_cfg.get("model", "deepseek-chat"))
        self._ds_model.grid(row=row, column=1, padx=8, pady=3)
        row += 1

        tk.Label(self._ds_frame, text="API Key:", font=FONT_LABEL).grid(
            row=row, column=0, sticky="w", padx=8, pady=3
        )
        api_key = llm_cfg.get("api_key", "")
        display_key = api_key[:8] + "****" if len(api_key) > 8 else api_key
        self._ds_apikey = tk.Entry(self._ds_frame, font=FONT_ENTRY, width=35, show="*")
        self._ds_apikey.insert(0, api_key)
        self._ds_apikey.grid(row=row, column=1, padx=8, pady=3)
        row += 1

        # ── Batch update button ──
        ttk.Separator(self._win, orient="horizontal").pack(fill="x", padx=16, pady=8)
        batch_frame = tk.Frame(self._win)
        batch_frame.pack(fill="x", padx=16, pady=(0, 4))
        tk.Label(
            batch_frame, text="批量更新词库: 用 Ollama 预处理所有音节,更新字形频率和权重",
            font=("Microsoft YaHei", 9), fg="#666",
        ).pack(anchor="w")
        self._batch_btn = tk.Button(
            batch_frame, text="开始批量更新", font=FONT_LABEL,
            bg="#FF9800", fg="white", command=self._batch_update,
        )
        self._batch_btn.pack(anchor="w", pady=4)
        self._batch_status = tk.StringVar(value="")
        tk.Label(
            batch_frame, textvariable=self._batch_status,
            font=("Microsoft YaHei", 9), fg="#666",
        ).pack(anchor="w")

        # ── Buttons ──
        btn_frame = tk.Frame(self._win)
        btn_frame.pack(fill="x", padx=16, pady=(12, 12))
        tk.Button(
            btn_frame, text="取消", font=FONT_LABEL,
            command=self._cancel,
        ).pack(side="right", padx=(6, 0))
        tk.Button(
            btn_frame, text="保存", font=FONT_LABEL,
            bg="#4CAF50", fg="white", command=self._save,
        ).pack(side="right", padx=(6, 0))

    def _collect(self) -> dict:
        """Read form values into a config dict."""
        return {
            "mode": self._mode_var.get(),
            "ollama": {
                "enabled": True,
                "endpoint": self._ollama_endpoint.get().strip(),
                "model": self._ollama_model.get().strip(),
                "timeout": max(1, int(self._ollama_timeout.get().strip() or "5")),
            },
            "llm": {
                "endpoint": self._ds_endpoint.get().strip(),
                "model": self._ds_model.get().strip(),
                "api_key": self._ds_apikey.get().strip(),
                "timeout": 15,
            },
            "engine": self._cfg.get("engine", {}),
            "learning": self._cfg.get("learning", {}),
        }

    def _save(self):
        self._result = self._collect()
        save_config(self._result)
        self._win.destroy()

    def _cancel(self):
        self._result = None
        self._win.destroy()

    def _batch_update(self):
        """Trigger batch map update via Ollama in a subprocess."""
        self._batch_btn.config(state="disabled", text="更新中...")
        self._batch_status.set("正在启动 Ollama 批量处理...")

        # Save config first so subprocess picks up Ollama settings
        cfg = self._collect()
        save_config(cfg)

        import subprocess, threading

        def _run():
            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "update_map.py",
            )
            env = os.environ.copy()
            env["OLLAMA_ENDPOINT"] = cfg["ollama"]["endpoint"]
            env["OLLAMA_MODEL"] = cfg["ollama"]["model"]
            env["OLLAMA_TIMEOUT"] = str(cfg["ollama"].get("timeout", 10))

            try:
                proc = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env,
                )
                for line in proc.stdout:
                    self._batch_status.set(line.strip()[-80:])
                proc.wait()
                if proc.returncode == 0:
                    self._batch_status.set("批量更新完成! 词库权重已优化。")
                else:
                    self._batch_status.set(f"更新失败 (exit={proc.returncode})，查看终端输出。")
            except Exception as e:
                self._batch_status.set(f"错误: {e}")
            finally:
                self._win.after(0, lambda: self._batch_btn.config(
                    state="normal", text="开始批量更新",
                ))

        threading.Thread(target=_run, daemon=True).start()

    @property
    def result(self) -> dict | None:
        """Returns the saved config dict, or None if cancelled."""
        return self._result
