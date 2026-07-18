"""PySide6 GUI application for the State-Projection Agent Loop examples.

Launch with:
    python -m examples.gui_app

Select a scenario, then chat freely. The left panel shows the conversation;
the right panel displays the system prompt, tool registry with categories
and counts, current state, and session metrics.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QRect, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# -- project imports ---------------------------------------------------------
from state_projection_loop import Registry, Session
from state_projection_loop.adapters import DeepSeekAdapter
from state_projection_loop.builtin.state import install_state

from examples.coding_agent.tools import CODING_KERNEL, build_coding_registry, seed_workspace
from examples.customer_support.tools import SUPPORT_KERNEL, SupportBackend, build_support_registry
from examples.game_master.tools import GM_KERNEL, MediaLog, build_game_registry, initial_seed

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ============================================================================
# Session worker (runs blocking session.send() on a background thread)
# ============================================================================

class SessionWorker(QThread):
    """Calls ``session.send(text)`` on a background thread and emits the
    assistant reply, tool observations (structured), and state updates."""

    reply_ready = Signal(str)
    tool_called = Signal(str, str, str)  # tool_name, arguments_json, observation
    assistant_thought = Signal(str)       # intermediate assistant text before tool calls
    state_updated = Signal(dict)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, session: Session, user_text: str, parent=None):
        super().__init__(parent)
        self.session = session
        self.user_text = user_text
        self._interrupted = False

    def run(self) -> None:
        try:
            # Record conversation length before sending
            initial_len = len(self.session.conversation)

            reply = self.session.send(self.user_text)

            # Extract tool calls and intermediate assistant thoughts from new messages
            new_messages = self.session.conversation[initial_len:]
            for msg in new_messages:
                role = getattr(msg, "role", "")
                if role == "assistant":
                    text = msg.text() if hasattr(msg, "text") else str(msg)
                    tc_list = getattr(msg, "tool_calls", None) or []
                    # Emit intermediate thought/text (if any, before tool calls)
                    if text and text != reply:
                        self.assistant_thought.emit(text)
                    # Emit each tool call
                    for tc in tc_list:
                        tc_name = getattr(tc, "name", "?")
                        tc_args = getattr(tc, "arguments", {})
                        args_json = json.dumps(tc_args, ensure_ascii=False)
                        self.tool_called.emit(tc_name, args_json, "")
                elif role in ("tool", "observation"):
                    # Tool observation - find the preceding tool_call and emit combined
                    text = msg.text() if hasattr(msg, "text") else str(msg)
                    tc_name = getattr(msg, "name", "?")
                    # Emit as tool call with observation
                    self.tool_called.emit(tc_name, "", text[:800])

            self.reply_ready.emit(reply)
            # Emit latest state from the session
            self.state_updated.emit(dict(self.session.state))
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def interrupt(self) -> None:
        self._interrupted = True
        self.session.interrupt()


# ============================================================================
# Syntax highlighter for JSON / code snippets in conversation
# ============================================================================

class ConversationHighlighter(QSyntaxHighlighter):
    """Minimal highlighting: role labels in colour, tool-call blocks distinct."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._user_fmt = QTextCharFormat()
        self._user_fmt.setForeground(QColor("#4fc3f7"))
        self._user_fmt.setFontWeight(QFont.Bold)

        self._assistant_fmt = QTextCharFormat()
        self._assistant_fmt.setForeground(QColor("#a5d6a7"))
        self._assistant_fmt.setFontWeight(QFont.Bold)

        self._tool_fmt = QTextCharFormat()
        self._tool_fmt.setForeground(QColor("#ffcc80"))
        self._tool_fmt.setFontWeight(QFont.Bold)

        self._error_fmt = QTextCharFormat()
        self._error_fmt.setForeground(QColor("#ef9a9a"))
        self._error_fmt.setFontWeight(QFont.Bold)

    def highlightBlock(self, text: str) -> None:
        if text.startswith("🧑 You:"):
            self.setFormat(0, len(text), self._user_fmt)
        elif text.startswith("🤖 Assistant:"):
            self.setFormat(0, len(text), self._assistant_fmt)
        elif text.startswith("💭 Thinking:"):
            self.setFormat(0, len(text), self._tool_fmt)
        elif text.startswith("📞"):
            self.setFormat(0, len(text), self._tool_fmt)
        elif text.startswith("   ↳"):
            self.setFormat(0, len(text), self._assistant_fmt)
        elif text.startswith("❌"):
            self.setFormat(0, len(text), self._error_fmt)


# ============================================================================
# Main window
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("State-Projection Agent Loop — GUI")
        self.resize(1280, 820)

        # Session state
        self.session: Optional[Session] = None
        self.worker: Optional[SessionWorker] = None
        self._scenario_backend: Any = None  # SupportBackend / MediaLog / temp dir
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

        # Build UI
        self._build_ui()
        self._apply_dark_theme()

        # Initially no session loaded
        self._set_session_ui_enabled(False)

    # -- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ---- Top bar: scenario selector + controls ----
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        top_bar.addWidget(QLabel("Scenario:"))

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("🎮 Game Master (TRPG)", "game_master")
        self.scenario_combo.addItem("📞 Customer Support", "customer_support")
        self.scenario_combo.addItem("💻 Coding Agent", "coding_agent")
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        top_bar.addWidget(self.scenario_combo)

        self.start_btn = QPushButton("▶ Start Session")
        self.start_btn.clicked.connect(self._start_session)
        top_bar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_session)
        top_bar.addWidget(self.stop_btn)

        self.status_label = QLabel("Select a scenario and click Start.")
        top_bar.addWidget(self.status_label, 1)

        root_layout.addLayout(top_bar)

        # ---- Main splitter: conversation | info panel ----
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        # LEFT: conversation panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.conversation_view = QTextEdit()
        self.conversation_view.setReadOnly(True)
        self.conversation_view.setFont(QFont("Segoe UI, Meiryo, sans-serif", 10))
        self._highlighter = ConversationHighlighter(self.conversation_view.document())
        left_layout.addWidget(self.conversation_view, 1)

        # Input row
        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here... (Enter to send)")
        self.input_field.returnPressed.connect(self._send_message)
        input_row.addWidget(self.input_field, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self.send_btn)

        left_layout.addLayout(input_row)
        splitter.addWidget(left_panel)

        # RIGHT: info panel (tabs)
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(380)

        # -- Tab 0: System Prompt --
        self.kernel_view = QTextEdit()
        self.kernel_view.setReadOnly(True)
        self.kernel_view.setFont(QFont("Segoe UI, Meiryo, monospace", 9))
        right_panel.addTab(self.kernel_view, "📜 System Prompt")

        # -- Tab 1: Tool Registry --
        tools_container = QWidget()
        tools_layout = QVBoxLayout(tools_container)
        tools_layout.setContentsMargins(0, 0, 0, 0)

        # TOC label (category summary)
        self.toc_label = QLabel("")
        self.toc_label.setWordWrap(True)
        self.toc_label.setFont(QFont("Segoe UI, Meiryo, sans-serif", 9))
        self.toc_label.setStyleSheet("padding: 4px; background: #1e1e2e; border-radius: 4px;")
        tools_layout.addWidget(self.toc_label)

        self.tool_tree = QTreeWidget()
        self.tool_tree.setHeaderLabels(["Tool / Category", "Signature", "Pinned"])
        self.tool_tree.setFont(QFont("Segoe UI, Meiryo, monospace", 9))
        self.tool_tree.setAlternatingRowColors(True)
        self.tool_tree.setIndentation(16)
        self.tool_tree.header().setStretchLastSection(False)
        self.tool_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tool_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tool_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tools_layout.addWidget(self.tool_tree, 1)

        right_panel.addTab(tools_container, "🔧 Tools")

        # -- Tab 2: State --
        self.state_view = QTextEdit()
        self.state_view.setReadOnly(True)
        self.state_view.setFont(QFont("Segoe UI, Meiryo, monospace", 10))
        right_panel.addTab(self.state_view, "📊 State")

        # -- Tab 3: Log / Metrics --
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Segoe UI, Meiryo, monospace", 9))
        right_panel.addTab(self.log_view, "📋 Log")

        splitter.addWidget(right_panel)
        splitter.setSizes([700, 500])

    # -- Dark theme ----------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyle("Fusion")
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#1a1a2e"))
        palette.setColor(palette.ColorRole.WindowText, QColor("#cdd6f4"))
        palette.setColor(palette.ColorRole.Base, QColor("#181825"))
        palette.setColor(palette.ColorRole.AlternateBase, QColor("#1e1e2e"))
        palette.setColor(palette.ColorRole.ToolTipBase, QColor("#313244"))
        palette.setColor(palette.ColorRole.ToolTipText, QColor("#cdd6f4"))
        palette.setColor(palette.ColorRole.Text, QColor("#cdd6f4"))
        palette.setColor(palette.ColorRole.Button, QColor("#313244"))
        palette.setColor(palette.ColorRole.ButtonText, QColor("#cdd6f4"))
        palette.setColor(palette.ColorRole.Highlight, QColor("#89b4fa"))
        palette.setColor(palette.ColorRole.HighlightedText, QColor("#1e1e2e"))
        app.setPalette(palette)

        app.setStyleSheet("""
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                color: #cdd6f4;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #89b4fa;
            }
            QPushButton {
                background: #45475a;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px 14px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background: #585b70;
            }
            QPushButton:disabled {
                background: #313244;
                color: #6c7086;
            }
            QLineEdit, QTextEdit, QTreeWidget, QComboBox {
                background: #181825;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #1e1e2e;
                selection-background-color: #45475a;
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
                background: #181825;
            }
            QTabBar::tab {
                background: #313244;
                border: 1px solid #45475a;
                padding: 4px 12px;
                color: #a6adc8;
            }
            QTabBar::tab:selected {
                background: #45475a;
                color: #cdd6f4;
            }
            QHeaderView::section {
                background: #313244;
                border: 1px solid #45475a;
                padding: 4px;
                color: #cdd6f4;
            }
            QTreeWidget::item:alternate {
                background: #1e1e2e;
            }
            QSplitter::handle {
                background: #45475a;
                width: 2px;
            }
        """)

    # -- Session lifecycle ---------------------------------------------------

    def _set_session_ui_enabled(self, enabled: bool) -> None:
        self.input_field.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.start_btn.setEnabled(not enabled)
        self.stop_btn.setEnabled(enabled)
        self.scenario_combo.setEnabled(not enabled)

    @Slot()
    def _start_session(self) -> None:
        scenario = self.scenario_combo.currentData()
        self._append_conversation("system", f"Starting scenario: {self.scenario_combo.currentText()}\n")

        try:
            if scenario == "game_master":
                self._init_game_master()
            elif scenario == "customer_support":
                self._init_customer_support()
            elif scenario == "coding_agent":
                self._init_coding_agent()
            else:
                self._append_conversation("error", f"Unknown scenario: {scenario}")
                return
        except Exception as exc:
            self._append_conversation("error", f"Failed to initialise session: {exc}")
            self.status_label.setText(f"Error: {exc}")
            return

        self._set_session_ui_enabled(True)
        self.status_label.setText("Session active — type a message and press Enter.")
        self._refresh_tool_display()
        self._refresh_state_display()

    @Slot()
    def _stop_session(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.interrupt()
            self.worker.wait(3000)
        self.session = None
        self._scenario_backend = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._set_session_ui_enabled(False)
        self.status_label.setText("Session stopped. Select a scenario to start again.")
        self._append_conversation("system", "⏹ Session stopped.\n")

    def _init_game_master(self) -> None:
        log = MediaLog()
        registry = build_game_registry(log)
        self.session = Session(
            DeepSeekAdapter(temperature=0.8),
            kernel=GM_KERNEL,
            registry=registry,
            seed=initial_seed(),
        )
        install_state(self.session)
        self._scenario_backend = log
        self._log(f"Game Master session created. {len(list(registry))} tools registered.")

    def _init_customer_support(self) -> None:
        backend = SupportBackend()
        registry = build_support_registry(backend)
        self.session = Session(
            DeepSeekAdapter(),
            kernel=SUPPORT_KERNEL,
            registry=registry,
        )
        self._scenario_backend = backend
        self._log(f"Customer Support session created. {len(list(registry))} tools registered.")

    def _init_coding_agent(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        seed_workspace(root)
        registry = build_coding_registry(root)
        self.session = Session(
            DeepSeekAdapter(),
            kernel=CODING_KERNEL,
            registry=registry,
        )
        self._scenario_backend = root
        self._log(f"Coding Agent session created. Workspace: {root}")
        self._log(f"Seeded: calculator.py, test_calculator.py")

    # -- Message sending -----------------------------------------------------

    @Slot()
    def _send_message(self) -> None:
        if self.session is None:
            return
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._append_conversation("user", text)

        self.worker = SessionWorker(self.session, text)
        self.worker.reply_ready.connect(self._on_reply)
        self.worker.assistant_thought.connect(self._on_assistant_thought)
        self.worker.tool_called.connect(self._on_tool_call)
        self.worker.state_updated.connect(self._on_state_update)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    @Slot(str)
    def _on_reply(self, text: str) -> None:
        self._append_conversation("assistant", text)

    @Slot(str)
    def _on_assistant_thought(self, text: str) -> None:
        """Intermediate assistant text emitted before tool calls."""
        if text.strip():
            self._append_conversation("assistant_thought", text)

    @Slot(str, str, str)
    def _on_tool_call(self, name: str, args_json: str, observation: str) -> None:
        if args_json and observation:
            # Tool call with args and observation together
            self._append_conversation("tool", f"📞 {name}({args_json})\n   ↳ {observation[:600]}")
        elif args_json:
            # Tool call (pending execution)
            self._append_conversation("tool", f"📞 Calling: {name}({args_json})")
        elif observation:
            # Tool observation (result)
            self._append_conversation("tool_obs", f"   ↳ {name}: {observation[:600]}")

    @Slot(dict)
    def _on_state_update(self, state: dict) -> None:
        self._refresh_state_display()
        self._refresh_tool_display()

    @Slot(str)
    def _on_error(self, error_text: str) -> None:
        self._append_conversation("error", error_text)

    @Slot()
    def _on_worker_finished(self) -> None:
        self.worker = None
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_field.setFocus()

        # Update backend-specific displays
        self._refresh_backend_display()

    # -- Conversation rendering ----------------------------------------------

    def _append_conversation(self, role: str, text: str) -> None:
        if role == "user":
            prefix = "\n🧑 You: "
        elif role == "assistant":
            prefix = "\n🤖 Assistant: "
        elif role == "assistant_thought":
            prefix = "\n💭 Thinking: "
        elif role == "tool":
            prefix = "\n🔧 "
        elif role == "tool_obs":
            prefix = "   ↳ "
        elif role == "system":
            prefix = ""
        elif role == "error":
            prefix = "\n❌ "
        else:
            prefix = "\n"

        self.conversation_view.moveCursor(QTextCursor.End)
        self.conversation_view.insertPlainText(f"{prefix}{text}\n")
        # Auto-scroll to bottom
        scrollbar = self.conversation_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # -- Right panel: tool display -------------------------------------------

    def _refresh_tool_display(self) -> None:
        self.tool_tree.clear()
        if self.session is None:
            self.toc_label.setText("(no session)")
            return

        registry = self.session.registry
        toc = registry.toc_text()
        self.toc_label.setText(
            f"📑 Tool Index:  {toc}\n"
            f"   (Np = all N are pinned/always-available; N, Mp = M of N pinned)"
        )

        # Group tools by category using categories_with_pinned()
        cat_info = registry.categories_with_pinned()
        cat_items: dict[str, QTreeWidgetItem] = {}
        for tool in sorted(registry.all(), key=lambda t: (t.category or "misc", t.name)):
            cat = tool.category or "misc"
            if cat not in cat_items:
                total, pinned_count = cat_info.get(cat, (0, 0))
                if pinned_count == total and total > 0:
                    label = f"{cat}  ({total} tools, all {total} pinned 📍)"
                elif pinned_count > 0:
                    label = f"{cat}  ({total} tools, {pinned_count} pinned 📍)"
                else:
                    label = f"{cat}  ({total} tools)"
                cat_item = QTreeWidgetItem([label, "", ""])
                cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
                font = cat_item.font(0)
                font.setBold(True)
                cat_item.setFont(0, font)
                cat_item.setForeground(0, QColor("#89b4fa"))
                self.tool_tree.addTopLevelItem(cat_item)
                cat_items[cat] = cat_item
            else:
                cat_item = cat_items[cat]

            pinned_mark = "📍 pinned" if tool.discovery.pinned else ""
            sig = tool.card.signature if tool.card.signature else tool.name
            tool_item = QTreeWidgetItem([f"  {tool.name}", sig, pinned_mark])
            tool_item.setToolTip(0, tool.spec.description or tool.card.summary)
            if tool.discovery.pinned:
                tool_item.setForeground(0, QColor("#f9e2af"))
            cat_item.addChild(tool_item)

        self.tool_tree.expandAll()

    # -- Right panel: state display ------------------------------------------

    def _refresh_state_display(self) -> None:
        if self.session is None:
            self.state_view.setPlainText("(no session)")
            return
        state = self.session.state
        if not state:
            self.state_view.setPlainText("(empty state)")
            return
        self.state_view.setPlainText(json.dumps(state, ensure_ascii=False, indent=2))

    # -- Right panel: backend-specific metrics -------------------------------

    def _refresh_backend_display(self) -> None:
        # Update system prompt tab
        if self.session is not None:
            kernel_sec = self.session.projection.get("kernel")
            if kernel_sec is not None and hasattr(kernel_sec, "_messages"):
                msgs = getattr(kernel_sec, "_messages", [])
                if msgs:
                    self.kernel_view.setPlainText(msgs[0].text() if hasattr(msgs[0], "text") else str(msgs[0]))

        # Update log with backend info
        backend = self._scenario_backend
        if backend is None:
            return

        if isinstance(backend, MediaLog):
            lines = ["--- Media Log ---"]
            if backend.bgm:
                lines.append(f"BGM tracks: {', '.join(backend.bgm)}")
            if backend.images:
                lines.append(f"Scenes: {', '.join(backend.images)}")
            if backend.expressions:
                lines.append(f"Expression changes: {len(backend.expressions)}")
            if backend.dice:
                lines.append(f"Dice rolls: {len(backend.dice)}")
            self._log("\n".join(lines))

        elif isinstance(backend, SupportBackend):
            lines = ["--- Support Backend ---"]
            if backend.tickets:
                lines.append(f"Escalated tickets: {len(backend.tickets)}")
                for t in backend.tickets:
                    lines.append(f"  {t['id']}: {t.get('problem_summary', '')[:80]}")
            if backend.charts:
                lines.append(f"Chart cards: {len(backend.charts)}")
            lines.append(f"Metrics: {json.dumps(backend.metrics, ensure_ascii=False)}")
            self._log("\n".join(lines))

        elif isinstance(backend, Path):
            # Coding agent workspace
            files = sorted(backend.rglob("*"))
            lines = ["--- Workspace Files ---"]
            for f in files:
                if f.is_file():
                    try:
                        content = f.read_text(encoding="utf-8")
                        lines.append(f"\n{'='*40}\n📄 {f.name}\n{'='*40}\n{content}")
                    except Exception:
                        lines.append(f"\n📄 {f.name} (binary)")
            self._log("\n".join(lines))

    # -- Log helper ----------------------------------------------------------

    def _log(self, text: str) -> None:
        self.log_view.append(text)

    # -- Scenario change (preview kernel before starting) --------------------

    @Slot()
    def _on_scenario_changed(self) -> None:
        scenario = self.scenario_combo.currentData()
        kernels = {
            "game_master": GM_KERNEL,
            "customer_support": SUPPORT_KERNEL,
            "coding_agent": CODING_KERNEL,
        }
        kernel_text = kernels.get(scenario, "")
        self.kernel_view.setPlainText(kernel_text)
        self._populate_tool_tree_preview(scenario)

    @staticmethod
    def _build_preview_registry(scenario: str) -> "Registry":
        """Build a throwaway registry for preview purposes."""
        if scenario == "game_master":
            log = MediaLog()
            return build_game_registry(log)
        elif scenario == "customer_support":
            backend = SupportBackend()
            return build_support_registry(backend)
        elif scenario == "coding_agent":
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                seed_workspace(root)
                return build_coding_registry(root)
        raise ValueError(f"Unknown scenario: {scenario}")

    def _populate_tool_tree_preview(self, scenario: str) -> None:
        self.tool_tree.clear()
        try:
            reg = self._build_preview_registry(scenario)
        except Exception:
            self.toc_label.setText("(preview unavailable)")
            return

        toc = reg.toc_text()
        self.toc_label.setText(
            f"📑 Tool Index (preview):  {toc}\n"
            f"   (Np = all N are pinned/always-available; N, Mp = M of N pinned)"
        )

        cat_info = reg.categories_with_pinned()
        cat_items: dict[str, QTreeWidgetItem] = {}
        for tool in sorted(reg.all(), key=lambda t: (t.category or "misc", t.name)):
            cat = tool.category or "misc"
            if cat not in cat_items:
                total, pinned_count = cat_info.get(cat, (0, 0))
                if pinned_count == total and total > 0:
                    label = f"{cat}  ({total} tools, all {total} pinned 📍)"
                elif pinned_count > 0:
                    label = f"{cat}  ({total} tools, {pinned_count} pinned 📍)"
                else:
                    label = f"{cat}  ({total} tools)"
                cat_item = QTreeWidgetItem([label, "", ""])
                cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
                font = cat_item.font(0)
                font.setBold(True)
                cat_item.setFont(0, font)
                cat_item.setForeground(0, QColor("#89b4fa"))
                self.tool_tree.addTopLevelItem(cat_item)
                cat_items[cat] = cat_item
            else:
                cat_item = cat_items[cat]

            pinned_mark = "📍 pinned" if tool.discovery.pinned else ""
            sig = tool.card.signature if tool.card.signature else tool.name
            tool_item = QTreeWidgetItem([f"  {tool.name}", sig, pinned_mark])
            tool_item.setToolTip(0, tool.spec.description or tool.card.summary)
            if tool.discovery.pinned:
                tool_item.setForeground(0, QColor("#f9e2af"))
            cat_item.addChild(tool_item)

        self.tool_tree.expandAll()


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("State-Projection Agent Loop GUI")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
