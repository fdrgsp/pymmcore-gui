"""Installation tab: manage the Micro-Manager installs this session can use.

Wraps :class:`pymmcore_widgets.InstallWidget` -- the download / activate /
uninstall table -- in the modern page shell. The page only *reports* that the
active installation changed (:attr:`InstallationPage.activeInstallChanged`);
acting on that (unloading devices, repointing the running core) belongs to the
main window, which owns the core and the pages whose contents go stale.
"""

from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pymmcore_plus import find_micromanager
from superqt.iconify import QIconifyIcon

from pymmcore_gui._array_viewer import set_source_icon, unstyle_widgets
from pymmcore_gui._qt.QtCore import QEvent, QSize, Qt, QTimer, Signal
from pymmcore_gui._qt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._busy import BusyOverlay, busy
from ._tab_page import TabPage
from ._theme import qcolor, theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymmcore_widgets import InstallWidget

    from pymmcore_gui._qt.QtGui import QResizeEvent, QShowEvent

LOADING_MSG: Final = "Looking for Micro-Manager releases…"

# label, icon, button variant, handler on this page, InstallWidget QAction gating it
#
# "primary" for the action buttons and "danger" for the destructive one is what
# the other tab-page toolbars do (Hardware Setup's New/Load/Save, Configurations'
# two Save buttons) -- leaving only Install primary made it the odd one out.
_ACTIONS: Final = (
    ("Refresh", "material-symbols:refresh-rounded", "primary", "_refresh", ""),
    ("Reveal", "material-symbols:search-rounded", "primary", "_reveal", "_act_reveal"),
    (
        "Set Active",
        "material-symbols:check-rounded",
        "primary",
        "_set_active",
        "_act_use",
    ),
    (
        "Install",
        "material-symbols:install-desktop-rounded",
        "primary",
        "_install",
        "",
    ),
    (
        "Uninstall",
        "material-symbols:delete-forever-outline-rounded",
        "danger",
        "_uninstall",
        "_act_uninstall",
    ),
)


def _forget_deleted_installs() -> None:
    """Drop no-longer-existing installs from pymmcore-plus's discovery cache.

    ``find_micromanager(return_first=False)`` returns the module-level
    ``_DISCOVERED_MMS`` dict rather than what the current scan actually found,
    and nothing ever evicts from it -- so an install stays discoverable for the
    rest of the process after being deleted, and a table re-listing itself puts
    the row it just removed straight back.

    Guarded, and reaching into a private name: the day pymmcore-plus stops
    needing this, it should quietly become a no-op rather than a crash.
    """
    with suppress(Exception):
        from pymmcore_plus import _discovery

        cache = _discovery._DISCOVERED_MMS
        for path in [p for p in cache if not p.is_dir()]:
            del cache[path]


def _test_adapters_note() -> str:
    """Upstream's warning that this platform only gets the test device adapters.

    Empty where full Micro-Manager nightly builds exist (Windows, Intel macOS).
    """
    with suppress(Exception):
        from pymmcore_widgets import _install_widget

        if not getattr(_install_widget, "FULL_RELEASES", True):
            return str(getattr(_install_widget, "TEST_ADAPTERS_NOTE", ""))
    return ""


class InstallReleaseDialog(QDialog):
    """Choose a Micro-Manager release to download.

    Shaped like the Group Editor's "Edit Properties" sheet -- modal and
    frameless, content over a button box -- so the two read as the same kind
    of decision.
    """

    def __init__(
        self,
        releases: Sequence[str],
        *,
        note: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Sheet
            | Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.FramelessWindowHint,
        )
        self.setWindowTitle("Install Micro-Manager")
        self.setModal(True)
        t = theme()

        self._releases = QComboBox(self)
        self._releases.addItems(list(releases))

        row = QHBoxLayout()
        row.setSpacing(t.sp_sm)
        row.addWidget(QLabel("Release:"), 0)
        row.addWidget(self._releases, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        install = buttons.addButton("Install", QDialogButtonBox.ButtonRole.AcceptRole)
        if install is not None:
            install.setProperty("variant", "primary")
            install.setDefault(True)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel is not None:
            cancel.setProperty("variant", "subtle")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.sp_lg, t.sp_lg, t.sp_lg, t.sp_lg)
        layout.setSpacing(t.sp_md)
        layout.addLayout(row)
        if note:
            explainer = QLabel(note)
            explainer.setWordWrap(True)
            font = explainer.font()
            font.setItalic(True)
            explainer.setFont(font)
            layout.addWidget(explainer)
        layout.addWidget(buttons)
        self.setMinimumWidth(t.scaled(460))

    @property
    def release(self) -> str:
        """The release the user chose."""
        return self._releases.currentText()


class InstallationPage(TabPage):
    """Download, activate and remove Micro-Manager device-adapter installs.

    The wrapped ``InstallWidget`` is built on first show rather than in
    ``__init__``: its constructor fetches the list of downloadable releases
    over the network, and ctypes-loads every installed ``mmgr_dal`` library to
    read its device interface version. Doing that while the main window is
    being constructed would add seconds to every launch -- and stall until the
    request times out when offline -- for a page most sessions never open.
    """

    activeInstallChanged = Signal(str)
    """Emitted with the new path when ``find_micromanager()`` resolves elsewhere.

    Empty when the last install was removed and there is nothing left to point
    the core at.
    """

    aboutToUninstall = Signal(set)
    """Emitted with the selected paths right before ``_uninstall`` deletes them.

    MainWindow (which owns the core) connects this to release any device
    adapter DLL one of these paths is about to delete out from under a still-
    loaded device -- see the note on ``_uninstall``. A signal rather than a
    plain callable attribute so the connection's lifetime is Qt's to manage
    (tied to both endpoints' C++ objects) instead of a bound method sitting in
    a Python attribute, which -- since ``MainWindow`` back-references this
    page as one of its own attributes -- would otherwise close a reference
    cycle over ``self`` that only the GC's (non-deterministic) cycle collector
    breaks, not the deterministic ``deleteLater``/``close`` teardown every
    other cross-page wire-up here relies on.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.left.hide()

        self._widget: InstallWidget | None = None
        self._active_install = find_micromanager(return_first=True) or ""
        self._overlay = BusyOverlay(self)

        # Repeated starts of a zero-interval single-shot timer collapse into
        # one call, so a refresh() -- which clears and refills the table row by
        # row -- triggers a single check rather than one per signal.
        self._active_check = QTimer(self)
        self._active_check.setSingleShot(True)
        self._active_check.setInterval(0)
        self._active_check.timeout.connect(self._check_active_install)

        # Mirrors of the wrapped widget's own toolbar actions (hidden below),
        # themed like the other pages' toolbars. They stay disabled until the
        # widget exists, and afterwards follow the enabled state of the actions
        # they stand in for, so the upstream selection rules remain the only
        # place those rules live.
        self._buttons: dict[str, QPushButton] = {}
        for label, _icon, variant, handler, _action in _ACTIONS:
            btn = QPushButton(label)
            btn.setProperty("variant", variant)
            btn.setEnabled(False)
            btn.clicked.connect(getattr(self, handler))
            self.toolbar.add_widget(btn)
            self._buttons[label] = btn
        self.toolbar.add_stretch()
        self._apply_button_icons()

        # Replaced by the real widget once it loads; also carries the error if
        # the release listing can't be fetched.
        self._placeholder = QLabel(LOADING_MSG)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self.add_content_widget(self._placeholder)

    # ── lazy construction ─────────────────────────────────────────

    @property
    def install_widget(self) -> InstallWidget | None:
        """The wrapped widget, or None while it hasn't been built."""
        return self._widget

    def ensure_loaded(self) -> InstallWidget | None:
        """Build the wrapped widget if it isn't built yet.

        Returns None if the release listing couldn't be fetched (typically no
        network); the placeholder then shows why and Refresh retries.
        """
        if self._widget is not None:
            return self._widget

        from pymmcore_widgets import InstallWidget

        # No busy overlay here: the placeholder *is* the whole page at this
        # point, so the overlay would only print the same sentence a second
        # time -- centered on the page rather than on the content area, and
        # dimming the first copy through its own backdrop. Repainting the
        # label instead gets the message on screen before the constructor
        # below blocks the event loop fetching the release listing (it also
        # covers a Refresh retry, where the label still holds the last error).
        self._placeholder.setText(LOADING_MSG)
        self._placeholder.repaint()

        try:
            widget = InstallWidget(self)
        except Exception as e:  # pragma: no cover - needs a dead network
            self._placeholder.setText(
                "Could not reach the Micro-Manager download listing:\n\n"
                f"{e}\n\nCheck your connection and press Refresh."
            )
            # Refresh is the retry, so it must not stay disabled.
            self._buttons["Refresh"].setEnabled(True)
            return None

        # Its QToolBar duplicates the page toolbar built above, and its inline
        # "Install release:" row (with the note explaining it) is replaced by
        # this page's Install button and its dialog.
        widget.toolbar.hide()
        for attr in ("install_row", "install_note"):
            if (replaced := getattr(widget, attr, None)) is not None:
                replaced.hide()
        unstyle_widgets(widget)
        # unstyle_widgets makes every button "subtle"; downloading a release is
        # this page's headline action.
        widget.install_btn.setProperty("variant", "primary")
        self._placeholder.hide()
        self.add_content_widget(widget)
        self._widget = widget

        # _InstallTable.keyPressEvent calls self.uninstall() on Delete/Backspace;
        # point that at ours too, so the keyboard can't reach the version this
        # page deliberately replaces (see _uninstall).
        widget.table.uninstall = self._uninstall  # type: ignore[method-assign]

        widget.table.itemSelectionChanged.connect(self._sync_buttons)
        widget.installFinished.connect(self._on_install_finished)
        # Every path that can change which install is active -- these buttons,
        # the table's own Delete-key uninstall, and a finished download --
        # ends in table.refresh(), i.e. in rows changing.
        if (model := widget.table.model()) is not None:
            model.rowsInserted.connect(self._active_check.start)
            model.rowsRemoved.connect(self._active_check.start)
        self._sync_buttons()
        return widget

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        if self._widget is None:
            # Deferred by a tick so the page (and the busy overlay) paint
            # before the blocking network fetch starts.
            QTimer.singleShot(0, self.ensure_loaded)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._overlay.setGeometry(self.rect())

    def _apply_button_icons(self) -> None:
        default_color = qcolor(theme().text_secondary).name()
        danger_color = qcolor(theme().status_red).name()
        size = theme().scaled(16)
        for label, icon, variant, _handler, _action in _ACTIONS:
            color = danger_color if variant == "danger" else default_color
            btn = self._buttons[label]
            set_source_icon(btn, QIconifyIcon(icon, color=color))
            btn.setIconSize(QSize(size, size))

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() == QEvent.Type.StyleChange:
            self._apply_button_icons()
        super().changeEvent(a0)

    # ── toolbar ───────────────────────────────────────────────────

    def _run(self, method: str) -> None:
        """Invoke a table action by name, building the widget first if needed."""
        if (widget := self.ensure_loaded()) is None:
            return
        getattr(widget.table, method)()

    def _refresh(self) -> None:
        self._run("refresh")

    def _reveal(self) -> None:
        self._run("reveal")

    def _set_active(self) -> None:
        self._run("set_active")

    def _install(self) -> None:
        """Choose a release in a dialog, then hand it to ``mmcore install``.

        The wrapped widget owns the machinery -- the subprocess thread, the
        streamed output box, the table refresh when it finishes -- so this only
        decides *what* to install, which is the part its hidden inline row used
        to do. While a download runs the button turns into Cancel, matching
        what that row did.
        """
        if (widget := self.ensure_loaded()) is None:
            return
        if widget.is_installing:
            widget.cancel_install()
            return

        combo = widget.version_combo
        releases = [combo.itemText(i) for i in range(combo.count())]
        dialog = InstallReleaseDialog(releases, note=_test_adapters_note(), parent=self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            release = dialog.release
        finally:
            dialog.deleteLater()

        widget.install(release)
        if widget.is_installing:
            self._buttons["Install"].setText("Cancel")

    def _on_install_finished(self, _returncode: int) -> None:
        """Put the Install button back once the download ends (or is cancelled)."""
        with suppress(RuntimeError):  # the page may already be torn down
            self._buttons["Install"].setText("Install")

    def _uninstall(self) -> None:
        """Delete the selected installs, reporting whatever couldn't be removed.

        Deliberately not delegated to the wrapped table's own uninstall, which
        deletes with ``ignore_errors=True`` -- an install that could *not* be
        removed (owned by another user, or with adapters still mapped into a
        running process on Windows) then looks exactly like one that was -- and
        which re-lists from a discovery cache still holding what it just
        deleted, so the row reappears as if nothing happened.

        ``aboutToUninstall`` fires after the user confirms but before any
        deletion, so a device adapter DLL the running core still has loaded
        from one of these paths gets released first -- otherwise Windows
        keeps it locked and every path sharing that install fails with
        ``PermissionError: [WinError 5] Access is denied``.
        """
        if (widget := self.ensure_loaded()) is None:
            return

        from pymmcore_widgets._install_widget import LOC_ROLE

        table = widget.table
        paths: set[str] = set()
        for index in table.selectedIndexes():
            if (item := table.item(index.row(), table.LOC_COL)) is not None:
                if path := item.data(LOC_ROLE):
                    paths.add(str(path))
        if not paths:
            return

        listing = "\n".join(f"  • {p}" for p in sorted(paths))
        plural = "s" if len(paths) > 1 else ""
        if (
            QMessageBox.question(
                self,
                "Uninstall",
                f"Delete the following Micro-Manager installation{plural}?\n\n"
                f"{listing}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        failures: dict[str, str] = {}
        with busy(self._overlay, f"Removing {len(paths)} installation{plural}…"):
            self.aboutToUninstall.emit(paths)
            for path in sorted(paths):
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    failures[path] = str(e)
            _forget_deleted_installs()
            table.refresh()

        if failures:
            reasons = "\n".join(f"  • {Path(p).name}: {e}" for p, e in failures.items())
            QMessageBox.warning(self, "Uninstall", f"Could not remove:\n\n{reasons}")

    def _sync_buttons(self) -> None:
        """Follow the enabled state of the actions these buttons stand in for."""
        if (widget := self._widget) is None:  # pragma: no cover
            return
        for label, _icon, _variant, _method, action in _ACTIONS:
            enabled = True
            if action:
                enabled = bool(getattr(widget, action).isEnabled())
            self._buttons[label].setEnabled(enabled)

    # ── active install ────────────────────────────────────────────

    @property
    def active_install(self) -> str:
        """Path of the install ``find_micromanager()`` currently resolves to."""
        return self._active_install

    def _check_active_install(self) -> None:
        with suppress(RuntimeError):  # widget may be torn down on the C++ side
            active = find_micromanager(return_first=True) or ""
            if active != self._active_install:
                self._active_install = active
                self.activeInstallChanged.emit(active)
