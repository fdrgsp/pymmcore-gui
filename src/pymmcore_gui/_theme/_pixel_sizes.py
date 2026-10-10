"""Make hard-coded pixel sizes follow zoom.

Zoom (``set_zoom``) scales the application font and the style's pixel
metrics, which is all a widget sized by its layout needs. Widgets that pin a
size in pixels -- ``setFixedSize(38, 38)``, ``setFixedWidth(110)``,
``setIconSize(QSize(28, 28))``, ``setMinimumSize(...)`` -- keep those pixels at
every zoom, which is how pymmcore-widgets' stage control ends up with tiny
buttons next to zoomed text. Rather than patch every such widget (most of them
upstream), :func:`rescale_pixel_sizes` scales each widget's explicit sizes by
the zoom ratio.

Every widget remembers the size it had at the zoom it was recorded at, so
scaling is always computed from that original rather than compounding
rounding over repeated zoom steps. A widget that resizes itself in the
meantime (e.g. re-applying ``theme().scaled(...)`` on ``StyleChange``) no
longer matches what was last applied here, and is simply re-recorded at the
zoom that was current when it did.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pymmcore_gui._qt.QtCore import QSize
from pymmcore_gui._qt.QtWidgets import QAbstractButton, QApplication

if TYPE_CHECKING:
    from pymmcore_gui._qt.QtWidgets import QWidget

# Qt's QWIDGETSIZE_MAX is 16777215, but some widgets cap at large sentinels of
# their own (524287 is common), so anything this big means "unbounded".
_UNBOUNDED = 100_000
_RECORD = "pmgZoomPixelSizes"
# Qt-Advanced-Docking-System chrome is sized by its own stylesheet
# (dock_chrome_stylesheet), which already follows zoom.
_ADS_MODULES = ("PyQt6Ads", "PySide6QtAds")

# (min w, min h, max w, max h, icon w, icon h); icon is -1 for non-buttons
_Sizes = tuple[int, ...]


def rescale_pixel_sizes(old_zoom: float, new_zoom: float) -> None:
    """Scale every widget's explicit pixel sizes from *old_zoom* to *new_zoom*.

    Must run before the new zoom reaches the style: a button's default icon
    size comes from the style, and has to be read at the zoom it belongs to.
    """
    if old_zoom <= 0 or new_zoom == old_zoom:
        return
    if not isinstance(app := QApplication.instance(), QApplication):
        return
    for w in app.allWidgets():
        if is_self_sized(w):
            continue
        # a widget may be deleted on the C++ side mid-pass
        with suppress(RuntimeError):
            _rescale(w, old_zoom, new_zoom)


def is_self_sized(w: QWidget) -> bool:
    """Whether *w*'s sizes are managed by Qt or ADS rather than hard-coded.

    Qt's own internal widgets (named ``qt_*``, e.g. a scroll area's scrollbar
    container) are re-sized from style metrics on every layout pass, and
    ADS chrome from its stylesheet.
    """
    return (
        w.objectName().startswith("qt_")
        or type(w).__module__.split(".")[0] in _ADS_MODULES
    )


def _rescale(w: QWidget, old_zoom: float, new_zoom: float) -> None:
    current = _sizes(w)
    record = _read_record(w)
    if record is not None and record[2] == current:
        zoom, base = record[0], record[1]
    else:  # never seen, or resized itself since: its current sizes are its own
        zoom, base = old_zoom, current
    if not any(_scalable(v) for v in base):
        return
    ratio = new_zoom / zoom
    target = tuple(round(v * ratio) if _scalable(v) else v for v in base)
    if target != current:
        _apply(w, current, target)
    w.setProperty(_RECORD, _encode(zoom, base, _sizes(w)))


def _scalable(v: int) -> bool:
    return 0 < v < _UNBOUNDED


def _sizes(w: QWidget) -> _Sizes:
    mn, mx = w.minimumSize(), w.maximumSize()
    if isinstance(w, QAbstractButton):
        icon = w.iconSize()
        iw, ih = icon.width(), icon.height()
    else:
        iw = ih = -1
    return (mn.width(), mn.height(), mx.width(), mx.height(), iw, ih)


def _apply(w: QWidget, current: _Sizes, target: _Sizes) -> None:
    if current[:4] != target[:4]:
        # Growing a fixed size means raising the maximum before the minimum,
        # and shrinking the reverse: lift the cap first so neither order
        # passes through an invalid min > max state.
        w.setMaximumSize(16777215, 16777215)
        w.setMinimumSize(target[0], target[1])
        w.setMaximumSize(target[2], target[3])
    if isinstance(w, QAbstractButton) and current[4:] != target[4:]:
        w.setIconSize(QSize(target[4], target[5]))


def _encode(zoom: float, base: _Sizes, applied: _Sizes) -> str:
    # a plain string, because PyQt6 and PySide6 round-trip Python tuples
    # through dynamic properties differently
    return f"{zoom!r}|{','.join(map(str, base))}|{','.join(map(str, applied))}"


def _read_record(w: QWidget) -> tuple[float, _Sizes, _Sizes] | None:
    if not isinstance(raw := w.property(_RECORD), str):
        return None
    try:
        zoom, base, applied = raw.split("|")
        return (
            float(zoom),
            tuple(int(v) for v in base.split(",")),
            tuple(int(v) for v in applied.split(",")),
        )
    except ValueError:  # pragma: no cover - only ever written by _encode
        return None
