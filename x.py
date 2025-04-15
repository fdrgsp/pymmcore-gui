import ndv
import useq
from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import QApplication, QWidget

from pymmcore_gui._ndv_viewers import NDVViewersManager
from pymmcore_gui.widgets.mda_widget._writers import _TensorStoreHandler, _TiffSequenceWriter, _OMETiffWriter


def _on_viewer_created(ndv_viewer: ndv.ArrayViewer, sequence: useq.MDASequence) -> None:
    """Called when a new viewer is created."""
    wdg = ndv_viewer.widget()
    wdg.show()


app = QApplication([])

main_wdg = QWidget()

mmc = CMMCorePlus()
mmc.loadSystemConfiguration()
mmc.setExposure(500)

m = NDVViewersManager(main_wdg, mmcore=mmc)
m.viewerCreated.connect(_on_viewer_created)

# h = _OMETiffWriter("/Users/fdrgsp/Desktop/test.ome.tiff")
# h = _TensorStoreHandler(path="/Users/fdrgsp/Desktop/test.zarr")

h = _TiffSequenceWriter("/Users/fdrgsp/Desktop/test")

seq = useq.MDASequence(
    axis_order="pc",
    stage_positions=[(1, 1)],
    channels=["DAPI", "FITC"],
    metadata={"hacky_handler": h},
)

mmc.run_mda(seq)

app.exec()
