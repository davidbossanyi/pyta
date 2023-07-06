from PyQt5 import QtCore, QtWidgets


class FileDragDropListWidget(QtWidgets.QListWidget):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

    def mimeTypes(self) -> list[str]:
        mimetypes = super().mimeTypes()
        mimetypes.append("text/uri-list")
        return mimetypes

    def dropMimeData(self, index: int, data: QtCore.QMimeData, action: QtCore.Qt.DropAction) -> bool:
        if data.hasUrls():
            for url in data.urls():
                item = QtWidgets.QListWidgetItem(str(url.toLocalFile()))
                self.addItem(item)
                self.setCurrentItem(item)
            return True
        else:
            return False
