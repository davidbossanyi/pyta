from pyta.main import Application


def test_main_window(qtbot) -> None:
    app = Application()
    app.show()
    qtbot.addWidget(app)
