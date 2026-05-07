import sys

from PyQt5.QtWidgets import QApplication

from crop_tool_modules.main_window import Tool


def main():
    app = QApplication(sys.argv)
    window = Tool()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
