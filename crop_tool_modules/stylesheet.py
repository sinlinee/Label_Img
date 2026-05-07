APP_STYLESHEET = """
QWidget {
    background: #eef2f7;
    color: #172033;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QFrame#sidebar, QFrame#canvasPanel {
    background: #ffffff;
    border: 1px solid #dde5ef;
    border-radius: 20px;
}
QLabel#appTitle {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 1px;
}
QLabel#panelTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: 800;
}
QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    background: #fbfdff;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #334155;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 9px 12px;
    color: #1e293b;
}
QPushButton:hover {
    border-color: #2563eb;
    background: #eff6ff;
}
QPushButton:checked {
    background: #1d4ed8;
    color: white;
    border-color: #1e40af;
    font-weight: 700;
}
QPushButton:pressed {
    background: #dbeafe;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 8px;
    selection-background-color: #93c5fd;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #2563eb;
    background: #f8fbff;
}
QTreeWidget {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    padding: 4px;
}
QTreeWidget::item {
    min-height: 28px;
    padding: 3px 4px;
}
QTreeWidget::item:selected {
    background: #dbeafe;
    color: #0f172a;
}
QTreeWidget QLineEdit {
    background: #ffffff;
    color: #172033;
    border: 1px solid #2563eb;
    border-radius: 4px;
    padding: 1px 4px;
    min-height: 20px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #dde5ef;
    border-radius: 18px;
    top: -1px;
}
QTabBar::tab {
    background: #e8eef7;
    border: 1px solid #d5deea;
    border-bottom: 0;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    padding: 10px 16px;
    margin-right: 4px;
    color: #475569;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
    font-weight: 800;
}
QSplitter::handle {
    background: transparent;
    width: 8px;
}
QLabel[class="hint"] {
    color: #64748b;
}
QLabel[class="cardTitle"] {
    font-size: 15px;
    font-weight: 700;
}
"""
