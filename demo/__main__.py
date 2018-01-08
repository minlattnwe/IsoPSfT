"""
__main__.py

enter point + ui
"""
import sys

from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .main_ui import Ui_MainWindow
from .gl import SurfaceRenderWidget

from .formula import get_formula
from .surface import TruthSurface
from .solution import solve


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    gtruth = None

    def __init__(self):
        super().__init__()
        Ui_MainWindow.setupUi(self, self)

        # override
        self.glGroundTruth = SurfaceRenderWidget(self.glGroundTruth)
        self.glResult = SurfaceRenderWidget(self.glResult)

        self.fig = Figure((8.0, 6.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.mplData)

        # event
        self.btnSolve.clicked.connect(self.on_generate_surface)

        self.sbGlSampX.valueChanged.connect(self.change_gl_resolution)
        self.sbGlSampY.valueChanged.connect(self.change_gl_resolution)

        self.show()

    def update_statusbar(self, message):
        self.statusBar().showMessage(message)

    @QtCore.pyqtSlot()
    def on_generate_surface(self):
        h = self.sbHeight.value()
        w = self.sbWidth.value()
        xrang = (-w/2, w/2)
        yrang = (-h/2, h/2)

        # create formula
        self.formula = get_formula(
            formula_x_coef=self.sbCoefX.value(),
            formula_x_power=self.sbPowX.value(),
            formula_constant=self.sbCoefC.value()
        )

        # create surface
        self.gtruth = TruthSurface(self.formula, xrang, self.sbGlSampX.value(), yrang, self.sbGlSampY.value())
        self.glGroundTruth.setSurface(self.gtruth)

        # solve
        self.solution = solve(
            fig=self.fig,
            formula=self.formula,
            xrang=xrang,
            yrang=yrang,
            xsamp=self.sbNumHorSamp.value(),
            ysamp=self.sbNumVertSamp.value(),
            nsamp=self.sbNumSample.value(),
            focal=self.sbFocal.value()
        )

        self.fig.tight_layout()
        self.canvas.draw()

        if self.solution is None:
            return
        self.solution.setNumSamp(self.sbGlSampX.value(), self.sbGlSampY.value())
        self.glResult.setSurface(self.solution)

    @QtCore.pyqtSlot()
    def change_gl_resolution(self):
        vsamp = self.sbGlSampX.value(), self.sbGlSampY.value()
        if self.gtruth:
            self.gtruth.setNumSamp(*vsamp)
            self.glGroundTruth.setSurface(self.gtruth)
        if self.solution:
            self.solution.setNumSamp(*vsamp)
            self.glResult.setSurface(self.solution)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
