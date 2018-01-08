"""
gl.py

customized widget for surface render
"""
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy

from .surface import Surface


FOV = 45


class SurfaceRenderWidget(QGLWidget):

    surf = None
    min_dist = 0
    lightpos = (0, 0, 0)

    xRot = 0
    yRot = 0

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is not None:
            self.setGeometry(parent.geometry())
            self.setSizePolicy(parent.sizePolicy())

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        glClearDepth(1.0)
        glFrontFace(GL_CW)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)

        self.resizeGL(self.width(), self.height())

        glLightfv(GL_LIGHT1, GL_AMBIENT, (.1, .1, .1, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (.5, .5, .5, 1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (.8, .8, .8, 1))
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT1)

        glDisable(GL_TEXTURE_2D)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, width / height, 10, 1000)

        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        if not self.surf:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glLightfv(GL_LIGHT1, GL_POSITION, *self.lightpos)

        # lookat
        d = self.min_dist
        horz = (self.xRot -90) /180 *numpy.pi
        vert = self.yRot /180 *numpy.pi

        gluLookAt(
            numpy.sin(horz) *d,  # eye x
            0, # eye y
            numpy.cos(horz) *d, # eye z
            # 0, 0, -self.min_dist, # eye
            0, 0, 1,    # direction
            0, 1, 0     # up
        )

        # draw
        rows, cols = self.surf.shape
        for irow in range(rows-1):
            for icol in range(cols-1):
                glBegin(GL_QUADS)
                self.drawVert(irow, icol)
                self.drawVert(irow+1, icol)
                self.drawVert(irow+1, icol+1)
                self.drawVert(irow, icol+1)
                glEnd()

    def drawVert(self, i, j):
        element, normal = self.surf[i, j]
        glNormal3fv(normal)
        glVertex3fv(element)

    def setSurface(self, surf):
        assert isinstance(surf, Surface)
        self.surf = surf

        rang = numpy.vstack([surf.xrange, surf.yrange])
        rang = rang[:,1] - rang[:,0]
        len_diag = numpy.max(rang)

        # evaluate distant
        self.min_dist = len_diag /2 /numpy.tan(FOV /360 *numpy.pi)
        self.min_dist += 30

        # evaluate light position
        self.lightpos = (
            (surf.xrange[1] - surf.xrange[0]) * -.45,
            (surf.yrange[1] - surf.yrange[0]) * .45,
            (surf.zrange[0] - surf.center[2]) * 5,
            1
        )

        # reset rotate
        self.xRot = 0
        self.yRot = 0

        self.updateGL()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.xRot -= dx /3
            self.yRot -= dy /3
            self.updateGL()

        self.lastPos = event.pos()
