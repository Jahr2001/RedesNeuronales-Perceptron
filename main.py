import sys
from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox
import random as rand

from perceptron import entrada

qtCreatorFile = "./view/mainView.ui"  # Nombre del archivo
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)



class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.definirValores.setChecked(True)

        self.setWindowTitle('.:: PERCEPTRÓN ::. 193258 - 193213')
        self.setFixedSize(800, 600)

        self.aleatorioT.toggled.connect(self.onClickRadioButton)
        self.definirValores.toggled.connect(self.onClickRadioButton)

        self.n1.setValidator(QtGui.QDoubleValidator())
        self.n2.setValidator(QtGui.QDoubleValidator())
        self.n3.setValidator(QtGui.QDoubleValidator())
        self.n4.setValidator(QtGui.QDoubleValidator())
        self.n5.setValidator(QtGui.QDoubleValidator())

        self.w1.setValidator(QtGui.QDoubleValidator())
        self.w2.setValidator(QtGui.QDoubleValidator())
        self.w3.setValidator(QtGui.QDoubleValidator())

        self.btnIniciarProceso.clicked.connect(self.inciar)
        self.btnSalir.clicked.connect(self.salir)

    def inciar(self):

        if self.onClickRadioButton() == False:
            
            n1, n2, n3, n4, n5 = self.getValuesN()
            w1, w2, w3 = self.getW()



            if not n1 or not n2 or not n3 or not n4 or not n5 or not w1 or not w2 or not w3:
                QMessageBox.warning(None, 'Campo(s) Vacio(s)', '¡Ingrese los datos que se solicitan!')

            else:

                self.lbOpcion.setText('')               
                n1 = float(self.n1.text())
                n2 = float(self.n2.text())
                n3 = float(self.n3.text())
                n4 = float(self.n4.text())
                n5 = float(self.n5.text())


                w1 = float(self.w1.text())
                w2 = float(self.w2.text())
                w3 = float(self.w3.text())

                if n1 == 0 or n2 == 0 or n3 == 0 or n4 == 0 or n5 == 0:
                    QMessageBox.warning(None, 'Valores de Ceros', 'No ingrese ceros en la taza de aprendizaje!!')

                else:
                    wk = [w1, w2, w3]
                    ns = [n1, n2, n3, n4, n5]

                
                    entrada(wk, ns)

                    wk.clear()
                    ns.clear()
        else:
            self.lbOpcion.setText('Valores generados aleatoriamente')
            
            wk = []
            ns = []
            
            for i in range(3):
                wk.append(round(rand.random(),3))

            for n in range(5):
                ns.append(round(rand.uniform(1, 0), 3))
 
            entrada(wk, ns)

            wk.clear()
            ns.clear()

       

    def getValuesN(self):
        n1 = self.n1.text()
        n2 = self.n2.text()
        n3 = self.n3.text()
        n4 = self.n4.text()
        n5 = self.n5.text()
        return n1, n2, n3, n4, n5

    def getW(self):
        w1 = self.w1.text()
        w2 = self.w2.text()
        w3 = self.w3.text()
        return w1, w2, w3
    
    def onClickRadioButton(self):

        if self.aleatorioT.isChecked():
            self.lblTazaAprendizaje.setText('')
            self.lblValoresW.setText('')
            self.lbOpcion.setText('Valores generados aleatoriamente')

            self.n1.setHidden(True)
            self.n2.setHidden(True)
            self.n3.setHidden(True)
            self.n4.setHidden(True)
            self.n5.setHidden(True)

            self.n1.clear()
            self.n2.clear()
            self.n3.clear()
            self.n4.clear()
            self.n5.clear()

            self.w1.setHidden(True)
            self.w2.setHidden(True)
            self.w3.setHidden(True)

            self.w1.clear()
            self.w2.clear()
            self.w3.clear()

            boolean = True

        elif self.definirValores.isChecked():

            self.lblTazaAprendizaje.setText('η (Taza de aprendizaje)')
            self.lblValoresW.setText('Valores de W')
            self.lbOpcion.setText('Ingresar valores')
            
            self.n1.setHidden(False)
            self.n2.setHidden(False)
            self.n3.setHidden(False)
            self.n4.setHidden(False)
            self.n5.setHidden(False)

            self.w1.setHidden(False)
            self.w2.setHidden(False)
            self.w3.setHidden(False)

            boolean = False
        
        return boolean


    def salir(self):
        salir = QMessageBox.question(
            self, 'Salir', '¿Esta seguro/a de salir?', QMessageBox.Yes, QMessageBox.No)
        if salir == QMessageBox.Yes:
            self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
