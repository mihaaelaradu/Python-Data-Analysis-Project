
import numpy as np

class ACP:

    def __init__(self, matrice):
        self.X = matrice

        # Calculul matricei de corelație pentru X nestandardizat
        self.R = np.corrcoef(self.X, rowvar=False)  # Variabilele sunt pe coloane

        # Standardizarea valorilor din X
        medii = np.mean(self.X, axis=0)
        abateri = np.std(self.X, axis=0)
        self.Xstd = (self.X - medii) / abateri

        # Calculul matricei de covarianță pentru X standardizat
        self.Cov = np.cov(self.Xstd, rowvar=False)

        # Calculul valorilor proprii și vectorilor proprii pentru matricea de covarianță
        valProp, vectProp = np.linalg.eigh(self.Cov)
        print(valProp)

        # Sortarea descrescătoare a valorilor proprii și vectorilor proprii
        k_des = [k for k in reversed(np.argsort(valProp))]
        print(k_des)
        self.alpha = valProp[k_des]
        self.a = vectProp[:, k_des]
        print(self.alpha)

        # Regularizarea vectorilor proprii
        for col in range(self.a.shape[1]):
            minim = np.min(self.a[:, col])
            maxim = np.max(self.a[:, col])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, col] *= -1

        # Calculul componentelor principale
        self.C = self.Xstd @ self.a

        # Calculul matricei factorilor de corelație (factor loadings)
        # Reprezintă corelația dintre variabilele inițiale și componentele principale
        self.Rxc = self.a * np.sqrt(self.alpha)

        # Calculul scorurilor (componentelor principale standardizate)
        self.scoruri = self.C / np.sqrt(self.alpha)

        # Calculul calității reprezentării observațiilor pe axele componentelor principale
        C2 = self.C * self.C
        C2sum = np.sum(C2, axis=1)
        self.CalObs = np.transpose(np.transpose(C2) / C2sum)

        # Contribuția observațiilor la varianța componentelor principale
        self.betha = C2 / (self.alpha * self.X.shape[0])

        # Calculul comunalităților (regăsirea componentelor principale în variabilele inițiale)
        Rxc2 = self.Rxc * self.Rxc
        self.Comun = np.cumsum(Rxc2, axis=1)

    def getCorr(self):
        return self.R

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.alpha

    def getCompPrin(self):
        return self.C

    def getRxc(self):
        return self.Rxc

    def getScoruri(self):
        return self.scoruri

    def getCalObs(self):
        return self.CalObs

    def getBetha(self):
        return self.betha

    def getComun(self):
        return self.Comun



# import numpy as np
#
#
# class ACP:
#
#     def __init__(self, matrice):
#         self.X = matrice
#
#
#
#
#         medii = np.mean(self.X, axis=0)
#         abateri = np.std(self.X, axis=0)
#         self.Xstd = (self.X - medii) / abateri
#
#
#         self.Cov = np.cov(self.Xstd, rowvar=False)
#
#         valProp, vectProp = np.linalg.eigh(self.Cov)
#         print(valProp)
#
#         k_des = [k for k in reversed(np.argsort(valProp))]
#         print(k_des)
#         self.alpha = valProp[k_des]
#         self.a = vectProp[:, k_des]
#         print(self.alpha)
#
#         for col in range(self.a.shape[1]):
#             minim = np.min(self.a[:, col])
#             maxim = np.max(self.a[:, col])
#             if np.abs(minim) > np.abs(maxim):
#                 self.a[:, col] *= -1
#
#
#         self.C = self.Xstd @ self.a
#
#
#         self.Rxc = self.a * np.sqrt(self.alpha)
#
#
#         self.scoruri = self.C / np.sqrt(self.alpha)
#
#         C2 = self.C * self.C
#         C2sum = np.sum(C2, axis=1)
#         self.CalObs = np.transpose(np.transpose(C2) / C2sum)
#
#         self.betha = C2 / (self.alpha * self.X.shape[0])
#
#         Rxc2 = self.Rxc * self.Rxc
#         self.Comun = np.cumsum(Rxc2, axis=1)
#
#     def getCorr(self):
#         return self.R
#
#     def getXstd(self):
#         return self.Xstd
#
#     def getValProp(self):
#         return self.alpha
#
#     def getCompPrin(self):
#         return self.C
#
#     def getRxc(self):
#         return self.Rxc
#
#     def getScoruri(self):
#         return self.scoruri
#
#     def getCalObs(self):
#         return self.CalObs
#
#     def getBetha(self):
#         return self.betha
#
#     def getComun(self):
#         return self.Comun