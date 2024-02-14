import numpy as np
import pandas as pd
import utile as utl
import aef.AEF as aef
import acp.ACP as acp
import factor_analyzer as fa
import grafice as graf
from sklearn.preprocessing import StandardScaler

# Citirea datelor din fișierul CSV
tabel = pd.read_csv('lifeexpectancy.csv', index_col=0)
print(tabel)

# Extragerea numelor variabilelor și observațiilor
numeVariabila = list(tabel.columns)[1:]
print(numeVariabila)

numeObs = list(tabel.index)
print(numeObs)

# Extrage matricea de date pentru analiza ACP
X = tabel[numeVariabila].values
print(X)


# Calculul Xstd_df si salvare in fisier CSV
XstdDataFrame = pd.DataFrame(data=X, index=numeObs, columns=numeVariabila)
XstdDataFrame.to_csv('./dataOUT/X.csv')

# Testul Bartlett pentru sfericitate
sfericitateBartlett = fa.calculate_bartlett_sphericity(XstdDataFrame)
print(sfericitateBartlett, type(sfericitateBartlett))
if sfericitateBartlett[0] > sfericitateBartlett[1]:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista nici un factor comun!')
    exit(-1)

# Calculul indiciilor Kaiser-Meyer-Olkin (KMO) de factorabilitate a variabilelor observate
kmo = fa.calculate_kmo(XstdDataFrame)
print(kmo, type(kmo))
# Testul valorii globale a indicelui KMO
if kmo[1] > 0.5:
    print('Exista cel putin un factor comun!')
else:
    print('Variabilele initiale, observate sunt independente!')
    exit(-2)

# Corelograma indicilor KMO
vector = kmo[0]
print(vector, type(vector))
matrice = vector[:, np.newaxis]
print(matrice, type(matrice))
matrice_df = pd.DataFrame(data=matrice, index=numeVariabila,
                          columns=['Indici KMO'])
matrice_df.to_csv('./dataOUT/KMO.csv')
graf.corelograma(matrice=matrice_df, titlu='Corelograma indicilor KMO')

# Extragere factori utilizând Analiza Factorială (FA)
numarFactoriSemnificativi = 1
chi2TabMin = 1
for k in range(1, len(numeVariabila)):
    modelFA = fa.FactorAnalyzer(n_factors=k)
    modelFA.fit(X=XstdDataFrame)
    factoriComuni = modelFA.loadings_
    print(factoriComuni, type(factoriComuni))
    factoriSpecifici = modelFA.get_uniquenesses()
    print(factoriSpecifici, type(factoriSpecifici))

    # Aplicare Analiza de Echivalență a Factorilor (AEF)
    modelAEF = aef.AEF(matrice=X)
    chi2Calc, chi2Tab = modelAEF.calculTestBartlett(factoriComuni, factoriSpecifici)
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2TabMin > chi2Tab:
        chi2TabMin = chi2Tab
        numarFactoriSemnificativi = k

print('Nr. factori semnificativi: ', numarFactoriSemnificativi)

# Cream un model FA cu numarul semnificativ de factori determinat
fitModelFA = fa.FactorAnalyzer(n_factors=numarFactoriSemnificativi)
fitModelFA.fit(XstdDataFrame)
factorLoadingsFA = fitModelFA.loadings_  # Factor loadings (corelatia dintre variabilele initiale si factorii comuni)

# Salvare factor loadings
factori = ['F' + str(j + 1) for j in range(numarFactoriSemnificativi)]
factorLoadingsFA_df = pd.DataFrame(data=factorLoadingsFA,
                                   index=numeVariabila, columns=factori)
factorLoadingsFA_df.to_csv('./dataOUT/factorLoadingsFA.csv')

# Creare corelograma a factorilor de corelatie din FA
graf.corelograma(matrice=factorLoadingsFA_df, titlu='Corelograma factorilor de corelatie din FA')

# Extragere valori proprii din FA
valPropFA = fitModelFA.get_eigenvalues()
print(valPropFA, type(valPropFA))

# Realizare grafic al variantei explicate de factori din FA
graf.componentePrincipale(valoriProprii=valPropFA[1], titlu='Varianta explicate de factorii comuni FA')


# Inițierea modelului de Analiză Componentelor Principale (ACP)
acpModel = acp.ACP(X)

# Obținerea matricei standardizate (Xstd) și salvarea într-un fișier CSV
Xstd = acpModel.getXstd()
XstdDataFrame = pd.DataFrame(data=Xstd, index=numeObs, columns=numeVariabila)
XstdDataFrame.to_csv('OUT/Xstd.csv')

# Obținerea valorilor proprii și realizarea unui grafic cu componentele principale
valProp = acpModel.getValProp()
graf.componentePrincipale(valoriProprii=valProp)

# Obținerea matricei factorilor de corelație (Rxc) și salvarea într-un fișier CSV
Rxc = acpModel.getRxc()
RxcDataFrame = pd.DataFrame(data=Rxc, index=numeVariabila, columns=('C' + str(k + 1) for k in range(len(numeVariabila))))
RxcDataFrame.to_csv('OUT/FactoriCorelatie.csv')
graf.corelograma(matrice=RxcDataFrame, dec=2, titlu='Corelograma factorilor de corelatie')

# Obținerea scorurilor și salvarea într-un fișier CSV
scoruri = acpModel.getScoruri()
scoruriDataFrame = pd.DataFrame(data=scoruri, index=numeObs, columns=('C' + str(k + 1) for k in range(len(numeVariabila))))
scoruriDataFrame.to_csv('OUT/Scoruri.csv')
graf.corelograma(matrice=scoruriDataFrame, dec=2, titlu='Corelograma scorurilor (standardizarea componentelor principale)')

# Obținerea matricei Calitatea Observațiilor (CalObs) și salvarea într-un fișier CSV
calObs = acpModel.getCalObs()
calObsDataFrame = pd.DataFrame(data=calObs, index=numeObs, columns=('C' + str(k + 1) for k in range(len(numeVariabila))))
calObsDataFrame.to_csv('OUT/CalitateObservatii.csv')
graf.corelograma(matrice=calObsDataFrame, dec=2, titlu='Calitatea reprezentarii observatiilor pe axele componentelor principale')

# Obținerea matricei Betha și salvarea într-un fișier CSV
betha = acpModel.getBetha()
bethaDataFrame = pd.DataFrame(data=betha, index=numeObs, columns=('C' + str(k + 1) for k in range(len(numeVariabila))))
bethaDataFrame.to_csv('OUT/Betha.csv')
graf.corelograma(matrice=bethaDataFrame, dec=2, titlu='Contributia observatiilor la varianta axelor componentelor principale')

# Obținerea matricei Comunității și salvarea într-un fișier CSV
comunitati = acpModel.getComun()
comunitatiDataFrame = pd.DataFrame(data=comunitati, index=numeVariabila, columns=('C' + str(k + 1) for k in range(len(numeVariabila))))
comunitatiDataFrame.to_csv('OUT/Comunitati.csv')
graf.corelograma(matrice=comunitatiDataFrame, dec=2, titlu='Corelograma comunalitatii')

# Realizarea cercului corelațiilor între variabilele inițiale și C1 și C2
graf.cerculCorelatiilor(matrice=RxcDataFrame, titlu='Cercul corelatiilor intre variabilele initiale si C1 si C2')

# Calcularea valorilor minime și maxime ale scorurilor și realizarea cercului corelațiilor
scor_max = np.max(scoruri)
scor_min = np.min(scoruri)
print('Scor folosit ca raza pentru cercul corelatiilor: ', scor_max)

graf.cerculCorelatiilor(matrice=scoruriDataFrame, raza=scor_max, valMin=scor_min, valMax=scor_max,
                        titlu='Distributia observatiilor in spatiul componentelor C1 si C2')
graf.afisare()

