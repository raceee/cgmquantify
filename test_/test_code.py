import cgmquantify as cgm
from cgmquantify import CGMQuantify

# data = cgm.importdexcom('test_file.csv')
data = CGMQuantify('./test_/test_file.csv')
# print('interdaysd is: ' + str(data.interdaysd()))
# print('interdaycv is: ' + str(data.interdaycv()))
# print('intradaysd is: ' + str(data.intradaysd()))
# print('intradaycv is: ' + str(data.intradaycv()))
# print('TOR is: ' + str(data.TOR()))
# print('TIR is: ' + str(data.TIR()))
# print('POR is: ' + str(data.POR()))
# print('PIR is: ' + str(data.PIR()))
# print('MGE is: ' + str(data.MGE()))
# print('MGN is: ' + str(data.MGN()))
# print('MAGE is: ' + str(data.MAGE()))
# print('MODD is: ' + str(data.MODD()))
# print('J_index is: ' + str(data.J_index()))
# print('LBGI is: ' + str(data.LBGI()))
# print('HBGI is: ' + str(data.HBGI()))
# print('ADRR is: ' + str(data.ADRR()))
# print('CONGA24 is: ' + str(data.CONGA24()))
# print('GMI is: ' + str(data.GMI()))
# print('eA1c is: ' + str(data.eA1c()))
print('summary is: ' + str(data.summary()))

data.plotglucosebounds()

data.plotglucosesd()

data.plotglucosesmooth()
