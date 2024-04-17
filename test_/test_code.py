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
print('MGE is: ' + str(data.MGE()))
# print('MGN is: ' + str(cgm.MGN(data)))
# print('MAGE is: ' + str(cgm.MAGE(data)))
# print('MODD is: ' + str(cgm.MODD(data)))
# print('J_index is: ' + str(cgm.J_index(data)))
# print('LBGI is: ' + str(cgm.LBGI(data)))
# print('HBGI is: ' + str(cgm.HBGI(data)))
# print('ADRR is: ' + str(cgm.ADRR(data)))
# print('CONGA24 is: ' + str(cgm.CONGA24(data)))
# print('GMI is: ' + str(cgm.GMI(data)))
# print('eA1c is: ' + str(cgm.eA1c(data)))
# print('summary is: ' + str(cgm.summary(data)))

# cgm.plotglucosebounds(data)

# cgm.plotglucosesd(data)

# cgm.plotglucosesmooth(data)
