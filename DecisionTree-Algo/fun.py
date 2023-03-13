def E(perc):
    Erp = perc*0.2 + (1-perc)*0.1
    var_p = (perc**2 * 0.3**2 + (1-perc)**2 * 0.1**2 + 2*perc*(1-perc)*0.2*0.3*0.1)**0.5
    return Erp*100, var_p*100

for i in range(0, 110, 10):
    print(i,": " ,E(i/100))