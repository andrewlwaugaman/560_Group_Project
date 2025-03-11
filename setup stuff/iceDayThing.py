import csv
import numpy as np
import pandas

mendotaDays = np.ndarray((164))
mononaDays = np.ndarray((164))
with open('./mendota.csv', newline='') as csvfile: # Filepath might need to be changed
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        mendotaDays[i] = int(row["Days of Ice Cover"])

with open('./monona.csv', newline='') as csvfile: # Filepath might need to be changed
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        mononaDays[i] = int(row["Days of Ice Cover"])

allDays = np.array([range(2018, 1854, -1), mendotaDays, mononaDays]).T
print(allDays)
allDaysDataframe = pandas.DataFrame({"Year": allDays.T[0], "Mendota": allDays.T[1], "Monona": allDays.T[2]})
with open('./mendotaMonona.csv', newline='', mode="w") as csvfile:
    allDaysDataframe.to_csv(path_or_buf=csvfile, index=False)