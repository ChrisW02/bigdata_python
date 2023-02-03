import csv

with open("cities.csv") as csvfile:
    ct = list(csv.DictReader(csvfile))

print("len:", len(ct))
print("keys:", ct[0].keys())