import csv

with open('complete_field_mappings.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

modified_data = []
for i, row in enumerate(data):
    if i == 0:
        modified_data.append(row)
        continue
    modified_data.append([i-1] + row)

with open('complete_fixed.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(modified_data)