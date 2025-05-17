import json
import pandas as pd
from collections import defaultdict

# Load the provided JSON data from the attachment
json_file_path = "appraisals_dataset.json"

with open(json_file_path, 'r') as file:
    data = json.load(file)

# Initialize dictionaries to store field names by section and sample values
field_names = {
    'subject': defaultdict(set),
    'comps': defaultdict(set),
    'properties': defaultdict(set)
}

# Extract field names and collect up to 3 unique sample values for each field
for appraisal in data.get('appraisals', []):
    # Subject property
    if 'subject' in appraisal:
        for field, value in appraisal['subject'].items():
            if value and value != '' and value != 'n/a' and value != 'N/A':
                if len(field_names['subject'][field]) < 3:
                    field_names['subject'][field].add(str(value))
    # Comps
    if 'comps' in appraisal:
        for comp in appraisal['comps']:
            for field, value in comp.items():
                if value and value != '' and value != 'n/a' and value != 'N/A':
                    if len(field_names['comps'][field]) < 3:
                        field_names['comps'][field].add(str(value))
    # Properties
    if 'properties' in appraisal:
        for prop in appraisal['properties']:
            for field, value in prop.items():
                if value and value != '' and value != 'n/a' and value != 'N/A':
                    if len(field_names['properties'][field]) < 3:
                        field_names['properties'][field].add(str(value))

# Convert sets to sorted lists for consistent output
for section in field_names:
    for field in field_names[section]:
        field_names[section][field] = sorted(field_names[section][field])

# Create a DataFrame for the mapping template
all_fields = []
for section, fields in field_names.items():
    for field, samples in fields.items():
        all_fields.append({
            'section': section,
            'original_field': field,
            'sample_values': ', '.join(samples),
            'canonical_field': '',  # To be filled manually
            'data_type': ''  # To be filled manually
        })

mapping_df = pd.DataFrame(all_fields)

# Save the DataFrame to CSV
mapping_df.to_csv('field_mapping_template_with_samples.csv', index=True)

mapping_df.head()  # Display the first few rows as output
