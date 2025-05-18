# Import necessary libraries
import re
import pandas as pd
import numpy as np
import json
from IPython.display import display, HTML

# Address parsing functions
def standardize_postal_code(postal_code):
    """Standardize postal code format"""
    if not postal_code:
        return None
    postal_code = re.sub(r'[^a-zA-Z0-9]', '', postal_code)
    postal_code = postal_code.upper()
    if len(postal_code) == 6 and re.match(r'^[A-Z]\d[A-Z]\d[A-Z]\d$', postal_code):
        return f"{postal_code[:3]} {postal_code[3:]}"
    return postal_code

def extract_street_components(address_str):
    """Extract street number and name from address string"""
    # Remove commas and strip
    address_str = address_str.replace(',', ' ').strip()
    components = address_str.split()
    
    # Handle case with two numbers at the beginning (unit and street number)
    if len(components) >= 2 and components[0].isdigit() and components[1].isdigit():
        unit_number = components[0]
        street_number = components[1]
        street_name = ' '.join(components[2:])
    else:
        # Regular address format
        match = re.match(r'^(\d+(?:-\d+)?(?:\s+\w+)?)\s+(.+)$', address_str)
        if match:
            unit_number = None
            street_number = match.group(1)
            street_name = match.group(2)
        else:
            unit_number = None
            street_number = None
            street_name = address_str
    
    return unit_number, street_number, street_name

def standardize_street_type(street_name):
    """Standardize common street type abbreviations"""
    street_type_map = {
        'avenue': 'Ave',
        'boulevard': 'Blvd',
        'circle': 'Cir',
        'court': 'Ct',
        'crescent': 'Cres',
        'drive': 'Dr',
        'expressway': 'Expy',
        'freeway': 'Fwy',
        'highway': 'Hwy',
        'lane': 'Ln',
        'parkway': 'Pkwy',
        'place': 'Pl',
        'road': 'Rd',
        'square': 'Sq',
        'street': 'St',
        'terrace': 'Ter',
        'trail': 'Trl',
        'way': 'Way'
    }
    
    if not street_name:
        return street_name
        
    # Split into words
    words = street_name.split()
    if not words:
        return street_name
        
    # Check if the last word is a street type that should be standardized
    last_word = words[-1].lower()
    if last_word in street_type_map:
        words[-1] = street_type_map[last_word]
        
    return ' '.join(words)

def extract_complex_address_components(address_str):
    """
    Extract components from complex address formats including various unit formats.
    Returns unit_number, street_number, street_name
    """
    if not address_str:
        return None, None, None
        
    # Clean the address string
    address_str = address_str.strip()
    
    # Handle "Unit X - Y" format
    unit_pattern = re.compile(r'^(?:Unit|UNIT|Apt\.?|APT\.?|#)\s*([A-Za-z0-9-]+)\s*[-:]\s*(\d+.*)$')
    unit_match = unit_pattern.match(address_str)
    
    if unit_match:
        unit_number = unit_match.group(1)
        remaining = unit_match.group(2)
        
        # Extract street number and name from remaining part
        components = remaining.split(maxsplit=1)
        if len(components) >= 2:
            street_number = components[0]
            street_name = components[1]
        else:
            street_number = remaining
            street_name = ""
            
        return unit_number, street_number, street_name
    
    # Handle "X-Y Street" format (like 210-40 Regency Park Dr)
    complex_pattern = re.compile(r'^(\d+)[-:](\d+)\s+(.+)$')
    complex_match = complex_pattern.match(address_str)
    
    if complex_match:
        unit_number = complex_match.group(1)
        street_number = complex_match.group(2)
        street_name = complex_match.group(3)
        return unit_number, street_number, street_name
    
    # Handle standard "X Street" format
    standard_pattern = re.compile(r'^(\d+(?:\s+[A-Za-z0-9]+)?)\s+(.+)$')
    standard_match = standard_pattern.match(address_str)
    
    if standard_match:
        street_number = standard_match.group(1)
        street_name = standard_match.group(2)
        return None, street_number, street_name
    
    # If no patterns match, return the whole string as street_name
    return None, None, address_str

def standardize_street_type(street_name):
    """Standardize common street type abbreviations"""
    street_type_map = {
        'avenue': 'Ave',
        'boulevard': 'Blvd',
        'circle': 'Cir',
        'court': 'Ct',
        'crescent': 'Cres',
        'drive': 'Dr',
        'expressway': 'Expy',
        'freeway': 'Fwy',
        'highway': 'Hwy',
        'lane': 'Ln',
        'parkway': 'Pkwy',
        'place': 'Pl',
        'road': 'Rd',
        'square': 'Sq',
        'street': 'St',
        'terrace': 'Ter',
        'trail': 'Trl',
        'way': 'Way'
    }
    
    if not street_name:
        return street_name
        
    # Split into words
    words = street_name.split()
    if not words:
        return street_name
        
    # Check if the last word is a street type that should be standardized
    last_word = words[-1].lower()
    if last_word in street_type_map:
        words[-1] = street_type_map[last_word]
        
    return ' '.join(words)

def parse_address(address_str):
    """
    Parse address string into components.
    Returns unit_number, street_number, street_name
    """
    if not address_str:
        return None, None, None
    
    # Clean the address string
    address_str = address_str.strip()
    
    # Case 1: Unit X - Y Street Name format
    unit_prefix_pattern = re.compile(r'^(?:Unit|UNIT|Apt\.?|APT\.?|#)\s*(\w+)\s*[-:]\s*(\d+)\s+(.+)$', re.IGNORECASE)
    match = unit_prefix_pattern.match(address_str)
    if match:
        unit_number = match.group(1)
        street_number = match.group(2)
        street_name = match.group(3)
        return unit_number, street_number, street_name
    
    # Case 2: X-Y Street Name format (unit-street number)
    unit_street_pattern = re.compile(r'^(\d+)[-:]\s*(\d+)\s+(.+)$')
    match = unit_street_pattern.match(address_str)
    if match:
        unit_number = match.group(1)
        street_number = match.group(2)
        street_name = match.group(3)
        return unit_number, street_number, street_name
    
    # Case 3: Standard street address with number
    # Look for the first number followed by the rest of the address
    standard_pattern = re.compile(r'^(\d+(?:\s*-\s*\d+)?)\s+(.+)$')
    match = standard_pattern.match(address_str)
    if match:
        street_number = match.group(1).strip()
        street_name = match.group(2).strip()
        return None, street_number, street_name
    
    # If no patterns match, return the whole string as street_name
    return None, None, address_str

def normalize_address_components(unit_number, street_number, street_name):
    """
    Normalize address components by handling special cases and standardizing formats.
    """
    # Standardize unit number format
    if unit_number:
        unit_number = unit_number.strip()
    
    # Standardize street number format
    if street_number:
        street_number = street_number.strip()
    
    # Standardize street name and handle multi-word street names
    if street_name:
        street_name = street_name.strip()
        # Standardize street type abbreviations
        street_name = standardize_street_type(street_name)
    
    return unit_number, street_number, street_name

# Process different address types
def process_subject_address(subject_data):
    """Process subject property address"""
    # Get the raw address strings
    full_address = subject_data.get('address', '') or ''
    city_province_zip = subject_data.get('subject_city_province_zip', '') or ''
    
    # Clean the strings
    full_address = full_address.replace(',', ' ').strip()
    city_province_zip = city_province_zip.replace(',', ' ').strip().replace('"', '').strip()
    
    # Extract city, province, postal code from subject_city_province_zip
    parts = city_province_zip.split()
    if len(parts) >= 3:
        # Last element is likely postal code
        postal_code = parts[-1]
        # Second-to-last is likely province
        province = parts[-2]
        # Everything else is city
        city = ' '.join(parts[:-2])
    else:
        city = city_province_zip
        province = None
        postal_code = None
    
    # Remove city_province_zip from full_address if it appears at the end
    street_address = full_address
    if city and city in street_address:
        street_address = street_address[:street_address.find(city)].strip()
    
    # Extract street components
    unit_number, street_number, street_name = extract_street_components(street_address)
    
    # Create standardized address dictionary
    standardized = {
        'std_unit_number': unit_number,
        'std_street_number': street_number,
        'std_street_name': street_name,
        'std_city': city,
        'std_province': province,
        'std_postal_code': standardize_postal_code(postal_code)
    }
    
    # Create standardized full address
    std_parts = []
    if unit_number:
        std_parts.append(f"Unit {unit_number}")
    if street_number:
        std_parts.append(street_number)
    if street_name:
        std_parts.append(street_name)
    if city:
        std_parts.append(city)
    if province:
        std_parts.append(province)
    if postal_code:
        std_parts.append(postal_code)
    
    standardized['std_full_address'] = ', '.join(std_parts)
    return standardized

def process_comp_address(comp_data):
    """Process comp property address"""
    # Get the raw address strings
    address = comp_data.get('address', '') or ''
    city_province = comp_data.get('city_province', '') or ''
    
    # Clean the strings
    address = address.replace(',', ' ').strip()
    city_province = city_province.replace(',', ' ').strip()
    
    # Extract city, province, postal code from city_province
    parts = city_province.split()
    if len(parts) >= 3:
        # Last element is likely postal code
        postal_code = parts[-1]
        # Second-to-last is likely province
        province = parts[-2]
        # Everything else is city
        city = ' '.join(parts[:-2])
    else:
        city = city_province
        province = None
        postal_code = None
    
    # Extract street components from address
    unit_number, street_number, street_name = extract_street_components(address)
    
    # Create standardized address dictionary
    standardized = {
        'std_unit_number': unit_number,
        'std_street_number': street_number,
        'std_street_name': street_name,
        'std_city': city,
        'std_province': province,
        'std_postal_code': standardize_postal_code(postal_code)
    }
    
    # Create standardized full address
    std_parts = []
    if unit_number:
        std_parts.append(f"Unit {unit_number}")
    if street_number:
        std_parts.append(street_number)
    if street_name:
        std_parts.append(street_name)
    if city:
        std_parts.append(city)
    if province:
        std_parts.append(province)
    if postal_code:
        std_parts.append(postal_code)
    
    standardized['std_full_address'] = ', '.join(std_parts)
    return standardized

def process_property_address(property_data):
    """Process property address with province standardization"""
    # Get address components - handle None values
    address = property_data.get('address', '') or ''
    city = property_data.get('city', '') or ''
    province = property_data.get('province', '') or ''
    postal_code = property_data.get('postal_code', '') or ''
    
    # Clean the strings
    address = address.strip()
    city = city.strip()
    province = province.strip()
    postal_code = postal_code.strip()
    
    # Standardize province (convert full name to abbreviation)
    province_map = {
        'Alberta': 'AB',
        'British Columbia': 'BC',
        'Manitoba': 'MB',
        'New Brunswick': 'NB',
        'Newfoundland': 'NL',
        'Newfoundland And Labrador': 'NL',
        'Northwest Territories': 'NT',
        'Nova Scotia': 'NS',
        'Nunavut': 'NU',
        'Ontario': 'ON',
        'Prince Edward Island': 'PE',
        'Quebec': 'QC',
        'Saskatchewan': 'SK',
        'Yukon': 'YT'
    }
    
    # If province is a full name, convert to abbreviation
    if province in province_map:
        province = province_map[province]
    
    # Parse and normalize address components
    unit_number, street_number, street_name = parse_address(address)
    unit_number, street_number, street_name = normalize_address_components(unit_number, street_number, street_name)
    
    # Create standardized address dictionary
    standardized = {
        'std_unit_number': unit_number,
        'std_street_number': street_number,
        'std_street_name': street_name,
        'std_city': city,
        'std_province': province,
        'std_postal_code': standardize_postal_code(postal_code)
    }
    
    # Create standardized full address
    std_parts = []
    if unit_number:
        std_parts.append(f"Unit {unit_number}")
    if street_number:
        std_parts.append(street_number)
    if street_name:
        std_parts.append(street_name)
    if city:
        std_parts.append(city)
    if province:
        std_parts.append(province)
    if postal_code:
        std_parts.append(standardize_postal_code(postal_code))
    
    standardized['std_full_address'] = ', '.join(std_parts)
    
    return standardized

# Data cleaning functions
def clean_string(value):
    """Clean string values by removing unwanted characters and handling null values"""
    if pd.isnull(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value in ["", "CONDO - N/A", "N/A Condominium", "N/A - CONDO", "N/A-CONDO LAND", 
                    "N/A", "n/a", "Condo n/a", "n/a condo", "n/a-condo land"]:
            return np.nan
        # Remove unicode characters
        value = value.replace('\u00b1', '')
        value = value.replace('\xb1', '')
        return value
    return value

def clean_dataframe(df):
    """Apply cleaning to all string columns in a dataframe"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_string)
    return df

# Additional data processing functions for specific fields
def process_sale_price(value):
    """Remove commas from sale price and convert to float"""
    if pd.isnull(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove commas
        value = value.replace(',', '')
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan

def process_bedroom_count(value):
    """Process bedroom counts like '2+1' to total number"""
    if pd.isnull(value):
        return np.nan
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Handle format like "2+1"
        if '+' in value:
            parts = value.split('+')
            total = 0
            for part in parts:
                try:
                    total += int(part.strip())
                except ValueError:
                    pass
            return total
        # Try to convert directly to integer
        try:
            return int(value.strip())
        except ValueError:
            return np.nan
    return np.nan

def process_bathroom_count(value):
    """
    Process bathroom counts like '2:1' or '2F1P' to extract full and half bathrooms
    Returns a tuple of (full_baths, half_baths)
    """
    if pd.isnull(value):
        return (np.nan, np.nan)
    
    if isinstance(value, str):
        # Handle format "2:1" (2 full, 1 half)
        if ':' in value:
            parts = value.split(':')
            try:
                full_baths = int(parts[0].strip())
                half_baths = int(parts[1].strip()) if len(parts) > 1 else 0
                return (full_baths, half_baths)
            except ValueError:
                pass
        
        # Handle format "2F1P" or "2F 1P" or "2F 1H" (2 full, 1 half)
        match = re.search(r'(\d+)\s*[Ff].*?(\d+)\s*[PpHh]', value)
        if match:
            try:
                full_baths = int(match.group(1))
                half_baths = int(match.group(2))
                return (full_baths, half_baths)
            except ValueError:
                pass
        
        # Try to extract just a single number (assume full baths)
        match = re.search(r'(\d+)', value)
        if match:
            try:
                return (int(match.group(1)), 0)
            except ValueError:
                pass
    
    return (np.nan, np.nan)

def process_gla(value):
    """
    Process Gross Living Area (GLA) values with different formats and units
    Converts square meters to square feet if needed
    """
    if pd.isnull(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove any +/- symbols and unicode variants
        value = value.replace('+/-', '').replace('\u00b1', '').replace('±', '')
        
        # Extract the numeric part
        match = re.search(r'([\d.]+)', value)
        if not match:
            return np.nan
        
        try:
            numeric_value = float(match.group(1))
            
            # Check if units are square meters and convert if needed
            if 'sqm' in value.lower() or 'sq m' in value.lower():
                # Convert square meters to square feet (1 sq m = 10.764 sq ft)
                numeric_value *= 10.764
            
            return numeric_value
        except ValueError:
            return np.nan
    
    return np.nan

def remove_units_and_symbols(value):
    """Remove units and symbols from numeric string values"""
    if pd.isnull(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove +/- symbols
        value = value.replace('+/-', '').replace('\u00b1', '').replace('±', '')
        
        # Remove common units
        value = re.sub(r'\s*(SqFt|sqft|sf|Sq\.Ft\.|square feet|sq\. ft\.)\s*$', '', value, flags=re.IGNORECASE)
        value = re.sub(r'\s*(SqM|sq m|square meters|sq\. m\.)\s*$', '', value, flags=re.IGNORECASE)
        value = re.sub(r'\s*(KM|km|kilometers)\s*$', '', value, flags=re.IGNORECASE)
        value = re.sub(r'\s*(Acres|acres|ac)\s*$', '', value, flags=re.IGNORECASE)
        
        # Try to convert to float
        try:
            return float(value.strip())
        except ValueError:
            return np.nan
    
    return np.nan

# Apply specific field processing to DataFrames
def apply_specific_processing(subjects_df, comps_df, properties_df):
    """Apply specific processing to fields that need special handling"""
    
    # Process sale_price in comps
    if 'sale_price' in comps_df.columns:
        comps_df['sale_price'] = comps_df['sale_price'].apply(process_sale_price)
    
    # Process bedroom counts
    if 'num_beds' in subjects_df.columns:
        subjects_df['bedrooms'] = subjects_df['num_beds'].apply(process_bedroom_count)
        # Keep original column for reference if needed
        subjects_df.drop('num_beds', axis=1, inplace=True)
    
    if 'bed_count' in comps_df.columns:
        comps_df['bedrooms'] = comps_df['bed_count'].apply(process_bedroom_count)
        # Keep original column for reference if needed
        comps_df.drop('bed_count', axis=1, inplace=True)
    
    # Process bathroom counts
    if 'num_baths' in subjects_df.columns:
        bath_results = subjects_df['num_baths'].apply(process_bathroom_count)
        subjects_df['full_baths'] = [result[0] for result in bath_results]
        subjects_df['half_baths'] = [result[1] for result in bath_results]
        # Keep original column for reference if needed
        subjects_df.drop('num_baths', axis=1, inplace=True)
    
    if 'bath_count' in comps_df.columns:
        bath_results = comps_df['bath_count'].apply(process_bathroom_count)
        comps_df['full_baths'] = [result[0] for result in bath_results]
        comps_df['half_baths'] = [result[1] for result in bath_results]
        # Keep original column for reference if needed
        comps_df.drop('bath_count', axis=1, inplace=True)
    
    # Process GLA (Gross Living Area)
    if 'gla' in subjects_df.columns:
        subjects_df['gla'] = subjects_df['gla'].apply(process_gla)
    
    if 'gla' in comps_df.columns:
        comps_df['gla'] = comps_df['gla'].apply(process_gla)
    
    if 'gla' in properties_df.columns:
        properties_df['gla'] = properties_df['gla'].apply(process_gla)

    # Remove units from numeric fields
    numeric_fields_with_units = [
        'distance_to_subject', 
        'gla',
        'lot_size'
        'lot_size_sf',
        'main_lvl_area',
        'second_lvl_area',
        'third_lvl_area',
        'basement_area',
        'main_level_finished_area',
        'upper_lvl_fin_area'
    ]
    
    for df in [subjects_df, comps_df, properties_df]:
        for field in numeric_fields_with_units:
            if field in df.columns:
                df[field] = df[field].apply(remove_units_and_symbols)
    
    return subjects_df, comps_df, properties_df

# Data type conversion function
def convert_column_types(df, mapping_df, section_name):
    """
    Convert dataframe columns to their assigned data types based on mapping.
    Also handles field renaming and dropping excluded features.
    
    Args:
        df: DataFrame to process
        mapping_df: DataFrame containing field mappings
        section_name: Section name ('subject', 'comps', or 'properties')
    
    Returns:
        Processed DataFrame
    """
    
    # Rename fields based on canonical_field in mapping
    rename_dict = {}
    for _, row in mapping_df.iterrows():
        orig_field = row['original_field']
        canonical_field = row['canonical_field']
        
        # Skip if no canonical field specified or if it's the same as original
        if pd.isna(canonical_field) or canonical_field == '' or canonical_field == 'split' or canonical_field == orig_field:
            continue
            
        # Add to rename dictionary if the original field exists in the dataframe
        if orig_field in df.columns:
            rename_dict[orig_field] = canonical_field
    
    # Apply renaming
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    # Drop excluded features
    excluded_features = [
        # 'address', 'subject_city_province_zip', 'city_province', 'city', 'province', 'postal_code',
        "site_dimensions", "units_sq_ft", 
        "lot_size", "lot_size_sf",
        "water_heater", "exterior_finish", 
        "style", "levels",
        "effective_age", "subject_age", "age",
        "effective_date", "close_date", "sale_date"
    ]
    cols_to_drop = [col for col in excluded_features if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    
    # Convert data types
    for _, row in mapping_df.iterrows():
        orig_field = row['original_field']
        dtype = row['data_type']
        
        # Determine which field name to use (original or canonical)
        if orig_field in rename_dict:
            col = rename_dict[orig_field] # canonical
        else:
            col = orig_field
        
        # Skip if column doesn't exist in the dataframe
        if col not in df.columns:
            continue
        
        try:
            if dtype == 'integer':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif dtype == 'float':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif dtype == 'categorical':
                df[col] = df[col].astype('category')
            elif dtype == 'ordinal':
                # Handle ordinal fields based on section and field name
                if section_name == 'subject' and col == 'condition':
                    # For subject condition (absolute rating)
                    condition_order = ['Poor', 'Fair', 'Average', 'Good', 'Excellent']
                    df[col] = pd.Categorical(df[col], categories=condition_order, ordered=True)
                elif section_name == 'comps' and col in ['condition_relative', 'location_similarity']:
                    # For comps comparison fields (relative rating)
                    comparison_order = ['Inferior', 'Similar', 'Superior']
                    df[col] = pd.Categorical(df[col], categories=comparison_order, ordered=True)
            elif dtype == 'string':
                # Keep strings as is, but handle NaN values properly
                df[col] = df[col].astype(object)
            elif dtype == 'text':
                # For text fields like public_remarks
                df[col] = df[col].astype(object)
        except Exception as e:
            print(f"Error converting {col} to {dtype}: {e}")
    
    return df

def load_and_process_data(json_file_path, mapping_file_path):
    """
    Load JSON data and field mapping, then process the data
    """
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Load the field mapping
    mapping_df = pd.read_csv(mapping_file_path)
    
    # Initialize lists to store processed data
    subjects = []
    comps = []
    properties = []
    
    # Process each appraisal
    for appraisal in data.get('appraisals', []):
        # Process subject property
        if 'subject' in appraisal:
            subject_data = appraisal['subject'].copy()
            # Add standardized address
            subject_data.update(process_subject_address(subject_data))
            subjects.append(subject_data)
        
        # Process comp properties
        if 'comps' in appraisal:
            for comp in appraisal['comps']:
                comp_data = comp.copy()
                # Add standardized address
                comp_data.update(process_comp_address(comp_data))
                # Add reference to subject property
                if 'subject' in appraisal and 'address' in appraisal['subject']:
                    comp_data['subject_address'] = appraisal['subject']['address']
                comps.append(comp_data)
        
        # Process available properties
        if 'properties' in appraisal:
            for prop in appraisal['properties']:
                prop_data = prop.copy()
                # Add reference to subject property
                if 'subject' in appraisal and 'address' in appraisal['subject']:
                    prop_data['subject_address'] = appraisal['subject']['address']
                properties.append(prop_data)
    
    # Convert to DataFrames
    subjects_df = pd.DataFrame(subjects)
    comps_df = pd.DataFrame(comps)
    properties_df = pd.DataFrame(properties)
    
    # Clean the DataFrames (basic cleaning)
    subjects_df = clean_dataframe(subjects_df)
    comps_df = clean_dataframe(comps_df)
    properties_df = clean_dataframe(properties_df)
    
    # Apply specific field processing
    subjects_df, comps_df, properties_df = apply_specific_processing(subjects_df, comps_df, properties_df)
    
    # Convert column types
    subjects_df = convert_column_types(subjects_df, mapping_df[mapping_df['section'] == 'subject'])
    comps_df = convert_column_types(comps_df, mapping_df[mapping_df['section'] == 'comps'])
    properties_df = convert_column_types(properties_df, mapping_df[mapping_df['section'] == 'properties'])
    
    return subjects_df, comps_df, properties_df


