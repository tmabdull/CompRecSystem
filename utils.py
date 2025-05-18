# Import necessary libraries
import re
import pandas as pd
import numpy as np
import json

def save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number:int=None, idx:bool=False):
    '''Save processed data to CSV files for further analysis'''
    
    v_str, v_str_spaces = "", ""
    if version_number:
        v_str = f"_v{version_number}"
        v_str_spaces = f" v{version_number}"
    
    processed_path_prefix = "./data/processed/processed_"

    subjects_df.to_csv(f"{processed_path_prefix}subjects{v_str}.csv", index=idx)
    comps_df.to_csv(f"{processed_path_prefix}comps{v_str}.csv", index=idx)
    properties_df.to_csv(f"{processed_path_prefix}properties{v_str}.csv", index=idx)

    print(f"Processed data{v_str_spaces} saved to CSV files.")

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

def normalize_address_components(unit_number, street_number, street_name, city=None, province=None, postal_code=None):
    """
    Normalize address components by handling special cases and standardizing formats.
    Also prepares components for deduplication by standardizing case and format.
    """
    # Standardize unit number format
    if unit_number:
        unit_number = unit_number.strip()
    
    # Standardize street number format
    if street_number:
        street_number = street_number.strip()
    
    # Standardize street name with proper capitalization
    if street_name:
        street_name = street_name.strip()
        street_name = standardize_street_type(street_name)

        # Apply title case (capitalize first letter of each word)
        street_name = street_name.title()
    
    # Normalize city with proper capitalization
    if city:
        city = city.strip()

        # Apply title case (capitalize first letter of each word)
        street_name = street_name.title()
    
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

    if province:
        if province in province_map:
            province = province_map[province]
        province = province.strip().upper()
    
    # Standardize postal code (remove spaces)
    if postal_code:
        postal_code = standardize_postal_code(postal_code)
        # Remove spaces for matching purposes
        postal_code = postal_code.replace(' ', '')
    
    return unit_number, street_number, street_name, city, province, postal_code

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

def create_standardized_address_dict(address, city, province, postal_code):
    """
    Create a standardized address dictionary and full address string from components.
    
    Args:
        address: Contains unparsed unit/apt number, street number, and street name if they exist
        city: city name (not standardized) if it exists
        province: province name or code (not standardized) if it exists
        postal_code: postal code (not standardized) if it exists
        
    Returns:
        Dictionary with standardized address components and full address
    """
    
    # Parse and normalize address components
    unit_number, street_number, street_name = parse_address(address)

    # Normalize all components for standardization and deduplication
    unit_number, street_number, street_name, city, province, postal_code = normalize_address_components(
        unit_number, street_number, street_name, city, province, postal_code
    )

    # Create standardized address dictionary
    standardized = {
        'std_unit_number': unit_number,
        'std_street_number': street_number,
        'std_street_name': street_name,
        'std_city': city,
        'std_province': province,
        'std_postal_code': postal_code
    }
    
    # Create standardized full address with conventional formatting
    std_parts = []

    # Combining street num and name so they're not comma separated
    street_address_parts = []

    if street_number:
        street_address_parts.append(street_number)
    if street_name:
        street_address_parts.append(street_name)
    
    # Join the street address parts with spaces instead of commas
    if street_address_parts:
        std_parts.append(" ".join(street_address_parts))
    
    # Add city, province, postal code with conventional formatting
    if unit_number:
        std_parts.append(f"Unit {unit_number}")
    if city:
        std_parts.append(city)
    if province:
        std_parts.append(province)
    if postal_code:
        std_parts.append(postal_code)
    
    standardized['std_full_address'] = ", ".join(std_parts)
    
    return standardized

# Process different address types
def process_subject_address(subject_data):
    """Process subject property address"""
    # Get the raw address strings
    full_address = subject_data.get('address', '') or ''
    city_province_zip = subject_data.get('subject_city_province_zip', '') or ''
    
    # Clean the strings
    full_address = full_address.strip()
    city_province_zip = city_province_zip.replace('"', '').strip()
    
    # Extract postal code from city_province_zip
    postal_code = None
    postal_match = re.search(r'([A-Za-z]\d[A-Za-z])\s*(\d[A-Za-z]\d)', city_province_zip)
    if postal_match:
        postal_code = f"{postal_match.group(1)} {postal_match.group(2)}"
        # Remove postal code from the string
        city_province_zip = city_province_zip[:postal_match.start()].strip().rstrip(',')
    
    # Now split the remaining city_province_zip to get city and province
    city_province_parts = city_province_zip.split()
    
    # Default values
    city = city_province_zip
    province = None
    
    # If we have at least two parts, assume the last part is the province
    if len(city_province_parts) >= 2:
        # Last element is likely province
        province = city_province_parts[-1]

        # Everything else is city
        city = ' '.join(city_province_parts[:-1])
    
    # Extract street address by removing city from full_address
    street_address = full_address
    
    # If we have a city, try to find it in the full address and remove everything from that point
    if city and len(city) > 2:  # Avoid very short city names that might cause false matches
        city_pos = -1
        # Try to find the city in the address
        if city.lower() in full_address.lower():
            city_pos = full_address.lower().find(city.lower())
        
        # If city is found in the address, remove it and everything after
        if city_pos > 0:
            street_address = full_address[:city_pos].strip()
    
    return create_standardized_address_dict(
        street_address, city, province, postal_code
    )

def process_comp_address(comp_data):
    """Process comp property address"""
    # Get the raw address strings
    address = comp_data.get('address', '') or ''
    city_province_postal = comp_data.get('city_province', '') or ''
    
    # Clean the strings
    address.strip()
    city_province_postal.strip()

    # Parse city_province_postal which may contain all three components
    # Look for postal code pattern at the end
    postal_code = None
    province = None
    city = city_province_postal
    
    # Try to extract postal code (Canadian format: A1A 1A1 or A1A1A1)
    postal_match = re.search(r'([A-Za-z]\d[A-Za-z])\s*(\d[A-Za-z]\d)$', city_province_postal)
    if postal_match:
        postal_code = f"{postal_match.group(1)} {postal_match.group(2)}"
        # Remove postal code from the string
        city = city_province_postal[:postal_match.start()].strip()
    
    # Now try to extract province (typically 2 letters before postal code)
    province_match = re.search(r'\b([A-Z]{2})\b', city)
    if province_match:
        province = province_match.group(1)
        # Split by the province to get the city
        parts = city.split(province)
        if len(parts) >= 2:
            city = parts[0].strip().rstrip(',')
            # If there's content after the province and it's not the postal code
            # (which we've already extracted), it might be part of the city
            if parts[1].strip() and not re.match(r'^\s*[A-Z]\d[A-Z]\s*\d[A-Z]\d', parts[1]):
                city += " " + parts[1].strip()
    
    # Handle special case where city contains "Alberta" instead of "AB"
    if "Alberta" in city and not province:
        city = city.replace("Alberta", "").strip().rstrip(',')
        province = "AB"
    
    return create_standardized_address_dict(
        address, city, province, postal_code
    )

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

    return create_standardized_address_dict(
        address, city, province, postal_code
    )

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

    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=3, idx=True)

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

    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=4, idx=True)
    
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
        'address', 'subject_city_province_zip', 'city_province', 'city', 'province', 'postal_code',
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

def load_and_process_data(json_file_path="./data/raw/appraisals_dataset.json", mapping_file_path="./data/mappings/complete_field_mappings.csv"):
    """
    Load JSON raw data and field mapping, then process the data
    """
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        raw_data = json.load(file)

    # Initialize lists to store processed data
    subjects = []
    comps = []
    properties = []
    
    # Process each appraisal
    for appraisal in raw_data.get('appraisals', []):
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
                # Add standardized address
                prop_data.update(process_property_address(prop_data))
                # Add reference to subject property
                if 'subject' in appraisal and 'address' in appraisal['subject']:
                    prop_data['subject_address'] = appraisal['subject']['address']
                properties.append(prop_data)
    
    # Convert to DataFrames
    subjects_df = pd.DataFrame(subjects)
    comps_df = pd.DataFrame(comps)
    properties_df = pd.DataFrame(properties)

    # subjects_df.head()
    # comps_df.head()
    # properties_df.head()

    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=1, idx=True)
    
    # Clean the DataFrames (basic cleaning)
    subjects_df = clean_dataframe(subjects_df)
    comps_df = clean_dataframe(comps_df)
    properties_df = clean_dataframe(properties_df)

    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=2, idx=True)
    
    # Apply specific field processing
    subjects_df, comps_df, properties_df = apply_specific_processing(subjects_df, comps_df, properties_df)
    
    # Convert column types
    mapping_df = pd.read_csv("./data/mappings/complete_field_mappings.csv") # contains user-filled data types (and replacement names if needed) for each col

    subjects_df = convert_column_types(subjects_df, mapping_df[mapping_df['section'] == 'subject'], 'subject')
    comps_df = convert_column_types(comps_df, mapping_df[mapping_df['section'] == 'comps'], 'comps')
    properties_df = convert_column_types(properties_df, mapping_df[mapping_df['section'] == 'properties'], 'properties')

    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=5, idx=True)
    
    return subjects_df, comps_df, properties_df

# Deduplication
def merge_duplicates_keep_most_complete(df, subset_cols):
    """
    Merge duplicate rows in a DataFrame based on subset_cols, keeping the row with the most non-null values.
    Logs the number of entries before and after merging.
    """
    print(f"Number of entries before merging duplicates: {len(df)}")
    
    # Create a completeness score (count of non-null values per row)
    df['completeness'] = df.notnull().sum(axis=1)
    
    # Sort by completeness descending so that the most complete rows come first
    df_sorted = df.sort_values(by='completeness', ascending=False)
    
    # Drop duplicates keeping the first (most complete) row
    df_deduped = df_sorted.drop_duplicates(subset=subset_cols, keep='first')
    
    # Drop the completeness column
    df_deduped = df_deduped.drop(columns=['completeness'])
    
    print(f"Number of entries after merging duplicates: {len(df_deduped)}")
    print(f"Removed {len(df) - len(df_deduped)} duplicate entries")
    
    return df_deduped

def handle_duplicates(subjects_df, comps_df, properties_df):
    '''
    Handle duplicates in all three dataframes by keeping the most complete records.
    '''
    # Define deduplication keys for each dataframe using the std_ fields
    subject_dedup_keys = ['std_street_number', 'std_street_name', 'std_city', 'std_postal_code']
    comps_dedup_keys = ['std_street_number', 'std_street_name', 'std_city', 'std_postal_code']
    properties_dedup_keys = ['std_street_number', 'std_street_name', 'std_city', 'std_postal_code']

    # Filter keys to only include columns that exist
    subject_dedup_keys = [k for k in subject_dedup_keys if k in subjects_df.columns]
    comps_dedup_keys = [k for k in comps_dedup_keys if k in comps_df.columns]
    properties_dedup_keys = [k for k in properties_dedup_keys if k in properties_df.columns]

    # Deduplicate each dataframe
    print("\nDeduplicating Subject Properties:")
    subjects_df = merge_duplicates_keep_most_complete(subjects_df, subject_dedup_keys)

    print("\nDeduplicating Comp Properties:")
    comps_df = merge_duplicates_keep_most_complete(comps_df, comps_dedup_keys)

    print("\nDeduplicating Available Properties:")
    properties_df = merge_duplicates_keep_most_complete(properties_df, properties_dedup_keys)

    return subjects_df, comps_df, properties_df

# Model
import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

def select_features_for_knn(subjects_df, properties_df):
    """
    Select appropriate features for KNN based on missing value analysis
    """
    # Features with relatively low missing values across both dataframes
    knn_features = [
        'gla',                # Core size metric
        'bedrooms',           # Important room count
        'full_baths',         # Important feature with few missing values in subjects
        'structure_type',     # Property type is usually available
        'cooling',            # Reasonable availability
        'heating'             # Good availability
    ]
    
    # Check which features exist and have sufficient data
    available_features = []
    for feature in knn_features:
        if feature in subjects_df.columns and feature in properties_df.columns:
            # Calculate missing percentage
            subj_missing = subjects_df[feature].isna().mean()
            prop_missing = properties_df[feature].isna().mean()
            
            # Only include features with less than 30% missing values
            if subj_missing < 0.3 and prop_missing < 0.3:
                available_features.append(feature)
    
    print(f"Selected KNN features: {available_features}")
    return available_features

def prepare_features(subjects_df, properties_df, selected_features):
    """
    Prepare features for model training with appropriate handling of missing values
    """
    # Create copies to avoid modifying original data
    subjects_copy = subjects_df.copy()
    properties_copy = properties_df.copy()
    
    # For numerical features, impute missing values with median
    numerical_features = [f for f in selected_features 
                         if f in subjects_df.columns and 
                         pd.api.types.is_numeric_dtype(subjects_df[f])]
    
    for feature in numerical_features:
        # Calculate median from combined data
        combined_values = pd.concat([subjects_df[feature], properties_df[feature]]).dropna()
        median_value = combined_values.median()
        
        # Impute missing values
        subjects_copy[feature] = subjects_copy[feature].fillna(median_value)
        properties_copy[feature] = properties_copy[feature].fillna(median_value)
    
    # For categorical features, impute with most frequent value
    categorical_features = [f for f in selected_features 
                           if f in subjects_df.columns and 
                           pd.api.types.is_categorical_dtype(subjects_df[f])]
    
    for feature in categorical_features:
        # Calculate most frequent value
        combined_values = pd.concat([subjects_df[feature], properties_df[feature]]).dropna()
        mode_value = combined_values.mode()[0] if not combined_values.empty else None
        
        # Impute missing values
        if mode_value is not None:
            subjects_copy[feature] = subjects_copy[feature].fillna(mode_value)
            properties_copy[feature] = properties_copy[feature].fillna(mode_value)
    
    return subjects_copy, properties_copy

def scale_features_for_knn(subjects_df, properties_df, numerical_features):
    """
    Scale numerical features to 0-1 range for KNN
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Combine data for fitting scaler
    combined_data = pd.concat([
        subjects_df[numerical_features],
        properties_df[numerical_features]
    ])
    
    # Fit scaler
    scaler.fit(combined_data)
    
    # Transform data
    subjects_scaled = subjects_df.copy()
    properties_scaled = properties_df.copy()
    
    subjects_scaled[numerical_features] = scaler.transform(subjects_df[numerical_features])
    properties_scaled[numerical_features] = scaler.transform(properties_df[numerical_features])
    
    return subjects_scaled, properties_scaled, scaler

def encode_categorical_features(subjects_df, properties_df, categorical_features):
    """
    Encode categorical features for model input
    """
    from sklearn.preprocessing import OneHotEncoder
    
    # Initialize encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine data for fitting encoder
    combined_data = pd.concat([
        subjects_df[categorical_features],
        properties_df[categorical_features]
    ])
    
    # Fit encoder
    encoder.fit(combined_data)
    
    # Transform data
    subjects_encoded = encoder.transform(subjects_df[categorical_features])
    properties_encoded = encoder.transform(properties_df[categorical_features])
    
    # Get feature names
    feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = encoder.categories_[i]
        feature_names.extend([f"{feature}_{category}" for category in categories])
    
    # Create DataFrames with encoded features
    subjects_encoded_df = pd.DataFrame(subjects_encoded, columns=feature_names, index=subjects_df.index)
    properties_encoded_df = pd.DataFrame(properties_encoded, columns=feature_names, index=properties_df.index)
    
    # Combine with original numerical features
    numerical_features = [f for f in subjects_df.columns if f not in categorical_features and pd.api.types.is_numeric_dtype(subjects_df[f])]
    
    subjects_final = pd.concat([subjects_df[numerical_features], subjects_encoded_df], axis=1)
    properties_final = pd.concat([properties_df[numerical_features], properties_encoded_df], axis=1)
    
    return subjects_final, properties_final, encoder, feature_names

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographical points using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point
        lat2, lon2: Latitude and longitude of the second point
        
    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def is_selected_comp(subject_row, candidate_row, comps_df):
    """
    Determine if a candidate property was selected as a comp for a subject property.
    Uses only standardized address fields for comparison.
    """
    # Match based on standardized address components
    matching_comps = comps_df[
        # Match the subject property
        (comps_df['subject_address'] == subject_row.get('std_full_address', '')) &
        (
            # Match on standardized address components
            (comps_df['std_street_number'] == candidate_row.get('std_street_number', '')) &
            (comps_df['std_street_name'] == candidate_row.get('std_street_name', '')) &
            (comps_df['std_city'] == candidate_row.get('std_city', ''))
        )
    ]
    
    return not matching_comps.empty

def create_similarity_features(subject_row, candidate_df, comps_df):
    """Create similarity features between a subject property and candidate properties"""
    similarity_features = []
    
    # Core features (low missing values)
    core_features = ['gla', 'bedrooms', 'full_baths', 'structure_type']
    
    # Secondary features (more missing values, but still valuable)
    secondary_features = [
        'room_count', 'half_baths', 'year_built', 
        'basement', 'cooling', 'heating'
    ]
    
    # Geographical features
    geo_features = ['latitude', 'longitude']
    
    for _, candidate_row in candidate_df.iterrows():
        features = {}
        
        # Add identifiers
        features['subject_address'] = subject_row['std_full_address']
        features['candidate_address'] = candidate_row['std_full_address']
        
        # Calculate core feature differences
        for col in core_features:
            if col in subject_row and col in candidate_row and not pd.isna(subject_row[col]) and not pd.isna(candidate_row[col]):
                if isinstance(subject_row[col], (int, float)) and isinstance(candidate_row[col], (int, float)):
                    # Numerical feature
                    features[f'{col}_diff'] = abs(float(subject_row[col]) - float(candidate_row[col]))
                    
                    # Add ratio for key size metrics
                    if col == 'gla' and float(subject_row[col]) > 0:
                        features[f'{col}_ratio'] = float(candidate_row[col]) / float(subject_row[col])
                else:
                    # Categorical feature
                    features[f'{col}_match'] = 1 if subject_row[col] == candidate_row[col] else 0
        
        # Calculate secondary feature differences (when available)
        for col in secondary_features:
            if col in subject_row and col in candidate_row and not pd.isna(subject_row[col]) and not pd.isna(candidate_row[col]):
                if isinstance(subject_row[col], (int, float)) and isinstance(candidate_row[col], (int, float)):
                    features[f'{col}_diff'] = abs(float(subject_row[col]) - float(candidate_row[col]))
                else:
                    features[f'{col}_match'] = 1 if subject_row[col] == candidate_row[col] else 0
        
        # Calculate geographical distance if coordinates are available
        if all(col in subject_row and col in candidate_row for col in geo_features):
            if not pd.isna(subject_row['latitude']) and not pd.isna(subject_row['longitude']) and \
               not pd.isna(candidate_row['latitude']) and not pd.isna(candidate_row['longitude']):
                features['geo_distance'] = calculate_distance(
                    subject_row['latitude'], subject_row['longitude'],
                    candidate_row['latitude'], candidate_row['longitude']
                )
        
        # Add label (1 if this candidate was selected as a comp, 0 otherwise)
        features['is_comp'] = 1 if is_selected_comp(subject_row, candidate_row, comps_df) else 0
        
        similarity_features.append(features)
    
    return pd.DataFrame(similarity_features)

def build_property_recommendation_system(subjects_df, properties_df, comps_df):
    """
    Build the two-stage property recommendation system with appropriate handling of missing values
    """
    # 1. Select features based on missing value analysis
    knn_features = select_features_for_knn(subjects_df, properties_df)
    
    # 2. Handle missing values
    subjects_processed, properties_processed = prepare_features(subjects_df, properties_df, knn_features)
    
    # 3. Split features into numerical and categorical
    numerical_features = [f for f in knn_features if pd.api.types.is_numeric_dtype(subjects_processed[f])]
    categorical_features = [f for f in knn_features if pd.api.types.is_categorical_dtype(subjects_processed[f])]
    
    # 4. Scale numerical features
    subjects_scaled, properties_scaled, scaler = scale_features_for_knn(
        subjects_processed, properties_processed, numerical_features
    )
    
    # 5. Encode categorical features
    subjects_final, properties_final, encoder, encoded_feature_names = encode_categorical_features(
        subjects_scaled, properties_scaled, categorical_features
    )
    
    # 6. Build KNN model
    knn_model = NearestNeighbors(n_neighbors=min(30, len(properties_final)), algorithm='auto')
    knn_model.fit(properties_final)
    
    # 7. Generate candidates using KNN
    candidates_by_subject = []
    for idx, subject_row in subjects_final.iterrows():
        distances, indices = knn_model.kneighbors([subject_row])
        candidates = properties_df.iloc[indices[0]].copy()
        candidates['subject_idx'] = idx
        candidates['knn_distance'] = distances[0]
        candidates_by_subject.append(candidates)
    
    all_candidates = pd.concat(candidates_by_subject, ignore_index=True)
    
    # 8. Create similarity features for LightGBM
    training_data = []
    for idx, subject_row in subjects_df.iterrows():
        # Get candidates for this subject
        subject_candidates = all_candidates[all_candidates['subject_idx'] == idx]
        
        # Create similarity features
        similarity_df = create_similarity_features(subject_row, subject_candidates, comps_df)
        training_data.append(similarity_df)
    
    combined_training_data = pd.concat(training_data, ignore_index=True)
    
    # 9. Train LightGBM ranking model
    X = combined_training_data.drop(['is_comp', 'subject_address', 'candidate_address'], axis=1)
    y = combined_training_data['is_comp']
    
    # Group by subject for ranking
    groups = combined_training_data.groupby('subject_idx').size().values
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    # Create LightGBM dataset
    lgb_train = lgb.Dataset(X, y, group=groups)
    
    # Set parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    # Train model
    gbm = lgb.train(params, lgb_train, num_boost_round=100)
    
    return {
        'knn_model': knn_model,
        'knn_scaler': scaler,
        'knn_encoder': encoder,
        'knn_features': knn_features,
        'lgb_model': gbm,
        'feature_importance': dict(zip(X.columns, gbm.feature_importance()))
    }
