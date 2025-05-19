import re
import pandas as pd
import numpy as np
import json

def save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number:int=None, idx:bool=False):
    '''Save processed data to CSV files for further analysis'''
    
    v_str, v_str_spaces = "", ""
    if version_number:
        v_str = f"_v{version_number}"
        v_str_spaces = f" v{version_number}"
    
    processed_path_prefix = "./data/processed/processed_"

    subjects_df.to_csv(f"{processed_path_prefix}subjects{v_str}.csv", index=idx)
    comps_df.to_csv(f"{processed_path_prefix}comps{v_str}.csv", index=idx)
    properties_df.to_csv(f"{processed_path_prefix}properties{v_str}.csv", index=idx)

    subjects_df.to_pickle(f"{processed_path_prefix}subjects{v_str}.pkl")
    comps_df.to_pickle(f"{processed_path_prefix}comps{v_str}.pkl")
    properties_df.to_pickle(f"{processed_path_prefix}properties{v_str}.pkl")

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

    # save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=3, idx=True)

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

    # save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=4, idx=True)
    
    return subjects_df, comps_df, properties_df

# Data type conversion function
# Define exclusion lists for model training
EXCLUDED_FEATURES_PREPROCESS = set([
    # Original address fields (replaced with standardized versions)
    'address', 'subject_city_province_zip', 'city_province', 'city', 'province', 'postal_code',
    
    # Fields we decided to exclude earlier
    "site_dimensions", "units_sq_ft", 
    "lot_size", "lot_size_sf",
    "water_heater", "exterior_finish", 
    "style", "levels",
    "effective_age", "subject_age", "age",
    
    # Date fields
    "effective_date", "close_date", "sale_date",
    
    # Ordinal fields
    'condition', 'condition_relative', 'location_similarity',
    
    # Categorical fields
    'prop_type', 'stories', 'basement_finish', 'parking', 'neighborhood', 'roof',
    'roofing', 'construction', 'windows', 'basement', 'foundation_walls', 
    'flooring', 'plumbing_lines', 'heating', 'cooling', 'fuel_type', 
    'water_heater', 'property_sub_type', 'structure_type',
    
    # Text fields
    "public_remarks"
])

ORDINAL_FEATURES = set(['condition', 'condition_relative', 'location_similarity'])

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
    cols_to_drop = [col for col in EXCLUDED_FEATURES_PREPROCESS if col in df.columns]
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
                # First convert to object type and handle NaN values
                df[col] = df[col].astype(object)
                # Replace 'nan' strings with actual NaN
                df[col] = df[col].replace('nan', np.nan)

                # # Then convert to category
                # df[col] = pd.Categorical(df[col])

                # Then convert non-null values to category
                mask = df[col].notna()
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].astype('category')
            elif dtype == 'ordinal':
                # Handle ordinal fields based on section and field name
                # Creating a new numeric version of each ordinal col

                if section_name == 'subject' and col in ORDINAL_FEATURES:
                    # For subject condition (absolute rating)
                    condition_map = {
                        'Poor': 1,
                        'Fair': 2,
                        'Average': 3,
                        'Good': 4,
                        'Excellent': 5
                    }
                    df[f'{col}_score'] = df[col].map(condition_map) # Numerical (new col)

                    # Original col will be made categorical
                    condition_order = list(condition_map.keys())
                    df[col] = df[col].astype(str).replace('nan', np.nan)
                    df[col] = pd.Categorical(df[col], categories=condition_order, ordered=True)

                elif section_name == 'comps' and col in ORDINAL_FEATURES:
                    # For comps comparison fields (relative rating) - condition_relative and location_similarity
                    comparison_map = {
                        'Inferior': 1,
                        'Similar': 2,
                        'Superior': 3
                    }
                    df[f'{col}_score'] = df[col].map(comparison_map) # Numerical (new col)

                    # Original col will be made categorical
                    comparison_order = list(comparison_map.keys())
                    df[col] = df[col].astype(str).replace('nan', np.nan)
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

# Main Processing function
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
        # Store the standardized subject address for use with comps and properties
        subject_std_full_address = None # will be updated in the 'subject' if block

        # Process subject property
        if 'subject' in appraisal:
            subject_data = appraisal['subject'].copy()
            # Add standardized address
            subject_data.update(process_subject_address(subject_data))
            subjects.append(subject_data)
            
            # Update the standardized address (with fallback to original if needed)
            subject_std_full_address = subject_data.get('std_full_address')
            if not subject_std_full_address and 'address' in subject_data:
                subject_std_full_address = subject_data['address']
        
        # Process comp properties
        if 'comps' in appraisal:
            for comp in appraisal['comps']:
                comp_data = comp.copy()
                # Add standardized address
                comp_data.update(process_comp_address(comp_data))
                # Add reference to subject property
                if 'subject' in appraisal and 'address' in appraisal['subject']:
                    comp_data['subject_address'] = subject_std_full_address
                comps.append(comp_data)
        
        # Process available properties
        if 'properties' in appraisal:
            for prop in appraisal['properties']:
                prop_data = prop.copy()
                # Add standardized address
                prop_data.update(process_property_address(prop_data))
                # Add reference to subject property
                if 'subject' in appraisal and 'address' in appraisal['subject']:
                    prop_data['subject_address'] = subject_std_full_address
                properties.append(prop_data)
    
    # Convert to DataFrames
    subjects_df = pd.DataFrame(subjects)
    comps_df = pd.DataFrame(comps)
    properties_df = pd.DataFrame(properties)

    # subjects_df.head()
    # comps_df.head()
    # properties_df.head()

    # save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=1, idx=True)
    
    # Clean the DataFrames (basic cleaning)
    subjects_df = clean_dataframe(subjects_df)
    comps_df = clean_dataframe(comps_df)
    properties_df = clean_dataframe(properties_df)

    # save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=2, idx=True)
    
    # Apply specific field processing
    subjects_df, comps_df, properties_df = apply_specific_processing(subjects_df, comps_df, properties_df)
    
    # Convert column types
    mapping_df = pd.read_csv("./data/mappings/complete_field_mappings.csv") # contains user-filled data types (and replacement names if needed) for each col

    subjects_df = convert_column_types(subjects_df, mapping_df[mapping_df['section'] == 'subject'], 'subject')
    comps_df = convert_column_types(comps_df, mapping_df[mapping_df['section'] == 'comps'], 'comps')
    properties_df = convert_column_types(properties_df, mapping_df[mapping_df['section'] == 'properties'], 'properties')

    # save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=5, idx=True)

    # Deduplicate    
    print("\n=== SUMMARY BEFORE DEDUPLICATION ===")
    print(f"Total subject properties: {len(subjects_df)}")
    print(f"Total comp properties: {len(comps_df)}")
    print(f"Total available properties: {len(properties_df)}")

    subjects_df, comps_df, properties_df = handle_duplicates(subjects_df, comps_df, properties_df)

    print("\n=== SUMMARY AFTER DEDUPLICATION ===")
    print(f"Total subject properties: {len(subjects_df)}")
    print(f"Total comp properties: {len(comps_df)}")
    print(f"Total available properties: {len(properties_df)})")

    # Final processed data
    save_appraisal_dfs(subjects_df, comps_df, properties_df, version_number=None, idx=False)
    
    return subjects_df, comps_df, properties_df

# If running as a script
if __name__ == "__main__":
    load_and_process_data()
