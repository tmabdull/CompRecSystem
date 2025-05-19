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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import math

def select_features_dynamically(subjects_df, properties_df, missing_threshold=0.3):
    """
    Dynamically select features for modeling based on data quality metrics.
    
    Args:
        subjects_df: DataFrame containing subject properties
        properties_df: DataFrame containing available properties
        missing_threshold: Maximum acceptable proportion of missing values (default: 0.3)
        
    Returns:
        Dictionary containing selected features for different model components
    """
    # Get all numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    # Check data types in subjects_df
    for col in subjects_df.columns:
        if col in EXCLUDED_FEATURES_PREPROCESS:
            continue
            
        if pd.api.types.is_numeric_dtype(subjects_df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_categorical_dtype(subjects_df[col]) or subjects_df[col].dtype == 'object':
            if col not in ORDINAL_FEATURES:  # Handle ordinals separately
                categorical_cols.append(col)
    
    # Calculate missing value proportions
    missing_proportions = {}
    
    for col in numeric_cols + categorical_cols:
        # Check if column exists in both dataframes
        if col in subjects_df.columns and col in properties_df.columns:
            subj_missing = subjects_df[col].isna().mean()
            prop_missing = properties_df[col].isna().mean()
            missing_proportions[col] = max(subj_missing, prop_missing)
    
    # Select features based on missing value threshold
    selected_features = {
        'numeric': [col for col in numeric_cols 
                   if col in missing_proportions 
                   and missing_proportions[col] < missing_threshold],
        'categorical': [col for col in categorical_cols 
                       if col in missing_proportions 
                       and missing_proportions[col] < missing_threshold],
        'ordinal': [col for col in ORDINAL_FEATURES 
                   if col in subjects_df.columns and col in properties_df.columns
                   and col in missing_proportions 
                   and missing_proportions[col] < missing_threshold]
    }
    
    # Log the selected features
    print(f"Selected {len(selected_features['numeric'])} numeric features: {selected_features['numeric']}")
    print(f"Selected {len(selected_features['categorical'])} categorical features: {selected_features['categorical']}")
    print(f"Selected {len(selected_features['ordinal'])} ordinal features: {selected_features['ordinal']}")
    
    return selected_features

def prepare_features_for_knn(subjects_df, properties_df, selected_features=None, missing_threshold=0.3):
    """
    Prepare features for KNN with dynamic feature selection if needed.
    
    Args:
        subjects_df: DataFrame containing subject properties
        properties_df: DataFrame containing available properties
        selected_features: Dictionary of pre-selected features (optional)
        missing_threshold: Maximum acceptable proportion of missing values
        
    Returns:
        Processed feature matrices and metadata
    """
    # Dynamically select features if not provided
    if selected_features is None:
        selected_features = select_features_dynamically(
            subjects_df, properties_df, missing_threshold
        )
    
    # Combine all selected features
    knn_features = (
        selected_features['numeric'] + 
        selected_features['categorical'] + 
        selected_features['ordinal']
    )
    
    # Filter to features that exist in both dataframes
    knn_features = [f for f in knn_features 
                   if f in subjects_df.columns and f in properties_df.columns]
    
    # Split into numeric and categorical features
    numeric_features = [f for f in knn_features 
                       if pd.api.types.is_numeric_dtype(subjects_df[f])]
    
    categorical_features = [f for f in knn_features 
                           if not pd.api.types.is_numeric_dtype(subjects_df[f])]
    
    # Create preprocessing pipeline with imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop other columns not specified
    )
    
    # Fit the preprocessor on both subjects and properties
    combined_data = pd.concat([
        subjects_df[knn_features], 
        properties_df[knn_features]
    ], axis=0)
    
    preprocessor.fit(combined_data)
    
    # Transform the data
    subjects_features = preprocessor.transform(subjects_df[knn_features])
    properties_features = preprocessor.transform(properties_df[knn_features])
    
    # Get feature names for interpretability
    feature_names = []
    
    # Add numeric feature names
    if numeric_features:
        feature_names.extend(numeric_features)
    
    # Add one-hot encoded feature names
    if categorical_features:
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        for i, col in enumerate(categorical_features):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    return subjects_features, properties_features, preprocessor, feature_names, selected_features

def is_selected_comp(subject_row, candidate_row, comps_df):
    """
    Determine if a candidate property was selected as a comp for a subject property.
    Uses standardized address fields for comparison.
    
    Args:
        subject_row: Row from subjects_df containing subject property data
        candidate_row: Row from properties_df containing candidate property data
        comps_df: DataFrame containing all comp properties
        
    Returns:
        Boolean indicating whether the candidate was selected as a comp
    """
    # Get standardized addresses
    subject_address = subject_row['std_full_address'] or ''
    candidate_address = candidate_row['std_full_address'] or ''
    
    # Check if this candidate appears in the comps dataframe for this subject
    matching_comps = comps_df[
        (comps_df['subject_address'] == subject_address) &
        (comps_df['std_full_address'] == candidate_address)
    ]
    
    # Alternative matching using street number, name, and city if needed
    if matching_comps.empty and 'std_street_number' in comps_df.columns:
        matching_comps = comps_df[
            (comps_df['subject_address'] == subject_address) &
            (comps_df['std_street_number'] == candidate_row.get('std_street_number', '')) &
            (comps_df['std_street_name'] == candidate_row.get('std_street_name', '')) &
            (comps_df['std_city'] == candidate_row.get('std_city', ''))
        ]
    
    return not matching_comps.empty

def create_similarity_features(subject_row, candidate_df, comps_df, selected_features=None):
    """
    Create similarity features between a subject property and candidate properties
    
    Args:
        subject_row: Row from subjects_df containing subject property data
        candidate_df: DataFrame containing candidate properties
        comps_df: DataFrame containing all comp properties
        selected_features: Dictionary of selected features by type
        
    Returns:
        DataFrame with similarity features
    """
    similarity_features = []
    
    # Define feature sets if not provided
    if selected_features is None:
        numeric_features = ['gla', 'bedrooms', 'full_baths', 'half_baths', 'year_built', 'room_count']
        categorical_features = ['structure_type', 'basement', 'cooling', 'heating']
    else:
        numeric_features = selected_features['numeric']
        categorical_features = selected_features['categorical'] + selected_features['ordinal']
    
    for _, candidate_row in candidate_df.iterrows():
        features = {}
        
        # Add identifiers
        features['subject_address'] = subject_row['std_full_address'] or ''
        features['candidate_address'] = candidate_row['std_full_address'] or ''
        features['subject_idx'] = subject_row.name
        
        # Calculate numerical differences/ratios
        for col in numeric_features:
            if col in subject_row and col in candidate_row and not pd.isna(subject_row[col]) and not pd.isna(candidate_row[col]):
                # Absolute difference
                features[f'{col}_diff'] = abs(float(subject_row[col]) - float(candidate_row[col]))
                
                # Ratio for size-related features
                if col in ['gla', 'lot_size_sf'] and float(subject_row[col]) > 0:
                    features[f'{col}_ratio'] = float(candidate_row[col]) / float(subject_row[col])
        
        # Calculate geographical distance if coordinates are available
        if all(col in subject_row and col in candidate_row for col in ['latitude', 'longitude']):
            if not pd.isna(subject_row['latitude']) and not pd.isna(subject_row['longitude']) and \
               not pd.isna(candidate_row['latitude']) and not pd.isna(candidate_row['longitude']):
                features['geo_distance'] = calculate_distance(
                    subject_row['latitude'], subject_row['longitude'],
                    candidate_row['latitude'], candidate_row['longitude']
                )
        
        # Categorical matches
        for col in categorical_features:
            if col in subject_row and col in candidate_row:
                features[f'{col}_match'] = 1 if subject_row[col] == candidate_row[col] else 0
        
        # Add label (1 if this candidate was selected as a comp, 0 otherwise)
        features['is_comp'] = 1 if is_selected_comp(subject_row, candidate_row, comps_df) else 0
        
        similarity_features.append(features)
    
    return pd.DataFrame(similarity_features)

def build_property_recommendation_system(subjects_df, comps_df, properties_df, missing_threshold=0.3):
    """
    Build the two-stage property recommendation system with dynamic feature selection
    
    Args:
        subjects_df: DataFrame containing subject properties
        properties_df: DataFrame containing available properties
        comps_df: DataFrame containing comp properties
        missing_threshold: Maximum acceptable proportion of missing values
        
    Returns:
        Dictionary containing trained models and metadata
    """
    # 1. Dynamically select features and prepare for KNN
    subjects_features, properties_features, knn_preprocessor, feature_names, selected_features = prepare_features_for_knn(
        subjects_df, properties_df, missing_threshold=missing_threshold
    )
    
    # 2. Build KNN model
    n_neighbors = min(30, len(properties_features))
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn_model.fit(properties_features)
    
    # 3. Generate candidates using KNN
    candidates_by_subject = []
    for i, subject_row in enumerate(subjects_features):
        distances, indices = knn_model.kneighbors([subject_row])
        candidates = properties_df.iloc[indices[0]].copy()
        candidates['subject_idx'] = i
        candidates['subject_address'] = subjects_df.iloc[i]['std_full_address'] or ''
        candidates['knn_distance'] = distances[0]
        candidates_by_subject.append(candidates)
    
    all_candidates = pd.concat(candidates_by_subject, ignore_index=True)
    
    # 4. Create similarity features for LightGBM
    training_data = []
    for idx, subject_row in subjects_df.iterrows():
        # Get candidates for this subject
        subject_candidates = all_candidates[all_candidates['subject_idx'] == idx]
        
        # Create similarity features
        similarity_df = create_similarity_features(
            subject_row, subject_candidates, comps_df, None
        )
        training_data.append(similarity_df)
    
    combined_training_data = pd.concat(training_data, ignore_index=True)
    
    # 5. Train LightGBM ranking model
    X = combined_training_data.drop(['is_comp', 'subject_address', 'candidate_address', 'subject_idx'], axis=1)
    y = combined_training_data['is_comp']
    
    # Group by subject for ranking
    groups = combined_training_data.groupby('subject_idx').size().values
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    # Create LightGBM dataset
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X, y, group=groups)
    
    # Set parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.2,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    # Train model
    gbm = lgb.train(params, lgb_train, num_boost_round=500)
    
    # Get feature importance
    feature_importance = dict(zip(X.columns, 
                                  gbm.feature_importance(importance_type='gain')))
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\nTop 10 most important features:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance}")
    
    return {
        'knn_model': knn_model,
        'knn_preprocessor': knn_preprocessor,
        'knn_feature_names': feature_names,
        'lgb_model': gbm,
        'feature_importance': feature_importance,
        'selected_features': selected_features
    }

# XGBoost Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Add standardized address components to exclusion list
address_components = set([
    'std_unit_number', 'std_street_number', 'std_street_name', 
    'std_city', 'std_province', 'std_postal_code'
])

MODEL_EXCLUSION_LIST = EXCLUDED_FEATURES_PREPROCESS.union(address_components)

def normalize_numerical_features(df):
    """
    Normalize numerical features to 0-1 range
    Returns the scaler and the normalized dataframe
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("Warning: No numeric columns found for normalization")
        return None, df
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Create a copy of the dataframe to avoid modifying the original
    df_normalized = df.copy()
    
    # Apply scaling to numeric columns
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].median()))
    
    return scaler, df_normalized

def create_feature_interactions(df):
    """
    Create simple interactions between important numerical features
    """
    # List of important features that might have meaningful interactions
    important_features = ['gla', 'room_count', 'bedrooms', 'full_baths', 'half_baths']
    
    # Only use features that exist in the dataframe
    existing_features = [f for f in important_features if f in df.columns]
    
    # Create interactions between pairs of features
    for i, feat1 in enumerate(existing_features):
        for feat2 in existing_features[i+1:]:
            # Skip if either column has all NaN values
            if df[feat1].isna().all() or df[feat2].isna().all():
                continue
                
            # Create ratio features (with handling for division by zero)
            df[f'{feat1}_per_{feat2}'] = df[feat1] / df[feat2].replace(0, np.nan)
            df[f'{feat2}_per_{feat1}'] = df[feat2] / df[feat1].replace(0, np.nan)
            
            # Create product feature
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    
    return df

def prepare_training_data(subjects_df, comps_df, properties_df):
    """
    Prepare training data for XGBoost model
    """
    print("Preparing training data...")
    
    # Get unique subject addresses (each represents one appraisal)
    all_subject_addresses = subjects_df['std_full_address'].unique()
    print(f"Total number of unique appraisals: {len(all_subject_addresses)}")
    
    # Split appraisals into train and validation sets
    train_addresses, val_addresses = train_test_split(
        all_subject_addresses, test_size=0.2, random_state=42
    )
    print(f"Training appraisals: {len(train_addresses)}, Validation appraisals: {len(val_addresses)}")
    
    # Filter dataframes based on the split
    train_subjects = subjects_df[subjects_df['std_full_address'].isin(train_addresses)]
    train_comps = comps_df[comps_df['subject_address'].isin(train_addresses)]
    train_properties = properties_df[properties_df['subject_address'].isin(train_addresses)]
    
    val_subjects = subjects_df[subjects_df['std_full_address'].isin(val_addresses)]
    val_comps = comps_df[comps_df['subject_address'].isin(val_addresses)]
    val_properties = properties_df[properties_df['subject_address'].isin(val_addresses)]
    
    print(f"Training subjects: {len(train_subjects)}, comps: {len(train_comps)}, properties: {len(train_properties)}")
    print(f"Validation subjects: {len(val_subjects)}, comps: {len(val_comps)}, properties: {len(val_properties)}")
    
    # Create positive examples (selected comps) and negative examples (non-selected properties)
    # For each subject in the training set
    train_X = []
    train_y = []
    
    for subject_address in train_addresses:
        # Get the subject property
        subject = train_subjects[train_subjects['std_full_address'] == subject_address].iloc[0]
        
        # Get comps for this subject (positive examples)
        subject_comps = train_comps[train_comps['subject_address'] == subject_address]
        
        # Get other properties for this subject (potential negative examples)
        subject_properties = train_properties[train_properties['subject_address'] == subject_address]
        
        # Process positive examples (selected comps)
        for _, comp in subject_comps.iterrows():
            # Create feature vector by comparing subject and comp
            features = create_comparison_features(subject, comp)
            train_X.append(features)
            train_y.append(1)  # 1 for selected comp
        
        # Process negative examples (non-selected properties)
        # Use properties that aren't in the comps list
        for _, prop in subject_properties.iterrows():
            # Skip if this property is already a comp (based on std_full_address)
            if any(comp['std_full_address'] == prop['std_full_address'] for _, comp in subject_comps.iterrows()):
                continue
                
            # Create feature vector by comparing subject and property
            features = create_comparison_features(subject, prop)
            train_X.append(features)
            train_y.append(0)  # 0 for non-selected property
    
    # Convert to dataframes
    train_X_df = pd.DataFrame(train_X)
    
    # Create validation data in the same way
    val_X = []
    val_y = []
    
    for subject_address in val_addresses:
        # Get the subject property
        subject = val_subjects[val_subjects['std_full_address'] == subject_address].iloc[0]
        
        # Get comps for this subject (positive examples)
        subject_comps = val_comps[val_comps['subject_address'] == subject_address]
        
        # Get other properties for this subject (potential negative examples)
        subject_properties = val_properties[val_properties['subject_address'] == subject_address]
        
        # Process positive examples (selected comps)
        for _, comp in subject_comps.iterrows():
            # Create feature vector by comparing subject and comp
            features = create_comparison_features(subject, comp)
            val_X.append(features)
            val_y.append(1)  # 1 for selected comp
        
        # Process negative examples (non-selected properties)
        for _, prop in subject_properties.iterrows():
            # Skip if this property is already a comp (based on std_full_address)
            if any(comp['std_full_address'] == prop['std_full_address'] for _, comp in subject_comps.iterrows()):
                continue
                
            # Create feature vector by comparing subject and property
            features = create_comparison_features(subject, prop)
            val_X.append(features)
            val_y.append(0)  # 0 for non-selected property
    
    # Convert to dataframes
    val_X_df = pd.DataFrame(val_X)
    
    # Create feature interactions
    train_X_df = create_feature_interactions(train_X_df)
    val_X_df = create_feature_interactions(val_X_df)
    
    # Normalize features
    scaler, train_X_normalized = normalize_numerical_features(train_X_df)
    _, val_X_normalized = normalize_numerical_features(val_X_df)
    
    return train_X_normalized, np.array(train_y), val_X_normalized, np.array(val_y), scaler

def create_comparison_features(subject, comp_or_property):
    """
    Create features that compare a subject property to a comp or potential comp
    """
    features = {}
    
    # Get all numerical features from both objects
    numerical_features = []
    for key in set(list(subject.keys()) + list(comp_or_property.keys())):
        # Skip excluded features
        if key in MODEL_EXCLUSION_LIST:
            continue
            
        # Check if the feature exists in both and is numeric
        if key in subject and key in comp_or_property:
            # Try to convert to numeric to check if it's a numerical feature
            try:
                float(subject[key]) if not pd.isna(subject[key]) else 0
                float(comp_or_property[key]) if not pd.isna(comp_or_property[key]) else 0
                numerical_features.append(key)
            except (ValueError, TypeError):
                pass
    
    # Add absolute differences for numerical features
    for feature in numerical_features:
        if feature in subject and feature in comp_or_property:
            # Handle NaN values
            subj_value = float(subject[feature]) if not pd.isna(subject[feature]) else 0
            comp_value = float(comp_or_property[feature]) if not pd.isna(comp_or_property[feature]) else 0
            
            # Calculate absolute difference
            features[f'{feature}_diff'] = abs(subj_value - comp_value)
            
            # Calculate percentage difference
            if subj_value != 0:
                features[f'{feature}_pct_diff'] = abs(subj_value - comp_value) / subj_value
            else:
                features[f'{feature}_pct_diff'] = np.nan
    
    # Add raw values from comp/property
    for feature in numerical_features:
        if feature in comp_or_property:
            features[feature] = float(comp_or_property[feature]) if not pd.isna(comp_or_property[feature]) else 0
    
    return features

def train_xgboost_model(train_X, train_y, val_X, val_y):
    """
    Train XGBoost model and evaluate on validation data
    """
    print("Training XGBoost model...")
    
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 30,  # Adjust based on your actual class ratio
        'random_state': 42
    }
    
    # Store feature names
    feature_names = list(train_X.columns)

    # Create DMatrix for XGBoost with explicit feature names
    dtrain = xgb.DMatrix(train_X, label=train_y, feature_names=feature_names)
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names)
    
    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Store feature names in the model for later use
    model.feature_names = feature_names
    
    # Make predictions on validation set
    val_preds_prob = model.predict(dval)
    val_preds = (val_preds_prob > 0.01).astype(int)
    
    # Evaluate model
    accuracy = accuracy_score(val_y, val_preds)
    precision = precision_score(val_y, val_preds, zero_division=0)
    recall = recall_score(val_y, val_preds)
    f1 = f1_score(val_y, val_preds)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    return model, val_preds_prob, accuracy, precision, recall, f1

def recommend_comps(model, scaler, subject, candidate_properties, top_n=3, max_distance_km=5.0):
    """
    Recommend top N comparable properties for a given subject property
    
    Args:
        model: Trained XGBoost model
        scaler: Feature scaler used during training
        subject: Subject property (Series or dict)
        candidate_properties: DataFrame of candidate properties
        top_n: Number of recommendations to return (default: 3)
        max_distance_km: Maximum distance in kilometers to consider (default: 5.0)
    
    Returns:
        DataFrame with top N recommended properties
    """
    # Filter by distance if distance information is available
    filtered_candidates = candidate_properties.copy()
    
    # If we have latitude/longitude coordinates
    if all(col in filtered_candidates.columns for col in ['latitude', 'longitude']) and \
       all(col in subject for col in ['latitude', 'longitude']):
        # Calculate distance using Haversine formula
        filtered_candidates['distance_km'] = filtered_candidates.apply(
            lambda row: calculate_distance(
                subject['latitude'], subject['longitude'],
                row['latitude'], row['longitude']
            ),
            axis=1
        )
        # Filter by calculated distance
        filtered_candidates = filtered_candidates[filtered_candidates['distance_km'] <= max_distance_km]
    
    # If we have distance_to_subject field in the candidates
    elif 'distance_to_subject' in filtered_candidates.columns:
        # Ensure distance is in km and is numeric
        filtered_candidates = filtered_candidates[
            pd.to_numeric(filtered_candidates['distance_to_subject'], errors='coerce') <= max_distance_km
        ]
    
    # If no candidates remain after distance filtering, return empty DataFrame
    if len(filtered_candidates) == 0:
        print(f"Warning: No candidates within {max_distance_km} km of subject property")
        return pd.DataFrame()
    
    # Create comparison features for each candidate property
    candidate_features = []
    for _, prop in filtered_candidates.iterrows():
        features = create_comparison_features(subject, prop)
        candidate_features.append(features)
    
    # Convert to dataframe
    candidate_features_df = pd.DataFrame(candidate_features)
    
    # Create feature interactions
    candidate_features_df = create_feature_interactions(candidate_features_df)

    # Get the feature names used during training, in the same order
    train_features = model.feature_names

    # Ensure all training features exist in candidate features
    for feature in train_features:
        if feature not in candidate_features_df.columns:
            candidate_features_df[feature] = 0  # Add missing features with default values
    
    # Select only the features used in training and in the same order
    candidate_features_df = candidate_features_df[train_features]
    
    # Normalize features using the same scaler used during training
    candidate_features_normalized = candidate_features_df.copy()
    
    # Apply scaling if there are numeric columns
    if hasattr(scaler, 'feature_names_in_'):
        numeric_cols = candidate_features_df.columns
        candidate_features_normalized = pd.DataFrame(
            scaler.transform(candidate_features_df.fillna(candidate_features_df.median())),
            columns=numeric_cols
        )
    
    # Create DMatrix
    dcandidate = xgb.DMatrix(candidate_features_normalized)
    
    # Get predictions
    candidate_scores = model.predict(dcandidate)
    
    # Add scores to candidate properties
    filtered_candidates['comp_score'] = candidate_scores
    
    # Sort by score and get top N
    top_comps = filtered_candidates.sort_values('comp_score', ascending=False).head(top_n)
    
    return top_comps

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def evaluate_recommendations(model, scaler, val_subjects, val_comps, val_properties):
    """
    Evaluate the recommendation system on validation data
    
    Args:
        model: Trained XGBoost model
        scaler: Feature scaler used during training
        val_subjects: DataFrame of validation subject properties
        val_comps: DataFrame of validation comp properties
        val_properties: DataFrame of validation candidate properties
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating recommendation system...")
    
    # Initialize metrics
    total_subjects = 0
    total_recommendations = 0
    total_hits = 0
    precision_at_3_sum = 0
    
    # Process each subject property
    for _, subject in val_subjects.iterrows():
        subject_address = subject['std_full_address']
        
        # Get actual comps for this subject
        actual_comps = val_comps[val_comps['subject_address'] == subject_address]
        
        # Skip if no actual comps
        if len(actual_comps) == 0:
            continue
            
        # Get candidate properties for this subject
        candidates = val_properties[val_properties['subject_address'] == subject_address]
        
        # Skip if no candidates
        if len(candidates) == 0:
            continue
            
        # Get recommended comps
        recommended_comps = recommend_comps(model, scaler, subject, candidates, top_n=3)
        
        # Skip if no recommendations
        if len(recommended_comps) == 0:
            continue
            
        # Count hits (recommended comps that match actual comps)
        hits = 0
        for _, rec_comp in recommended_comps.iterrows():
            # Check if this recommendation matches any actual comp
            for _, act_comp in actual_comps.iterrows():
                # Match based on standardized full address
                if ('std_full_address' in rec_comp and 'std_full_address' in act_comp and 
                    rec_comp['std_full_address'] == act_comp['std_full_address']):
                    hits += 1
                    break
                # Fallback to regular address if standardized not available
                elif 'id' in rec_comp and 'id' in act_comp and rec_comp['id'] == act_comp['id']:
                    hits += 1
                    break
        
        # Update metrics
        total_subjects += 1
        total_recommendations += len(recommended_comps)
        total_hits += hits
        precision_at_3 = hits / min(3, len(recommended_comps))
        precision_at_3_sum += precision_at_3
    
    # Calculate aggregate metrics
    if total_subjects > 0:
        avg_precision_at_3 = precision_at_3_sum / total_subjects
        overall_precision = total_hits / total_recommendations if total_recommendations > 0 else 0
        recall = total_hits / (total_subjects * 3) if total_subjects > 0 else 0
        
        print(f"Evaluated on {total_subjects} subject properties")
        print(f"Average Precision@3: {avg_precision_at_3:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        
        return {
            'num_subjects': total_subjects,
            'avg_precision_at_3': avg_precision_at_3,
            'overall_precision': overall_precision,
            'overall_recall': recall
        }
    else:
        print("No valid subjects for evaluation")
        return {
            'num_subjects': 0,
            'avg_precision_at_3': 0,
            'overall_precision': 0,
            'overall_recall': 0
        }

# Main execution
import datetime
def train_and_evaluate_xgboost():
    # Load and process data
    subjects_df, comps_df, properties_df = load_and_process_data()

    # Final processed data
    save_dfs_to_csv(subjects_df, comps_df, properties_df, version_number=None, idx=False)
    
    # Prepare training data
    train_X, train_y, val_X, val_y, scaler = prepare_training_data(subjects_df, comps_df, properties_df)

    # Train XGBoost model
    model, val_preds_prob, accuracy, precision, recall, f1 = train_xgboost_model(train_X, train_y, val_X, val_y)

    # Save model and scaler for later use
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save_model(f"./data/models/comp_recommendation_model_{timestamp}_{accuracy:.2f}_{precision:.2f}_{recall:.2f}_{f1:.2f}.json")

    # Get validation subjects, comps, and properties
    all_subject_addresses = subjects_df['std_full_address'].unique()
    train_addresses, val_addresses = train_test_split(
        all_subject_addresses, test_size=0.2, random_state=42
    )

    val_subjects = subjects_df[subjects_df['std_full_address'].isin(val_addresses)]
    val_comps = comps_df[comps_df['subject_address'].isin(val_addresses)]
    val_properties = properties_df[properties_df['subject_address'].isin(val_addresses)]

    # Evaluate recommendations on validation set
    evaluation_metrics = evaluate_recommendations(model, scaler, val_subjects, val_comps, val_properties)
    print(evaluation_metrics)

    # Example of recommending comps for a specific subject property
    if len(val_subjects) > 0:
        example_subject = val_subjects.iloc[0]
        example_candidates = val_properties[val_properties['subject_address'] == example_subject['std_full_address']]
        
        print("\nExample Recommendations:")
        print(f"Subject Property: {example_subject['std_full_address']}")
        
        recommended_comps = recommend_comps(model, scaler, example_subject, example_candidates, top_n=3)
        
        if len(recommended_comps) > 0:
            print("\nRecommended comps:")
            display_cols = ['std_full_address', 'comp_score']
            display_cols = [col for col in display_cols if col in recommended_comps.columns]
            print(recommended_comps[display_cols])
            
            # Show actual comps for comparison
            actual_comps = val_comps[val_comps['subject_address'] == example_subject['std_full_address']]
            if len(actual_comps) > 0:
                print("\nActual comps selected by appraiser:")
                display_cols = [col for col in display_cols if col in actual_comps.columns]
                print(actual_comps[display_cols])

# If running as a script
if __name__ == "__main__":
    train_and_evaluate_xgboost()
