import pandas as pd

## Import all downloaded datasets from FAOSTAT and World Bank
gdp = pd.read_csv('downloads/FAOSTAT_GDP.csv')
land = pd.read_csv('downloads/FAOSTAT_LANDUSE.csv')
woodfuel = pd.read_csv('downloads/FAOSTAT_WOODFUEL.csv')
rural = pd.read_csv('downloads/WB_RURAL.csv')
temp = rural = pd.read_csv('downloads/WB_TEMP.csv')


### HELPER FUNCTIONS

def pivot_fao_dataset(df: pd.DataFrame, on='Item', keep_flag = False):
    """Simple helper to pivot datasets from FAOSTAT according to the desired columns"""

    sub_cols = ['Area Code (M49)', 'Area', 'Year', 'Element', 'Item', 'Value']
    pivot_index = ['Area Code (M49)', 'Area', 'Year']

    if keep_flag:
        sub_cols.append('Flag')
        pivot_index.append('Flag')

    subset = df[sub_cols]
    
    df_pivoted = subset.pivot(
        index= pivot_index, 
        columns=on,                     
        values='Value'                           
    ).reset_index()                                

    return df_pivoted

def clean_world_bank(df, feature_name):
    """Helper dropping irrelevant columns from World Bank and preparing for merge with FAO"""
    
    # 1. Pull only the 3 columns your model actually cares about
    clean_df = df[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    
    # 2. Rename columns to map perfectly to your existing FAO master pipeline
    clean_df.columns = ['ISO3', 'Year', feature_name]
    
    # 3. Clean up the types to ensure smooth matrix alignments
    clean_df['Year'] = clean_df['Year'].astype(int)
    clean_df[feature_name] = pd.to_numeric(clean_df[feature_name], errors='coerce')
    
    return clean_df


### DOWNLOAD M49 -> ISO3 TABLE MAP FOR MERGING

url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
bridge_df = pd.read_csv(url)

bridge_df = bridge_df[['country-code', 'alpha-3']].copy()
bridge_df.columns = ['Area Code (M49)', 'ISO3']


### CLEANING AND MERGING ALL DATAFRAMES (On Year + Country)

gdp_pivoted = pivot_fao_dataset(gdp, on='Element') # Pivot all FAO
land_pivoted = pivot_fao_dataset(land)
woodfuel_pivoted = pivot_fao_dataset(woodfuel, keep_flag=True)

fao_df = gdp_pivoted.copy() # Merge all FAO data, GDP as baseline (because no N/A)
fao_df = pd.merge(fao_df, land_pivoted, on=['Area Code (M49)', 'Area', 'Year'], how='left')
fao_df = pd.merge(fao_df, woodfuel_pivoted, on=['Area Code (M49)', 'Area', 'Year'], how='left')

fao_df = pd.merge(fao_df, bridge_df, on='Area Code (M49)', how='left') # Add the ISO3 codes

clean_rural = clean_world_bank(rural, 'rural_pop_percent')
clean_temp = clean_world_bank(temp, 'mean_temp')

final_df = fao_df.copy()
final_df = pd.merge(final_df, clean_rural, on=['ISO3', 'Year'], how='left')
final_df = pd.merge(final_df, clean_temp, on=['ISO3', 'Year'], how='left')

### FINAL FORMATTING AND EXPORT

columns_to_drop = [
    'Area Code (M49)', 
    'Area', 
    '15.2.1 Annual forest area change rate', 
    '15.1.1 Land area', 
    'Flag'
]

df_cleaned = final_df.drop(columns=columns_to_drop, errors='ignore')

rename_dict = {
    'Value US$ per capita': 'gdp_per_capita_current',
    'Value US$ per capita, 2015 prices': 'gdp_per_capita_constant_2015',
    '15.1.1 Forest area': 'forest_area_hectares',
    '15.1.1 Forest area as a proportion of total land area': 'forest_proportion',
    'Wood fuel, coniferous': 'woodfuel_coniferous_m3',
    'Wood fuel, non-coniferous': 'woodfuel_non_coniferous_m3'
}

df_cleaned.rename(columns=rename_dict, inplace=True)

df_cleaned.to_csv('feature_dataset.csv')