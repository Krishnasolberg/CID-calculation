"""
We need 
- Net positions (JAO)
- Shadowprices (JAO)
- PTDFs (JAO)
- Bidding zone prices (Nord Pool)
"""

import requests
import pandas as pd
import warnings
import time
import json
import pytz
import xlrd
import plotly.express as px
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta
from alive_progress import alive_bar
from datetime import datetime, timedelta
from functools import reduce
from Nordpool_V2 import Nordpool_API_V2
from pathlib import Path
from cid_calc_old_method import compute_flow_for_sharing_key_old, calculate_CI_nordic_old


def get_jao_data(start_date:datetime, end_date:datetime, token):
    url = f"https://publicationtool.jao.eu/nordic/api/data/fbDomainShadowPrice?Filter=%7B%7D&\
        Skip=0&Take=10000000&FromUtc={start_date}T00%3A00%3A00.000Z&ToUtc={end_date}T23%3A00%3A00.000Z"
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    session.verify = False
    content = requests.get(url, headers=headers, verify=False).text #noqa
    soup = BeautifulSoup(content, 'html.parser')
    soup = str(soup)
    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    return soup


def get_jao_border_cnec_data(start_date:datetime, end_date:datetime, token):
    url = f"https://publicationtool.jao.eu/nordic/api/data/fbDomainShadowPrice?Filter=%7B%22\
        CnecName%22%3A%22Border_CNEC%22%7D&Skip=0&Take=10000000&FromUtc={start_date}T23\
            %3A00%3A00.000Z&ToUtc={end_date}T00%3A00%3A00.000Z"
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    session = requests.Session()
    session.verify = False
    content = requests.get(url, headers=headers, verify=False).text #noqa
    soup = BeautifulSoup(content, 'html.parser')
    soup = str(soup)
    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    return soup


def create_dataframe_from_jao_data(start_date, end_date, token):
    dfs = []
    delta = end_date - start_date
    num_days = delta.days
    num_batches = (num_days // 20) + 1
    with alive_bar(num_batches) as bar:
        time.sleep(0.005)
        for batch in range(num_batches):
            batch_start_date = start_date + timedelta(days=batch * 20)
            batch_end_date = min(start_date + timedelta(days=(batch + 1) * 20), end_date)
            batch_start_date_str = batch_start_date.strftime('%Y-%m-%d')
            batch_end_date_str = batch_end_date.strftime('%Y-%m-%d')
            soup = get_jao_data(batch_start_date_str, batch_end_date_str, token)
            data = json.loads(soup)
            data_list = data.get("data", [])
            df = pd.DataFrame(data_list)
            dfs.append(df)
            print(f"Dataframe created for {batch_start_date_str} to {batch_end_date_str}")
            bar()
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


def create_border_dataframe_from_jao_data(start_date, end_date, token):
    dfs = []
    delta = end_date - start_date
    num_days = delta.days
    num_batches = (num_days // 20) + 1
    with alive_bar(num_batches) as bar:
        time.sleep(0.005)
        for batch in range(num_batches):
            batch_start_date = start_date + timedelta(days=batch * 20)
            batch_end_date = min(start_date + timedelta(days=(batch + 1) * 20), end_date)
            batch_start_date_str = batch_start_date.strftime('%Y-%m-%d')
            batch_end_date_str = batch_end_date.strftime('%Y-%m-%d')
            soup = get_jao_border_cnec_data(batch_start_date_str, batch_end_date_str, token)
            data = json.loads(soup)
            data_list = data.get("data", [])
            df = pd.DataFrame(data_list)
            dfs.append(df)
            print(f"Dataframe created for {batch_start_date_str} to {batch_end_date_str}")
            bar()
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


def extract_jao_net_position(df):
    """
    This function filters the input dataframe to include only the columns 'dateTimeUtc', 'cnecName', and 'flowFb',
    selects only rows where 'cnecName' contains 'NetPosition', removes 'NetPosition_' from each value in 'cnecName',
    renames 'cnecName' to 'Bidding Zone' and 'flowFb' to 'Net Position', sets 'dateTimeUtc' as datetime format,
    and makes it the index of the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe with multiple columns.

    Returns:
    pd.DataFrame: A new dataframe with 'dateTimeUtc' as index and specified transformations applied.
    """

    filtered_df = df[['dateTimeUtc', 'cnecName', 'flowFb']].copy()
    filtered_df = filtered_df[filtered_df['cnecName'].str.contains("NetPosition", na=False)]
    filtered_df['cnecName'] = filtered_df['cnecName'].str.replace("NetPosition_", "", regex=False)
    filtered_df.rename(columns={'cnecName': 'Bidding Zone', 'flowFb': 'Net Position'}, inplace=True)
    filtered_df['dateTimeUtc'] = pd.to_datetime(filtered_df['dateTimeUtc'])
    filtered_df.set_index('dateTimeUtc', inplace=True)

    return filtered_df


def extract_jao_ptdfs(df, border=False, remove_virtual=False, only_virtual=False, include_flow=False):
    """
    This function filters the input dataframe to include only the columns 'dateTimeUtc', 'cnecName', 
    and the last 31 columns, selects only rows where 'cnecName' contains 'NetPosition', 
    removes 'NetPosition_' from each value in 'cnecName', renames 'cnecName' to 'Bidding Zone' 
    and 'flowFb' to 'Net Position', sets 'dateTimeUtc' as datetime format, 
    and makes it the index of the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe with multiple columns.

    Returns:
    pd.DataFrame: A new dataframe with 'dateTimeUtc' as index and specified transformations applied.
    """

    # Select the columns 'dateTimeUtc', 'cnecName', and the last 31 columns
    if include_flow == True:
        columns_to_select = ['dateTimeUtc', 'cnecName', 'fall', 'flowFb', 'shadowPrice'] + df.columns[-31:].tolist()
    else:
        columns_to_select = ['dateTimeUtc', 'cnecName', 'fall', 'shadowPrice'] + df.columns[-31:].tolist()
    filtered_df = df[columns_to_select].copy()
    if border==True:
        filtered_df = filtered_df[filtered_df['cnecName'].str.contains('Border_CNEC', na=False)]
    filtered_df.columns = [col.replace('ptdf_', '') for col in filtered_df.columns]
    if remove_virtual == True:
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.contains('_')]

    if only_virtual == True:
        filtered_df = filtered_df[filtered_df['cnecName'].str.count('_') == 3]

    # Convert 'dateTimeUtc' to datetime and set it as the index
    filtered_df['dateTimeUtc'] = pd.to_datetime(filtered_df['dateTimeUtc'])
    filtered_df.set_index('dateTimeUtc', inplace=True)

    return filtered_df


def calculate_regional_net_positions(df, regional_groups, remove_virtual=False):
    # Step 1: Pivot the dataframe so each unique bidding zone becomes a column
    df_pivot = df.pivot_table(index=df.index, columns="Bidding Zone", values="Net Position", aggfunc="sum")
    
    # Step 2: Update the columns based on the regional_np_groups dictionary
    for main_zone, sub_zones in regional_groups.items():
        # Ensure the main zone column exists in the dataframe
        if main_zone in df_pivot.columns:
            # Sum the values of the main zone and all sub_zones specified in the dictionary
            df_pivot[main_zone] = df_pivot[sub_zones].sum(axis=1, skipna=True)
            # Drop the sub_zone columns if needed
            df_pivot.drop(
                columns=[col for col in df_pivot.columns if col in sub_zones  and col != main_zone],
                inplace=True
            )
        else:
            # Create the main zone column if it doesn't exist and populate with summed values of sub_zones
            df_pivot[main_zone] = df_pivot[sub_zones].sum(axis=1, skipna=True)
    if remove_virtual == True:
        df_pivot = df_pivot.loc[:, ~df_pivot.columns.str.contains('_')]
    
    return df_pivot


def extract_jao_shadow_prices(df):
    """
    This function filters the input dataframe to include only the columns 'dateTimeUtc', 'cnecName', and 'shadowPrice',
    sets 'dateTimeUtc' as datetime format,
    and makes it the index of the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe with multiple columns.

    Returns:
    pd.DataFrame: A new dataframe with 'dateTimeUtc' as index and specified transformations applied.
    """

    filtered_df = df[['dateTimeUtc', 'cnecName', 'shadowPrice', 'tso']].copy()
    filtered_df['dateTimeUtc'] = pd.to_datetime(filtered_df['dateTimeUtc'])
    filtered_df.set_index('dateTimeUtc', inplace=True)

    return filtered_df


def clean_nordpool_prices(df):
    filtered_df = df[['deliveryStart', 'deliveryArea', 'price']].copy()
    filtered_df['dateTimeutc'] = pd.to_datetime(filtered_df['deliveryStart']).dt.tz_localize('UTC')
    filtered_df.set_index('dateTimeutc', inplace=True)
    filtered_df = filtered_df.drop(columns=["deliveryStart"])
    return filtered_df

"""
def plot_bidding_area_prices(df):
    # Reset index to use deliveryStart as a column for plotting
    df = df.reset_index()
    
    # Create a line plot
    fig = px.line(df, x='deliveryStart', y='price', color='deliveryArea', 
                  title="Prices for Each Bidding Zone", 
                  labels={"deliveryStart": "Timestamp", "price": "Price"})
    
    # Show the plot
    fig.show()
"""

def calculate_CI_virtual_component(price_df, ptdf_df):
    no1_prices = price_df[price_df['deliveryArea'] == 'NO1']
    no1_prices = no1_prices.groupby(no1_prices.index)['price'].first()
    ptdf_df_filtered = ptdf_df[ptdf_df['shadowPrice'] > 0]

    exclude_patterns = ['_EXP', 'AC_Minimum', 'AC_Maximum']
    for pattern in exclude_patterns:
        ptdf_df_filtered = ptdf_df_filtered[
            ~ptdf_df_filtered['cnecName'].str.contains(pattern, na=False)
        ]

    relevant_columns = [col for col in ptdf_df.columns if '_' in col]

    lambda_values = (
        ptdf_df_filtered.groupby(ptdf_df_filtered.index)
        .apply(lambda group: (group['NO1'] * group['shadowPrice']).sum())
    )

    ptdf_df['lambda'] = ptdf_df.index.map(no1_prices) + ptdf_df.index.map(lambda_values)

    virtual_prices = pd.DataFrame(index=ptdf_df.index.unique(), columns=relevant_columns)

    for col in relevant_columns:
        summed_values = (
            ptdf_df_filtered.groupby(ptdf_df_filtered.index)
            .apply(lambda group: (group[col] * group['shadowPrice']).sum())
        )

        virtual_prices[col] = ptdf_df['lambda'].groupby(ptdf_df.index).first() - summed_values

    return virtual_prices




def calculate_CI_nordic_regional(regional_np_df, price_df):
    regional_np_df = regional_np_df.copy()
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    regional_np_df.index = pd.to_datetime(regional_np_df.index)
    common_index = regional_np_df.index.intersection(price_df.index)
    regional_np_df = regional_np_df.loc[common_index]
    price_df = price_df.loc[common_index]
    adjusted_df = pd.DataFrame(index=common_index)
    
    for area in price_df['deliveryArea'].unique():
        if area in regional_np_df.columns:
            negative_price = price_df.loc[price_df['deliveryArea'] == area, 'price']
            adjusted_values = negative_price.values * regional_np_df[area].values
            adjusted_df[area] = adjusted_values
    result_df = pd.DataFrame((-1)*adjusted_df.sum(axis=1), columns=['Total CI'])
            
    return result_df


def calcluate_CI_nordic_virtual(price_df, virtual_prices, net_position_df):
    exclude_substrings = {"SWL", "KS", "SK", "FS", "SB"}

    filtered_columns = [
        col for col in virtual_prices.columns
        if not any(substring in col for substring in exclude_substrings)
    ]
    virtual_prices = virtual_prices[filtered_columns]
    
    common_index = virtual_prices.index.intersection(price_df.index).intersection(net_position_df.index)
    virtual_prices = virtual_prices.loc[common_index]
    price_df = price_df.loc[common_index]
    net_position_df = net_position_df.loc[common_index]
    results = []

    for timestamp in virtual_prices.index:
        total_sum = 0
        virtual_prices_row = virtual_prices.loc[timestamp]
        price_row = price_df[price_df.index == timestamp]
        net_position_rows = net_position_df[net_position_df.index == timestamp]
        
        for area in virtual_prices.columns:
            P_v = virtual_prices_row[area]
            home_area = area.split('_')[0] 
            P_v_home = price_row.loc[price_row['deliveryArea'] == home_area, 'price'].values[0]
            
            NP_v = net_position_rows.loc[net_position_rows['Bidding Zone'] == area, 'Net Position'].values[0]
            
            contribution = (P_v - P_v_home) * -NP_v
            total_sum += contribution
        
        results.append(total_sum)
    
    results_df = pd.DataFrame(data=results, index=virtual_prices.index, columns=['Sum'])
    return results_df


def calculate_CI_nordic(ci_regional_df, ci_virtual_df):
    CI_nordic = ci_regional_df['Total CI'] + ci_virtual_df['Sum']
    return CI_nordic


def calculate_regional_flow(regional_np_df, border_ptdf_df):
    regional_np_df = regional_np_df.copy()
    border_ptdf_df = border_ptdf_df.copy()
    regional_np_df.index = pd.to_datetime(regional_np_df.index)
    border_ptdf_df.index = pd.to_datetime(border_ptdf_df.index)
    cnec_name_column = border_ptdf_df['cnecName']
    f0_column = border_ptdf_df['fall']
    border_ptdf_df = border_ptdf_df.drop(columns=['cnecName', 'fall'])
    border_ptdf_df = border_ptdf_df[regional_np_df.columns]
    
    regional_np_values = regional_np_df.values
    border_ptdf_values = border_ptdf_df.values
    timestamps = regional_np_df.index
    timestamp_to_idx = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    result_values = np.empty_like(border_ptdf_values, dtype=np.float64)
    
    for i, timestamp in enumerate(border_ptdf_df.index):
        regional_idx = timestamp_to_idx.get(timestamp)
        
        if regional_idx is not None:
            result_values[i] = border_ptdf_values[i] * regional_np_values[regional_idx]
        else:
            result_values[i] = np.nan
    
    result_df = pd.DataFrame(result_values, index=border_ptdf_df.index, columns=border_ptdf_df.columns)
    
    result_df.insert(0, 'cnecName', cnec_name_column.values)
    result_df.insert(1, 'fall', f0_column.values)
    result_df['Flow'] = result_df.iloc[:, 1:].sum(axis=1)
    result_df['Flow_ex_f0'] = result_df["Flow"] - result_df['fall']
    
    return result_df



def calculate_nordic_congestion_income(CI_df, sharing_key_df):
    merged_df = sharing_key_df.merge(CI_df, left_index=True, right_index=True, how="left")
    merged_df['ci_shared'] = merged_df['Total CI'] * merged_df['normalized_value']

    return merged_df


def compute_flow_for_sharing_key(price_df, flow_df, net_position_df, exclude_list, replace_dict, internal_hvdc, bidding_zone_mapping):
    """
    Compute F_i using flow_df and net_position_df, ensuring alignment with price_df.

    Parameters:
        price_df (pd.DataFrame): DataFrame with a single datetime index and price information.
        flow_df (pd.DataFrame): DataFrame with 'cnecName' and 'Flow' columns, potentially other metadata.
        net_position_df (pd.DataFrame): DataFrame with 'Flow_FB' column and a datetime index.
        removal_list (list): List of strings to filter out from the 'cnecName' column in flow_df.
        rename_dict (dict): Dictionary to map old 'cnecName' values to new ones in flow_df.

    Returns:
        pd.DataFrame: DataFrame combining F_k and F_l.
    """
    flow_df['cnecName'] = flow_df['cnecName'].str.replace("Border_CNEC_", "", regex=False)

    if exclude_list:
        flow_df = flow_df[~flow_df['cnecName'].str.contains('|'.join(exclude_list))]
    
    # Replace specific names in 'cnecName' based on replace_dict
    if replace_dict:
        for old_value, new_value in replace_dict.items():
            flow_df['cnecName'] = flow_df['cnecName'].str.replace(old_value, new_value, regex=False)
    
    net_position_df = net_position_df[net_position_df['Bidding Zone'].isin(internal_hvdc)]
    
    # Step 4: Align indices
    common_indices = price_df.index.intersection(flow_df.index).intersection(net_position_df.index)
    flow_df = flow_df.loc[common_indices]
    net_position_df = net_position_df.loc[common_indices]
    price_df = price_df.loc[common_indices]
    
    # Step 5: Split 'cnecName' into 'from_ba' and 'to_ba'
    flow_df[['from_ba', 'to_ba']] = flow_df['cnecName'].str.split('-', expand=True)
    area_mapping = {"SE3_SWL": "SE4", "SE4_SWL": "SE3"}
    flow_df['from_ba'] = flow_df['from_ba'].replace(area_mapping)
    flow_df['to_ba'] = flow_df['to_ba'].replace(area_mapping)

    flow_df = flow_df.reset_index()
    price_df = price_df.reset_index()

    flow_df['cnecName'] = flow_df['cnecName'].apply(lambda x: '-'.join(sorted(x.split('-'))))
    flow_df = flow_df.reset_index().drop_duplicates(subset=['index', 'cnecName']).set_index('index')
    flow_df = flow_df.drop(columns=['level_0'])
    
    # Merge price_from
    flow_df = flow_df.merge(
        price_df.rename(columns={'deliveryArea': 'from_ba', 'price': 'price_from'}),
        how='left',
        on=['from_ba', 'index']
    )
    # Merge price_to
    flow_df = flow_df.merge(
        price_df.rename(columns={'deliveryArea': 'to_ba', 'price': 'price_to'}),
        how='left',
        on=['to_ba', 'index']
    )
    
    # Restore the datetime index
    flow_df.set_index('index', inplace=True)
    price_df.set_index('index', inplace=True)
    
    # Step 6: Compute F_k and F_l
    F_k = flow_df[['from_ba', 'to_ba', 'price_from', 'price_to', 'Flow', 'cnecName']]
    F_k['price_diff'] = F_k['price_from'] - F_k['price_to']

    F_l = net_position_df[['Bidding Zone', 'Net Position']]
    F_l['Bidding Zone'] = F_l['Bidding Zone'].replace(bidding_zone_mapping)
    F_l[['from_ba', 'to_ba']] = F_l['Bidding Zone'].str.split('-', expand=True)
    F_l['Flow'] = F_l['Net Position']
    # Combine into a result DataFrame
    F_l_reset = F_l.reset_index()
    price_df_reset = price_df.reset_index()

    F_l_reset = F_l_reset.merge(
        price_df_reset.rename(columns={'deliveryArea': 'from_ba', 'price': 'price_from'}),
        on=['from_ba', 'index'],
        how='left'
    )
    F_l_reset = F_l_reset.merge(
        price_df_reset.rename(columns={'deliveryArea': 'to_ba', 'price': 'price_to'}),
        on=['to_ba', 'index'],
        how='left'
    )
    F_l_reset.set_index('index', inplace=True)
    F_l = F_l_reset
    F_l['price_diff'] = F_l['price_from'] - F_l['price_to']

    F_l = F_l.drop(columns=['Bidding Zone', 'Net Position'])
    F_l['border'] = F_l.apply(lambda row: '-'.join(sorted([row['from_ba'], row['to_ba']])), axis=1)
    F_l = F_l.reset_index().drop_duplicates(subset=['index', 'border']).set_index('index')
    F_l = F_l.drop(columns=['border'])
    F_l = F_l[~((F_l['from_ba'] == "SE4") & (F_l['to_ba'] == "SE3"))]

    combinations_to_exclude = [
    ['SE3', 'FI'], ['FI', 'SE3'], 
    ['NO2', 'DK1'], ['DK1', 'NO2'], 
    ['DK1', 'DK2'], ['DK2', 'DK1'],
    ['SE3', 'DK1'], ['DK1', 'SE3'],
    ['SE4', 'SE3']
    ]
    combinations_df = pd.DataFrame(combinations_to_exclude, columns=['from_ba', 'to_ba'])
    F_k = F_k.reset_index().merge(
    combinations_df,
    on=['from_ba', 'to_ba'],
    how='left',
    indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])
    F_k.set_index('index', inplace=True)
    return F_k, F_l


def compute_summed_flow_for_sharing_key(F_k, F_l):
    exception_pairs = {"SE3-SE4", "SE4-SE3"}
    F_k['pair'] = F_k['from_ba'] + "-" + F_k['to_ba']
    F_l['pair'] = F_l['from_ba'] + "-" + F_l['to_ba']

    # Filter out rows from F_k whose pairs exist in F_l, except for the exception cases
    exclude_pairs = set(F_l['pair']) - exception_pairs
    F_k_filtered = F_k[~F_k['pair'].isin(exclude_pairs)]

    # Drop the 'pair' helper column
    F_k_filtered = F_k_filtered.drop(columns=['pair'])
    F_l = F_l.drop(columns=['pair'])

    # Combine the two dataframes
    F_combined = pd.concat([F_k_filtered, F_l])
    #F_combined = F_combined.drop(columns=['Bidding Zone', 'Net Position'])

    # Group by index, 'from_ba', and 'to_ba' to aggregate values
    F_i = (
        F_combined.groupby([F_combined.index, 'from_ba', 'to_ba'])
        .agg({
            'price_from': 'sum',
            'price_to': 'sum',
            'price_diff': 'sum',
            'Flow': 'sum'
        })
    )
    F_i = F_i.reset_index().set_index('index')
    return F_i, F_k


def compute_sharing_key(F_i, F_k):
    # Calculate the denominator
    denominator = F_i.groupby(F_i.index).apply(lambda df: (df['price_diff'] * df['Flow']).abs().sum())
    
    # Calculate the numerator for each border and hour
    numerator = abs(F_i.groupby([F_i.index, 'from_ba', 'to_ba']).apply(
        lambda df: (df['price_diff'] * df['Flow']).sum()
    ))
    
    # Convert numerator to a DataFrame
    numerator_df = numerator.reset_index(name='numerator')
    
    # Convert denominator to a DataFrame
    denominator_df = denominator.reset_index(name='denominator')
    denominator_df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Merge numerator and denominator on datetime
    result = pd.merge(
        numerator_df, 
        denominator_df, 
        left_on='index',  # numerator_df's datetime index from reset_index()
        right_on='datetime', 
        how='left'
    )
    
    # Calculate the fraction
    result['fraction'] = result['numerator'] / result['denominator']
    
    # Set datetime as index
    result.set_index('datetime', inplace=True)
    return result


def calculate_ci_per_border(CI_df, sharing_key):
    """
    Distribute "Total CI" across borders based on the "fraction" value in the sharing_key dataframe.

    Parameters:
    - CI_df (pd.DataFrame): DataFrame with datetime index and one column, "Total CI".
    - sharing_key (pd.DataFrame): DataFrame with datetime index and columns "from_ba", "to_ba", and "fraction".

    Returns:
    - pd.DataFrame: New DataFrame with datetime index, "from_ba", "to_ba", and the calculated values.
    """
    # Ensure both dataframes are aligned on the datetime index
    sharing_key = sharing_key.loc[CI_df.index]  # Align sharing_key to the CI_df index
    
    # Merge dataframes
    merged_df = sharing_key.merge(
        CI_df, left_index=True, right_index=True
    )
    
    # Calculate the CI value for each border
    merged_df['calculated_value'] = merged_df['fraction'] * merged_df['Total CI']
    
    # Select required columns
    result_df = merged_df[['from_ba', 'to_ba', 'calculated_value']]
    
    # Ensure the result has a datetime index
    result_df.index = merged_df.index
    
    return result_df

def calculate_new_ci_nordic(net_position_df, f_k, f_l, virtual_ci_component, price_df):
    common_index = (
        net_position_df.index.intersection(f_k.index)
        .intersection(f_l.index)
        .intersection(virtual_ci_component.index)
        .intersection(price_df.index)
    )

    # Reindex the dataframes to only include the common index
    net_position_df = net_position_df.loc[common_index]
    f_k = f_k.loc[common_index]
    f_l = f_l.loc[common_index]
    virtual_ci_component = virtual_ci_component.loc[common_index]
    price_df = price_df.loc[common_index]

    # Combine f_k and f_l into one DataFrame
    combined_f = pd.concat([f_k, f_l], axis=0)

    # Calculate the absolute value of price_diff multiplied by Flow for each hour
    combined_f['price_diff_x_Flow'] = abs(combined_f['price_diff'] * combined_f['Flow'])
    result_part1_per_hour = combined_f.groupby(combined_f.index)['price_diff_x_Flow'].sum()

    # Process net_position_df for the second part of the calculation
    columns_to_remove = [
        col for col in net_position_df.columns if any(substr in col for substr in ['KS', 'SK', 'SB', 'FS', 'SWL'])
    ]
    net_position_df = net_position_df.drop(columns=columns_to_remove, errors='ignore')
    net_position_df = net_position_df[net_position_df['Bidding Zone'].str.contains('_')]

    # Initialize second part result per hour
    result_part2_per_hour = pd.Series(0, index=common_index)

    columns_to_remove = [col for col in virtual_ci_component.columns if any(s in col for s in ['KS', 'SK', 'SB', 'FS', 'SWL'])]
    virtual_ci_component = virtual_ci_component.drop(columns=columns_to_remove, errors='ignore')
    # Calculate the second part of the equation for each hour
    for column in virtual_ci_component.columns:
        delivery_area = column.split('_')[0]  # Extract delivery area from column name
        corresponding_price = price_df.loc[price_df['deliveryArea'] == delivery_area, 'price']

        pd_v = virtual_ci_component[column] - corresponding_price.values
        np_v = net_position_df.loc[net_position_df['Bidding Zone'] == column, 'Net Position']

        # Calculate max(PD_v * NP_v, 0) for each hour
        max_pd_np = pd.Series([max(pd * -np, 0) for pd, np in zip(pd_v, np_v)], index=common_index)
        result_part2_per_hour += max_pd_np

    # Combine results
    total_result_per_hour = result_part1_per_hour + result_part2_per_hour

    return total_result_per_hour, result_part1_per_hour, result_part2_per_hour


def calculate_SK_i(ci_total, f_k, f_l):
    combined_f = pd.concat([f_k, f_l], axis=0)
    common_index = combined_f.index.intersection(ci_total.index)
    combined_f = combined_f.loc[common_index]
    ci_total = ci_total.loc[common_index]
    # Calculate the absolute value of price_diff multiplied by Flow
    combined_f['price_diff_x_Flow'] = abs(combined_f['price_diff'] * combined_f['Flow'])
    combined_f['ci_total'] = combined_f.index.map(ci_total.squeeze())
    combined_f['sharing_key_i'] = combined_f['price_diff_x_Flow'] / combined_f['ci_total']
    return combined_f

"""
def calculate_SK_v(virtual_ci_component, price_df, net_position_df, ci_total):
    net_position_df = net_position_df[net_position_df['Bidding Zone'].str.contains('_')]

    # Step 2: Add 'home_price' column
    # Extract the part before the underscore for each 'Bidding Zone'
    net_position_df['home_zone'] = net_position_df['Bidding Zone'].str.split('_').str[0]

    # Create a mapping of prices for delivery areas
    price_mapping = price_df.set_index('deliveryArea')['price'].to_dict()

    # Map the prices for each zone to 'home_price'
    net_position_df['home_price'] = net_position_df['home_zone'].map(price_mapping)

    # Step 3: Add 'virtual_price' column
    # Map values from virtual_ci_component based on 'Bidding Zone'
    def get_virtual_price(row):
        bidding_zone_column = row['Bidding Zone']
        try:
            return virtual_ci_component.at[row.name, bidding_zone_column] if bidding_zone_column in virtual_ci_component.columns else None
        except KeyError:
            return None

    net_position_df['virtual_price'] = net_position_df.apply(get_virtual_price, axis=1)

    # Step 4: Calculate 'SK_v' column
    # SK_v = Max{ (PD_v * -NP_v) / CI_total , 0 }
    def calculate_SK_v(row):
        try:
            PD_v = row['virtual_price'] - row['home_price']  # Price Difference
            NP_v = -row['Net Position']
            CI_v = ci_total.at[row.name, 'CI_total'] if isinstance(ci_total, pd.DataFrame) and 'CI_total' in ci_total.columns else ci_total.loc[row.name]
            if CI_v and CI_v != 0:
                return ((PD_v * NP_v) / CI_v, 0).clip(lower=0)
            else:
                return 0
        except (KeyError, TypeError, IndexError):
            return 0

    net_position_df['SK_v'] = net_position_df.apply(calculate_SK_v, axis=1)

    # Drop temporary column 'home_zone' used for mapping
    net_position_df.drop(columns=['home_zone'], inplace=True)

    return net_position_df
"""

def calculate_SK_v(virtual_ci_component, price_df, net_position_df, ci_total):
    common_index = (
        net_position_df.index
        .intersection(virtual_ci_component.index)
        .intersection(price_df.index)
    )
    net_position_df = net_position_df.loc[common_index]
    virtual_ci_component = virtual_ci_component.loc[common_index]
    price_df = price_df.loc[common_index]
    net_position_df = net_position_df.reset_index()
    net_position_df = net_position_df[~net_position_df.duplicated(subset=['index', 'Bidding Zone'], keep="first")]
    net_position_df = net_position_df.set_index('index')

    result_df = pd.DataFrame(columns=["pd_v", "np_v", "sharing_key_v", "ci_total"], index=pd.MultiIndex.from_product(
        [common_index, virtual_ci_component.columns], names=["timestamp", "delivery_area"]
    ))

    for column in virtual_ci_component.columns:
        delivery_area = column.split('_')[0]  # Extract delivery area from column name
        corresponding_price = price_df.loc[price_df['deliveryArea'] == delivery_area, 'price']
        corresponding_price = corresponding_price.reindex(common_index, fill_value=0)
        pd_v = virtual_ci_component[column] - corresponding_price
        np_v = net_position_df.loc[net_position_df['Bidding Zone'] == column, 'Net Position']
        np_v = np_v.reindex(common_index, fill_value=0)
        
        max_pd_np = ((pd_v * -np_v) / ci_total).clip(lower=0)

        # Add to result DataFrame
        result_df.loc[(slice(None), column), "pd_v"] = pd_v.values
        result_df.loc[(slice(None), column), "np_v"] = np_v.values
        result_df.loc[(slice(None), column), "sharing_key_v"] = max_pd_np.values
        result_df.loc[(slice(None), column), "ci_total"] = ci_total.reindex(common_index).values

    result_df = result_df.reset_index().set_index('timestamp')

    values_to_remove = ['DK1_KS', 'DK1_SK', 'DK1_SB', 'DK2_SB', 'SE3_SWL', 'SE4_SWL', 'SE3_KS', 'SE3_FS', 'NO2_SK', 'FI_FS']
    result_df = result_df[~result_df['delivery_area'].isin(values_to_remove)]

    return result_df


def calculate_CI_i(SK_i, CI_nordic):
    SK_i.loc[SK_i['cnecName'].str.contains('SWL', na=False), 'to_ba'] = 'SWL'
    SK_i.drop(columns=['cnecName'], inplace=True)
    SK_i['CI_i'] = SK_i['sharing_key_i'] * CI_nordic
    SK_i.rename(columns={'ci_total': 'ci_abs'}, inplace=True)
    SK_i['CI_nordic'] = SK_i.index.map(CI_nordic)
    return SK_i


def calculate_CI_v(SK_v, CI_nordic):
    SK_v['CI_v'] = SK_v['sharing_key_v'] * CI_nordic
    SK_v["from_ba"] = SK_v["delivery_area"].str.split('_').str[0]
    SK_v.rename(columns={"delivery_area": "to_ba"}, inplace=True)
    columns = list(SK_v.columns)
    columns.insert(columns.index("from_ba"), columns.pop(columns.index("to_ba")))
    SK_v = SK_v[columns]
    SK_v['CI_nordic'] = SK_v.index.map(CI_nordic)
    SK_v.rename(columns={'ci_total': 'ci_abs'}, inplace=True)
    return SK_v


def add_old_ci_to_new_ci(CI_i, CI_per_border_old):
    CI_i['old_CI'] = 0

    # Iterate through CI_per_border_old and update CI_i
    for idx, row in CI_per_border_old.iterrows():
        from_ba = row['from_ba']
        to_ba = row['to_ba']
        calculated_value = row['calculated_value']
        
        # Check if the row exists in CI_i
        if not CI_i[(CI_i['from_ba'] == from_ba) & (CI_i['to_ba'] == to_ba) & (CI_i.index == idx)].empty:
            CI_i.loc[(CI_i['from_ba'] == from_ba) & (CI_i['to_ba'] == to_ba) & (CI_i.index == idx), 'old_CI'] += calculated_value

    return CI_i


if __name__ == "__main__":
    bidding_zone_dict = {"FI":"10YFI-1--------U",
                        "NO1":"10YNO-1--------2",
                        "NO2":"10YNO-2--------T",
                        "NO3":"10YNO-3--------J",
                        "NO4":"10YNO-4--------9",
                        "NO5":"10Y1001A1001A48H",
                        "DK1":"10YDK-1--------W",
                        "DK2":"10YDK-2--------M",
                        "SE1":"10Y1001A1001A44P",
                        "SE2":"10Y1001A1001A45N",
                        "SE3":"10Y1001A1001A46L",
                        "SE4":"10Y1001A1001A47J",
                        "DK1_DE":"10Y1001A1001A82H",
                        "DK1A":"10YDK-1-------AA",
                        "DK2_KO": "10Y1001A1001A82H",
                        "DK1_CO":"10YNL----------L",
                        "FI_EL":"10Y1001A1001A39I",
                        "NO2_ND":"10YNL----------L",
                        "NO2_NK":"10Y1001A1001A82H",
                        "SE4_NB":"10YLT-1001A0008Q",
                        "SE4_SP":"10YPL-AREA-----S",
                        "SE4_BC": "10YDOM-PL-SE-LT2"
                        }


    border_dir = ["DK1-DK1_CO", "DK1_CO-DK1", "DK1-DK1_DE", "DK1_DE-DK1",
                  "DK1-NO2", "NO2-DK1", "DK1-SE3", "SE3-DK1", "DK2-DK1", "DK1-DK2", 
                  "DK2-DK2_KO", "DK2_KO-DK2", "DK2-SE4", "SE4-DK2", "FI_EL-FI", "FI-FI_EL", 
                  "FI-SE1", "SE1-FI", "FI-SE3", "SE3-FI", "NO1-NO3", "NO3-NO1", "NO1-SE3", 
                  "SE3-NO1", "NO1-NO2", "NO2-NO1", "NO1-NO5", "NO5-NO1","NO2_NK-NO2", 
                  "NO2-NO2_NK", "NO2_ND-NO2", "NO2-NO2_ND", "NO2-NO5", "NO5-NO2",
                  "NO3-NO4", "NO4-NO3", "NO3-NO5", "NO5-NO3", "NO3-SE2", "SE2-NO3", "NO4-SE1", 
                  "SE1-NO4", "NO4-SE2", "SE2-NO4", "SE1-SE2", "SE2-SE1", "SE2-SE3", "SE3-SE2", 
                  "SE3-SE4", "SE4-SE3", "SE4-SE4_NB", "SE4_NB-SE4", "SE4-SE4_SP", "SE4_SP-SE4"]
    

    internal_bidding_zone_borders = ['NO1-NO1', 'NO2-NO2', 'NO3-NO3', 'NO4-NO4', 
                                     'NO5-NO5', 'SE1-SE1', 'SE2-SE2', 'SE3-SE3', 
                                     'SE4-SE4', 'DK1-DK1', 'DK2-DK2', 'FI-FI']

    bzns = ["DK1", "DK2", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5", "FI"]

    nord_pool_areas = ['NO1','NO2','NO3','NO4','NO5','SE1','SE2','SE3','SE4','FI','DK1','DK2','GER','NL','LT','LV','EE']
    nordic_bidding_zones = ['NO1','NO2','NO3','NO4','NO5','SE1','SE2','SE3','SE4','FI','DK1','DK2']
    norwegian_bidding_zones = ['NO1','NO2','NO3','NO4','NO5']

    regional_np_groups = {
    'NO2': ['NO2', 'NO2_ND', 'NO2_NK'],
    'DK1': ['DK1', 'DK1_CO', 'DK1_DE'],
    'DK2': ['DK2', 'DK2_KO'],
    'SE4': ['SE4', 'SE4_SP', 'SE4_NB', 'SE4_BC'],
    'FI': ['FI', 'FI_EL']
    }

    virtual_bidding_zones_non_nordic = ["NO2_ND", "NO2_NK", "SE4_SP", "SE4_NB", "SE4_BC", "DK1_CO", 
                                        "DK1_DE", "DK2_KO", "FI_EL"]
    bidding_area_names = {"FI-FI_FS": "FI-SE3", "FI_FS-FI": "SE3-FI", "SE3-SE3_FS": "SE3-FI", 
                          "SE3_FS-SE3": "FI-SE3", "NO2-NO2_SK": "NO2-DK1", "NO2_SK-NO2": "DK1-NO2", 
                          "DK1-DK1_SK": "DK1-NO2", "DK1_SK-DK1": "NO2-DK1", "DK2-DK2_SB": "DK2-DK1", 
                          "DK2_SB-DK2": "DK1-DK2", "DK1-DK1_SB": "DK1-DK2", "DK1_SB-DK1": "DK2-DK1", 
                          "DK1-DK1_KS": "DK1-SE3", "DK1_KS-DK1": "SE3-DK1", "SE3-SE3_KS": "SE3-DK1", 
                          "SE3_KS-SE3": "DK1-SE3"}
    
    internal_hvdc = ['FI_FS', 'SE3_FS', 'NO2_SK', 'DK1_SK', 'DK1_SB', 'DK2_SB', 'SE3_KS', 'DK1_KS', 'SE3_SWL', 'SE4_SWL']

    internal_hvdc_to_border = {'FI_FS': 'SE3-FI', 'SE3_FS': 'FI-SE3', 'NO2_SK': 'DK1-NO2', 'DK1_SK': 'NO2-DK1', 
                               'DK1_SB': 'DK2-DK1', 'DK2_SB': 'DK1-DK2', 'SE3_KS': 'DK1-SE3', 'DK1_KS': 'SE3-DK1', 
                               'SE3_SWL': 'SE4-SE3', 'SE4_SWL': 'SE3-SE4'}

    entsoe_token = "788640ef-a0e0-4b55-b20f-aaf4b17ebe43"
    jao_token = "5080c637-d372-494e-87b4-8b706d132ecc"

    start_date = datetime(2024, 12, 31, 22, 0)
    end_date = datetime(2025, 1, 1, 22, 0)

    """
    Day ahead prices
    """
    price_DA = Nordpool_API_V2(areas=nordic_bidding_zones, nordpool_api_type='Auction', 
                               auction_type='Prices', date=start_date.strftime("%Y-%m-%d"), 
                               end_date=end_date.strftime("%Y-%m-%d"), timezone='UTC', 
                               currency='EUR', market='dayAhead').get_timeseries()
    price_df = clean_nordpool_prices(price_DA)
    jao_df = create_dataframe_from_jao_data(start_date, end_date, jao_token)
    ptdf_df = extract_jao_ptdfs(jao_df, border=False)
    border_ptdf_df = extract_jao_ptdfs(jao_df, border=True, remove_virtual=False)
    net_position_df = extract_jao_net_position(jao_df)
    #ac_net_position_df = net_position_df[~net_position_df['Bidding Zone'].str.contains('_')]
    #shadow_price_df = extract_jao_shadow_prices(jao_df)
    full_net_pos_df = net_position_df.pivot_table(index=net_position_df.index, columns="Bidding Zone", values="Net Position", aggfunc="sum")
    regional_np_df = calculate_regional_net_positions(net_position_df, regional_np_groups, remove_virtual=True)
    virtual_ci_component = calculate_CI_virtual_component(price_df, ptdf_df)
    CI_virtual_df = calcluate_CI_nordic_virtual(price_df, virtual_ci_component, net_position_df)
    CI_regional_df = calculate_CI_nordic_regional(regional_np_df, price_df)
    CI_df = calculate_CI_nordic(CI_regional_df, CI_virtual_df)
    #ac_flow_df = calculate_regional_flow(regional_np_df, border_ptdf_df)
    virtual_ptdf_df = extract_jao_ptdfs(jao_df, border=True, only_virtual=True, include_flow=True)
    #virtual_flow_df = virtual_ptdf_df[['cnecName', 'flowFb']]
    full_flow_df = calculate_regional_flow(full_net_pos_df, border_ptdf_df)
    F_k, F_l = compute_flow_for_sharing_key(price_df, 
                                            full_flow_df, 
                                            net_position_df, 
                                            virtual_bidding_zones_non_nordic, 
                                            bidding_area_names, 
                                            internal_hvdc,
                                            internal_hvdc_to_border)
    ci_total, ci_internal_ac, ci_internal_dc = calculate_new_ci_nordic(net_position_df, F_k, F_l, virtual_ci_component, price_df)
    SK_i = calculate_SK_i(ci_total, F_k, F_l)
    SK_v = calculate_SK_v(virtual_ci_component, price_df, net_position_df, ci_total)
    CI_i = calculate_CI_i(SK_i, CI_df)
    CI_v = calculate_CI_v(SK_v, CI_df)
    f_k_old, f_l_old = compute_flow_for_sharing_key_old(price_df, 
                                            full_flow_df, 
                                            net_position_df, 
                                            virtual_bidding_zones_non_nordic, 
                                            bidding_area_names, 
                                            internal_hvdc,
                                            internal_hvdc_to_border)
    CI_old = calculate_CI_nordic_old(regional_np_df, price_df)
    F_i, F_k_old = compute_summed_flow_for_sharing_key(F_k, F_l)
    sharing_key_old = compute_sharing_key(F_i, F_k)
    CI_per_border_old = calculate_ci_per_border(CI_old, sharing_key_old)
    CI_i = add_old_ci_to_new_ci(CI_i, CI_v, CI_per_border_old)

    CI_i.to_csv('congestion_income_internal.csv')
    CI_v.to_csv('congestion_income_external.csv')