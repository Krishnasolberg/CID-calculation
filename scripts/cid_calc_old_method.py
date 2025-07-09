"""
Vi trenger
- Nettoposisjoner (JAO)
- Skyggepris (JAO)
- PTDFer (JAO)
- Budområdepriser (Nord Pool)
"""

import requests
import pandas as pd
import warnings
import time
import json
import pytz
import xlrd
import plotly.express as px

from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta
from alive_progress import alive_bar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
from Nordpool_V2 import Nordpool_API_V2
from pathlib import Path


def get_jao_data(start_date:datetime, end_date:datetime, token):
    url = f"https://publicationtool.jao.eu/nordic/api/data/fbDomainShadowPrice?Filter=%7B%7D&\
        Skip=0&Take=100000&FromUtc={start_date}T00%3A00%3A00.000Z&ToUtc={end_date}T23%3A00%3A00.000Z"
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
        CnecName%22%3A%22Border_CNEC%22%7D&Skip=0&Take=100000&FromUtc={start_date}T23\
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


def get_entsoe_prices_url(start_date: datetime, 
                end_date: datetime, 
                bzn_in: str,
                bzn_out: str,  
                api_token: str,
                bidding_zone_dict: dict):
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    bzn_in = bidding_zone_dict.get(bzn_in, None)
    bzn_out = bidding_zone_dict.get(bzn_out, None)
    url = f"https://web-api.tp.entsoe.eu/api?securityToken={api_token}&documentType=A44&"\
    f"in_Domain={bzn_in}&out_Domain={bzn_out}&"\
        f"periodStart={start_date_str}2300&periodEnd={end_date_str}2300"
    return url


def get_entsoe_net_position_url(start_date: datetime, 
                end_date: datetime, 
                bzn_in: str,
                bzn_out: str,  
                api_token: str,
                bidding_zone_dict: dict):
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    bzn_in = bidding_zone_dict.get(bzn_in, None)
    bzn_out = bidding_zone_dict.get(bzn_out, None)
    url = f"https://web-api.tp.entsoe.eu/api?securityToken={api_token}&documentType=A25&"\
    f"businessType=B09&contract_MarketAgreement.Type=A01&in_Domain={bzn_in}&out_Domain={bzn_out}&"\
        f"periodStart={start_date_str}2300&periodEnd={end_date_str}2300"
    return url


def get_entso_e_data(url):
    """Creates an xml-response from ENTSO-E TP based on a certain URL.

    Parameters 
    -----------
    df:
        A URL for the requested data from ENTSO-E TP.
    Returns
    -------
        A soup (xlm) file with the response from the ENTSO-E TP API based on the URL.
        
    """
    session = requests.Session()
    session.verify = False
    content = requests.get(url).text
    soup = BeautifulSoup(content, features="lxml")
    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
    return soup

def parse_xml_to_dataframe(xml_file, bidding_zone_in, bidding_zone_out):
    """Creates a dataframe from the xml-file from ENTSO-E TP.

    Parameters 
    -----------
    xml_file:
        Dataframe of total ATC capacity in and out of each bidding area
    bidding_zone_in:
        The bidding zone in
    bidding_zone_out:
        The bidding zone out
    Returns
    -------
        A dataframe with the values and dataframes that came from the XML-file from ENTSO-E TP
        
    """
    data_dict = {'start_time': [], 'end_time': [], 'quantity': []}
    cet_timezone = pytz.timezone('CET')
    for period in xml_file.find_all('period'):
        start_time_str = period.find('start').text
        end_time_str = period.find('end').text
        quantities = period.find_all('quantity')

        start_time_utc = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%MZ')
        end_time_utc = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%MZ')

        start_time_cet = start_time_utc.astimezone(cet_timezone)
        end_time_cet = end_time_utc.astimezone(cet_timezone)
        start_time_cet = start_time_utc + timedelta(hours=2)
        end_time_cet = end_time_utc + timedelta(hours=2)

        for quantity in quantities:
            data_dict['start_time'].append(start_time_cet)
            data_dict['end_time'].append(end_time_cet)
            data_dict['quantity'].append(float(quantity.text))
            start_time_cet += timedelta(hours=1)
            end_time_cet += timedelta(hours=1)

    df = pd.DataFrame(data_dict)
    df = df.drop(['end_time'], axis=1)
    df = df.rename(columns={"start_time": "Timestamp", 
                            "quantity": f"{bidding_zone_out}-{bidding_zone_in}"})
    df = df.set_index('Timestamp')
    return df


def combine_bidding_zones_for_day_ahead_prices(start_date: datetime, 
                                       end_date: datetime, 
                                       border_dir, 
                                       bidding_zone_dict, 
                                       api_token):
    """Creates a dataframe that shows all situations where the trading possibilities are less 
    than a certain threshold.

    Parameters 
    -----------
    df:
        Dataframe of total ATC capacity in and out of each bidding area
    threshold:
        Amount of MW to be set as the threshold
    Returns
    -------
        A dataframe with all situations where the trade possibilities are less that the 
        threshold
        
    """
    dfs = []
    with alive_bar(len(border_dir)) as bar:
        for border in border_dir:
            parts = border.split('-')
            time.sleep(0.005)
            if len(parts) == 2:
                border_1, border_2 = parts
                url = get_entsoe_prices_url(start_date, 
                                                end_date, 
                                                border_1, 
                                                border_2, 
                                                api_token, 
                                                bidding_zone_dict)
                data = get_entso_e_data(url)
                df = parse_xml_to_dataframe(data, border_1, border_2)
                dfs.append(df)
                print(f"Dataframe created for {border_1}-{border_2}")
            bar()
    result_df = pd.concat(dfs, axis=1)
    result_df.insert(0, 'Backup', False)
    return result_df


def combine_bidding_zones_for_net_positions(start_date: datetime, 
                                       end_date: datetime, 
                                       border_dir, 
                                       bidding_zone_dict, 
                                       api_token):
    """Creates a dataframe that shows all situations where the trading possibilities are less 
    than a certain threshold.

    Parameters 
    -----------
    df:
        Dataframe of total ATC capacity in and out of each bidding area
    threshold:
        Amount of MW to be set as the threshold
    Returns
    -------
        A dataframe with all situations where the trade possibilities are less that the 
        threshold
        
    """
    dfs = []
    with alive_bar(len(border_dir)) as bar:
        for border in border_dir:
            parts = border.split('-')
            time.sleep(0.005)
            if len(parts) == 2:
                border_1, border_2 = parts
                url = get_entsoe_net_position_url(start_date, 
                                                end_date, 
                                                border_1, 
                                                border_2, 
                                                api_token, 
                                                bidding_zone_dict)
                data = get_entso_e_data(url)
                df = parse_xml_to_dataframe(data, border_1, border_2)
                dfs.append(df)
                print(f"Dataframe created for {border_1}-{border_2}")
            bar()
    result_df = pd.concat(dfs, axis=1)
    result_df.insert(0, 'Backup', False)
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
        columns_to_select = ['dateTimeUtc', 'cnecName', 'fall', 'flowFb'] + df.columns[-31:].tolist()
    else:
        columns_to_select = ['dateTimeUtc', 'cnecName', 'fall'] + df.columns[-31:].tolist()
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


def calculate_scaling_factor(price_df, regional_flow_df, ptdf_df):
    hei= 1


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


def plot_bidding_area_prices(df):
    # Reset index to use deliveryStart as a column for plotting
    df = df.reset_index()
    
    # Create a line plot
    fig = px.line(df, x='deliveryStart', y='price', color='deliveryArea', 
                  title="Prices for Each Bidding Zone", 
                  labels={"deliveryStart": "Timestamp", "price": "Price"})
    
    # Show the plot
    fig.show()


def calculate_CI_nordic_old(regional_np_df, price_df):
    # Ensure both dataframes have matching datetime indices
    regional_np_df = regional_np_df.copy()
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    regional_np_df.index = pd.to_datetime(regional_np_df.index)
    
    # Align the indices of both dataframes to include only the common hours
    common_index = regional_np_df.index.intersection(price_df.index)
    regional_np_df = regional_np_df.loc[common_index]
    price_df = price_df.loc[common_index]
    
    # Initialize an empty DataFrame to store the results
    adjusted_df = pd.DataFrame(index=common_index)
    
    # Iterate over each column (deliveryArea) in price_df
    for area in price_df['deliveryArea'].unique():
        # Check if the area is also in the regional_np_df columns
        if area in regional_np_df.columns:
            # Take the negative of the price values for the common index
            negative_price = price_df.loc[price_df['deliveryArea'] == area, 'price']
            
            # Multiply by the net position value in the regional_np_df for the corresponding area
            adjusted_values = negative_price.values * regional_np_df[area].values
            
            # Add the results to the adjusted_df with the area as column name
            adjusted_df[area] = adjusted_values
    result_df = pd.DataFrame((-1)*adjusted_df.sum(axis=1), columns=['Total CI'])
            
    return result_df


def calculate_regional_flow(regional_np_df, border_ptdf_df):
    # Ensure both dataframes have matching datetime indices and order columns to match
    regional_np_df = regional_np_df.copy()
    border_ptdf_df = border_ptdf_df.copy()
    regional_np_df.index = pd.to_datetime(regional_np_df.index)
    border_ptdf_df.index = pd.to_datetime(border_ptdf_df.index)
    
    # Separate the cnecName column from border_ptdf_df, as it will not be involved in calculations
    cnec_name_column = border_ptdf_df['cnecName']
    f0_column = border_ptdf_df['fall']
    border_ptdf_df = border_ptdf_df.drop(columns=['cnecName', 'fall'])
    
    # Reorder columns in border_ptdf_df to match those in regional_np_df
    border_ptdf_df = border_ptdf_df[regional_np_df.columns]
    
    # Convert to numpy arrays for faster processing
    regional_np_values = regional_np_df.values
    border_ptdf_values = border_ptdf_df.values
    
    # Find the unique timestamps in regional_np_df to create a lookup dictionary
    timestamps = regional_np_df.index
    timestamp_to_idx = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    
    # Prepare an empty array to store results with the same shape as border_ptdf_values
    result_values = np.empty_like(border_ptdf_values, dtype=np.float64)
    
    # Iterate over each row in border_ptdf_df and apply multiplication based on the matching timestamp
    for i, timestamp in enumerate(border_ptdf_df.index):
        # Get the index of the corresponding hour in regional_np_df
        regional_idx = timestamp_to_idx.get(timestamp)
        
        # If there is a matching timestamp in regional_np_df, perform the multiplication
        if regional_idx is not None:
            result_values[i] = border_ptdf_values[i] * regional_np_values[regional_idx]
        else:
            # If no match, set the result to NaN or another default value
            result_values[i] = np.nan
    
    # Convert the result back to a DataFrame with the same index and columns as border_ptdf_df
    result_df = pd.DataFrame(result_values, index=border_ptdf_df.index, columns=border_ptdf_df.columns)
    
    # Add the cnecName column back as the first column
    result_df.insert(0, 'cnecName', cnec_name_column.values)
    result_df.insert(1, 'fall', f0_column.values)
    
    # Sum all columns except 'cnecName' for each row and store in a new column 'Flow'
    result_df['Flow'] = result_df.iloc[:, 1:].sum(axis=1)
    result_df['Flow'] = result_df["Flow"] - result_df['fall']
    
    return result_df



def calculate_nordic_congestion_income(CI_df, sharing_key_df):
    merged_df = sharing_key.merge(CI_df, left_index=True, right_index=True, how="left")

    # Create the new column by multiplying 'Total CI' with 'normalized_value'
    merged_df['ci_shared'] = merged_df['Total CI'] * merged_df['normalized_value']

    return merged_df


def compute_flow_for_sharing_key_old(price_df, flow_df, net_position_df, exclude_list, replace_dict, internal_hvdc, bidding_zone_mapping):
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
    F_k = flow_df[['from_ba', 'to_ba', 'price_from', 'price_to', 'Flow']]
    F_k['price_diff'] = F_k['price_to'] - F_k['price_from']

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
    F_l['price_diff'] = F_l['price_to'] - F_l['price_from']

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
    F_combined = F_combined.drop(columns=['Bidding Zone', 'Net Position'])

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
    return F_i, F_k_filtered


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
    merged_df['CI'] = merged_df['fraction'] * merged_df['Total CI']
    
    # Select required columns
    result_df = merged_df[['from_ba', 'to_ba', 'CI']]
    
    # Ensure the result has a datetime index
    result_df.index = merged_df.index
    
    return result_df


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

    entsoe_token = "366e5d26-85e0-479e-b2e3-e61d1444b7f6"
    jao_token = "5080c637-d372-494e-87b4-8b706d132ecc"

    start_date = datetime(2024, 10, 29, 22, 0)
    end_date = datetime(2024, 10, 31, 22, 0) 

    """
    Day ahead prices
    """
    #net_position_df = combine_bidding_zones_for_net_positions(start_date, end_date, internal_bidding_zone_borders, bidding_zone_dict, entsoe_token)
    price_DA = Nordpool_API_V2(areas=nordic_bidding_zones, nordpool_api_type='Auction', 
                               auction_type='Prices', date=start_date.strftime("%Y-%m-%d"), 
                               end_date=end_date.strftime("%Y-%m-%d"), timezone='UTC', 
                               currency='EUR', market='dayAhead').get_timeseries()
    price_df = clean_nordpool_prices(price_DA)
    jao_df = create_dataframe_from_jao_data(start_date, end_date, jao_token)
    ptdf_df = extract_jao_ptdfs(jao_df, border=False)
    border_ptdf_df = extract_jao_ptdfs(jao_df, border=True, remove_virtual=False)
    net_position_df = extract_jao_net_position(jao_df)
    ac_net_position_df = net_position_df[~net_position_df['Bidding Zone'].str.contains('_')]
    shadow_price_df = extract_jao_shadow_prices(jao_df)
    full_net_pos_df = net_position_df.pivot_table(index=net_position_df.index, columns="Bidding Zone", values="Net Position", aggfunc="sum")
    regional_np_df = calculate_regional_net_positions(net_position_df, regional_np_groups, remove_virtual=True)
    CI_df = calculate_CI_nordic_old(regional_np_df, price_df)
    ac_flow_df = calculate_regional_flow(regional_np_df, border_ptdf_df)
    virtual_ptdf_df = extract_jao_ptdfs(jao_df, border=True, only_virtual=True, include_flow=True)
    virtual_flow_df = virtual_ptdf_df[['cnecName', 'flowFb']]
    full_flow_df = calculate_regional_flow(full_net_pos_df, border_ptdf_df)
    F_k, F_l = compute_flow_for_sharing_key_old(price_df, 
                                            full_flow_df, 
                                            net_position_df, 
                                            virtual_bidding_zones_non_nordic, 
                                            bidding_area_names, 
                                            internal_hvdc,
                                            internal_hvdc_to_border)
    F_i, F_k = compute_summed_flow_for_sharing_key(F_k, F_l)
    sharing_key = compute_sharing_key(F_i, F_k)
    CI_per_border = calculate_ci_per_border(CI_df, sharing_key)
    #net_position_df = combine_bidding_zones_for_net_positions(start_date, end_date, internal_bidding_zone_borders, bidding_zone_dict, api_token)
    breakpoint