

import json
import requests
import pandas as pd
from pathlib import Path

class Nordpool_API_V2:
    def __init__(self, date='2024-02-20', areas=['NO1', 'NO2', 'NO5'], config_name='Nordpool_config.json', currency=None, market=None, end_date='2024-02-22', nordpool_api_type="auction", auction_type='Volumes', timezone='UTC', block_order=False):
        self.areas = areas
        self.date = date
        self.currency = currency
        self.end_date = end_date
        self.auction_type = auction_type
        self.nordpool_api_type = nordpool_api_type
        self.market = market
        self.timezone = timezone
        self.block_order = block_order
        
        with open(Path(__file__).parent/config_name, 'r') as file:
            self.config = json.load(file)
    
    def generate_response(self):
        token_url = "https://sts.nordpoolgroup.com/connect/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic Y2xpZW50X21hcmtldGRhdGFfYXBpOmNsaWVudF9tYXJrZXRkYXRhX2FwaQ=="
        }
        payload = f"grant_type=password&scope=marketdata_api&username={self.config['user']}&password={self.config['password']}"
        response = requests.post(token_url, data=payload, headers=headers, verify=True)
        
        return response
    
    def generate_token(self):
        response = self.generate_response()
        if response.status_code == 200:
            d = json.loads(response.text)
            token = d["access_token"]
            headers = {
                'Accept-Encoding': '',
                'Authorization': f'Bearer {token}',
            }
        else:
            datas = json.loads(response.text)
            print("Response code:", response.status_code, ".This results in the following error: ", datas['errorMessages'])
            
        return headers
    
    def generate_url(self, date):
        base_url = f"https://data-api.nordpoolgroup.com/api/v2/{self.nordpool_api_type}/{self.auction_type}/ByAreas?"
        if self.nordpool_api_type=='System':
            base_url = f"https://data-api.nordpoolgroup.com/api/v2/{self.nordpool_api_type}/{self.auction_type}?"

        # Constructing the parameters dictionary
        params = {}
        if self.market is not None:
            params['market'] = self.market
        
        if self.nordpool_api_type != 'System':
            areas_param = "%2C".join(self.areas)
            params['areas'] = areas_param

        optional_params = {"currency": self.currency}
        
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value
        
        params["date"]=self.date
    
        # Constructing the URL string by joining parameters
        url = base_url + "&".join(f"{key}={value}" for key, value in params.items())
        
        if self.nordpool_api_type not in ['Auction', 'PowerSystem', 'Intraday', 'BalanceMarket', 'ExchangeRate', 'System']:
            raise ValueError(f'argument "{self.nordpool_api_type}" is not a valid argument. Please check this')

        return url

    
    def flatten_dict_to_df(self, dictionary):
        """
        

        Parameters
        ----------
        dictionary : Dict. Dictionary with contains information about the physical flow, like unit, time, totals etc. This might have a complex 

        Returns
        -------
        flat_dict : Flattened dictionary

        """
        flat_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                flat_dict.update({f"{key}_{k}": v for k, v in value.items()})
                df = pd.DataFrame.from_dict(flat_dict, orient='index').T
            else:
                flat_dict[key] = value
                df = pd.DataFrame.from_dict(flat_dict, orient='index').T
        return df
    
    def get_data(self):
        """
        

        Returns
        -------
        result_df : Datafrane where all values from the query is retrived

        """
        url = self.generate_url(self.date)
        headers = self.generate_token()
        response = requests.get(url, headers=headers, verify=True)
        if response.status_code == 200:
            data = response.json()
            result_df = pd.DataFrame()
            if self.auction_type=='Capacities' or self.auction_type=='Exchanges' or self.auction_type=='Flows':
                for item in data:
                    item_lists = {key: value for key, value in item.items() if isinstance(value, list)}  # Extract lists from the item dictionary
                    info_items = {key: value for key, value in item.items() if not isinstance(value, list)} # Extract key-value pairs of non-list type, e.g. floats and strings
                    info_df = self.flatten_dict_to_df(info_items)  # Retriving info 
                    for key, value in item_lists.items():
                        for subdict in value:
                            #result_df = pd.DataFrame()
                            list_subdicts = {}
                            for key, subvalue in subdict.items():
                                if isinstance(subvalue, list):
                                    list_subdicts[key] = subvalue
                                    result_3 = pd.DataFrame()
                                    info_subdicts = {key: subvalue for key, subvalue in subdict.items() if not isinstance(subvalue, list)}
                                    subinfo_df = self.flatten_dict_to_df(info_subdicts)
                                    df_info_merge = pd.concat([info_df, subinfo_df], axis=1)
                                    for key, subvalue_2 in list_subdicts.items():
                                        if isinstance(subvalue_2, list):
                                            df = pd.DataFrame(subvalue_2)
                                            df['Key'] = key
                                            info_3 = pd.concat([df_info_merge] * len(df), ignore_index=True)
                                            df = pd.concat([df, info_3], axis=1)
                                            result_3 = pd.concat([result_3, df], ignore_index=True)
                                            result_df = pd.concat([result_df, result_3], ignore_index=True)
                                            
            
            elif self.auction_type=='Productions':
                result_df = pd.DataFrame()
                for item in data:
                    item_lists = {key: value for key, value in item.items() if isinstance(value, list)}  # Extract lists from the item dictionary
                    info_items = {key: value for key, value in item.items() if not isinstance(value, list)} # Extract key-value pairs of non-list type, e.g. floats and strings
                    info_df = self.flatten_dict_to_df(info_items)  # Retriving info 
                    for key, value in item_lists.items():
                        for subdict in value:
                            #result_df = pd.DataFrame()
                            list_subdicts = {}
                            for key, subvalue in subdict.items():
                                if isinstance(subvalue, dict):
                                    timeseries = self.flatten_dict_to_df(subvalue)
                                    info_subdicts = {key: subvalue for key, subvalue in subdict.items() if not isinstance(subvalue, dict)}
                                    df_subinfo = self.flatten_dict_to_df(info_subdicts)
                                    df = pd.concat([df_subinfo, info_df, timeseries], axis=1)
                                    result_df = pd.concat([result_df, df], ignore_index=True)
                        
            
            

                

                                            
            elif self.auction_type=='Prices' or self.auction_type=='Volumes' or self.auction_type=='Price':
                df_list = []
                result_df = pd.DataFrame()
                for item in data:
                    item_lists = {key: value for key, value in item.items() if isinstance(value, list)}  # Extract lists from the item dictionary
                    info_items = {key: value for key, value in item.items() if not isinstance(value, list)}  # Extract key-value pairs of non-list type, e.g., floats and strings
                    info_df = self.flatten_dict_to_df(info_items)  # Retrieving info
                    for key, value in item_lists.items():
                        # Create DataFrame for each list
                        list_df = pd.DataFrame(value)
                        # Append it to the list of DataFrames
                        info = pd.concat([info_df] * len(list_df), ignore_index=True)
                        df = pd.concat([list_df, info], axis=1)
                        df_list.append(df)
                
                unique_sizes = set(len(df) for df in df_list)
                result_dfs = []
                
                # Process DataFrames based on unique sizes
                for size in unique_sizes:
                    size_dfs = [df for df in df_list if len(df) == size]
                    
                    if len(size_dfs) > 1:  # If there are multiple DataFrames of the same size
                        # Concatenate DataFrames with identical sizes
                        concatenated_df = pd.concat(size_dfs, ignore_index=True)
                        result_dfs.append(concatenated_df)
                    else:  # Only one DataFrame with this size
                        result_dfs.extend(size_dfs)
                    
                # Now process result_dfs to find the desired DataFrame
                result_df = None  # Initialize result_df as None
                
                for df in result_dfs:
                    if not self.block_order:
                        # Skip DataFrames that contain 'blockName' when block_order is False
                        if 'blockName' in df.columns:
                            continue
                        else:
                            result_df = df
                            break  # Stop as soon as the first matching DataFrame is found
                    else:
                        # When block_order is True, select the first DataFrame
                        result_df = df
                        break  # Stop as soon as the first matching DataFrame is found
                
                # result_df now holds the correct DataFrame based on your conditions

        
                        
            elif self.auction_type=='ContractStatistics' or self.auction_type=='HourlyStatistics' or self.auction_type=='ManualFrequencyRestorationReserves' or self.auction_type=='Consumptions':
                result_df = pd.DataFrame()
                info_items = {}
                for item in data:
                    info_items = {key: value for key, value in item.items() if not isinstance(value, list)}
                    for key, value in item.items():
                        if isinstance(value, list):
                                if len(value) >=24:
                                    timeseries = pd.DataFrame(value)
                                    flat_data_dict = self.flatten_dict_to_df(info_items)
                                    info = pd.concat([flat_data_dict] * len(timeseries), ignore_index=True)
                                    df = pd.concat([timeseries, info], axis=1)
                                    result_df = pd.concat([result_df, df], ignore_index=True)
                                    
        
        return result_df
    
            

        
    
    def get_timeseries(self):
        result_df = pd.DataFrame()  # Initialize an empty dataframe to store results
        if self.end_date is not None:
            date_range = pd.date_range(self.date, self.end_date)
            for date in date_range:
                self.date = date.strftime("%Y-%m-%d")
                print("Fetching data for:", self.nordpool_api_type, "-", self.auction_type, '-', self.date, "-", self.areas)
                df = self.get_data()  # Fetch data for the current date
                if df is not None and not df.empty:
                    result_df = pd.concat([result_df, df], ignore_index=True)  # Concatenate data to the result dataframe
                else:
                    print(f"{self.date} is the last day with data in the API.")
                    break
        else:
            print("Fetching data for:", self.nordpool_api_type, "-", self.auction_type, '-', self.date, "-", self.areas)
            df = self.get_data()
            if df is not None and not df.empty:
                result_df = pd.concat([result_df, df], ignore_index=True)
            else:
                print(f"{self.date} has no data in the API.")
    
        if 'deliveryStart' in result_df.columns and 'deliveryEnd' in result_df.columns:
            result_df['deliveryStart'] = pd.to_datetime(result_df['deliveryStart'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize('UTC')  # API is in UTC by default, need to specify for conversion
            result_df['deliveryStart'] = result_df['deliveryStart'].dt.tz_convert(self.timezone)  # Converting to CET
            result_df['deliveryStart'] = result_df['deliveryStart'].dt.tz_localize(None)
            
            result_df['deliveryEnd'] = pd.to_datetime(result_df['deliveryEnd'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize('UTC')  # API is in UTC by default, need to specify for conversion
            result_df['deliveryEnd'] = result_df['deliveryEnd'].dt.tz_convert(self.timezone)  # Converting to CET
            result_df['deliveryEnd'] = result_df['deliveryEnd'].dt.tz_localize(None)
        elif 'deliveryStart' in result_df.columns and 'deliveryEnd' not in result_df.columns:
            result_df['deliveryStart'] = pd.to_datetime(result_df['deliveryStart'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize('UTC')  # API is in UTC by default, need to specify for conversion
            result_df['deliveryStart'] = result_df['deliveryStart'].dt.tz_convert(self.timezone)  # Converting to CET
            result_df['deliveryStart'] = result_df['deliveryStart'].dt.tz_localize(None) 
                                
        return result_df

#start_date = '2024-06-30'
#end_date = '2024-07-01'
#areas = ['NO2']
#IDA3_volume = Nordpool_API_V2(areas=areas, nordpool_api_type='Auction', auction_type='Volumes', date=start_date, end_date=end_date, timezone='CET', market='SIDC_IntradayAuction3', block_order=False).get_data()

#IDA1_price = Nordpool_API_V2(areas=areas, nordpool_api_type='Auction', auction_type='Prices', date=start_date, end_date=end_date, timezone='CET', currency='EUR', market='SIDC_IntradayAuction3').get_timeseries()
#df = Nordpool_API_V2(nordpool_api_type='System', auction_type='Price', date='2024-06-24', end_date='2024-06-25', timezone='CET', currency='EUR').get_timeseries()
