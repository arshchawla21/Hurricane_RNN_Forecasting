import json
import os

data_parent_path: str = '/g/data/rt52/era5/pressure-levels/reanalysis'
data_info_path: str = '../data/era5-reanalysis-data-info.json'
data_var_available_list: list[str] = ['crwc', 't', 'clwc', 'q', 'w', 'cc', 'ciwc', 'cswc', 'vo', 'o3', 'v', 'z', 'pv', 'd', 'u', 'r']
data_var_picked_list: list[str] = ['pv', 'crwc', 'cswc', 't', 'u', 'v', 'z', 'w', 'vo', 'd', 'r', 'clwc', 'cc']

start_time: str = "1985-05"
end_time: str = "1986-07"

# This will list all the variables in the parent path
print(os.listdir(data_parent_path))

# Read the information about the data. This information is downloaded from the ERA5 home.
# If there is an description field, then that is my explaination of the parameter
with open(data_info_path, 'r') as file:
    data_info = json.load(file)
#    print(data_info[0])

# Now I should be parsing the time option. currently, I am using the format as YYYY-MM
class RequestedTime:
    def __init__(self, year, month):
        self.year = year
        self.month = month

    def __str__(self):
        return f"{{year: {self.year}, month: {self.month}}}"

    def get_month(self) -> str:
        year = self.year
        month = self.month

        if month == 1:
            return f"{year}0101-{year}0131"
        elif month == 2:
            return f"{year}0201-{year}0228"
        elif month == 3:
            return f"{year}0301-{year}0331"
        elif month == 4:
            return f"{year}0401-{year}0430"
        elif month == 5:
            return f"{year}0501-{year}0531"
        elif month == 6:
            return f"{year}0601-{year}0630"
        elif month == 7:
            return f"{year}0701-{year}0731"
        elif month == 8:
            return f"{year}0801-{year}0831"
        elif month == 9:
            return f"{year}0901-{year}0930"
        elif month == 10:
            return f"{year}1001-{year}1031"
        elif month == 11:
            return f"{year}1101-{year}1130"
        elif month == 12:
            return f"{year}1201-{year}1231"
        else:
            raise ValueError("month should be in between 1 and 12")

time_args = start_time.split('-')
start_time_parsed = RequestedTime(int(time_args[0]), int(time_args[1]))
time_args = end_time.split('-')
end_time_parsed = RequestedTime(int(time_args[0]), int(time_args[1]))

# Now I have to actually find the corresponding list of time

def get_month(time: RequestedTime) -> str:
    year = time.year
    month = time.month

    if month == 1:
        return f"{year}0101-{year}0131"
    elif month == 2:
        return f"{year}0201-{year}0228"
    elif month == 3:
        return f"{year}0301-{year}0331"
    elif month == 4:
        return f"{year}0401-{year}0430"
    elif month == 5:
        return f"{year}0501-{year}0531"
    elif month == 6:
        return f"{year}0601-{year}0630"
    elif month == 7:
        return f"{year}0701-{year}0731"
    elif month == 8:
        return f"{year}0801-{year}0831"
    elif month == 9:
        return f"{year}0901-{year}0930"
    elif month == 10:
        return f"{year}1001-{year}1031"
    elif month == 11:
        return f"{year}1101-{year}1130"
    elif month == 12:
        return f"{year}1201-{year}1231"
    else:
        raise ValueError("month should be in between 1 and 12")

def get_data_path_with_param(param_short_name: str, year: str, month: int, parent_path: str) -> str:
    return f"{parent_path}/{param_short_name}/{year}/{param_short_name}_era5_oper_pl_{get_month(RequestedTime(year, month))}.nc"

def get_data_path_with_param_all_months(param_short_name: str, year: str, parent_path: str) -> list[str]:
    result: list[str] = []
    for month in range(1, 13):
        result.append(get_data_path_with_param(param_short_name, year, month, parent_path))

    return result;

def get_data_path_with_params(params: list[str], year: str, parent_path: str) -> list[list[str]]:
    result: list[list[str]] = []
    for param in params:
        result.append(get_data_path_with_param_all_months(param, year, parent_path))
    return result

def get_time_in_range(start: RequestedTime, end: RequestedTime) -> list[RequestedTime]:
    if start.year > end.year or (start.year == end.year and start.month > end.month):
        raise ValueError("The start time cannot be later the end time")
    
    result: list[RequestedTime] = []

    for year in range(start.year, end.year + 1):
        for month in range(start.month if year == start.year else 1, end.month + 1 if year == end.year else 13):
            # print(f"Getting Date: [{year}-{month}]")
            result.append(RequestedTime(year, month))

    return result

time_range = get_time_in_range(start_time_parsed, end_time_parsed)
print(f"the time range for the first one in ERA5 data format is: {time_range[0].get_month()}")

def get_data_path_with_range(params: list[str], time_range: list[RequestedTime], parent_path: str) -> dict[str]:
    result: dict[list[str]] = {}
    for param in params:
        result[param] = []

    for param in params:
        for requesting_time in time_range:
            result[param].append(f"{parent_path}/{param}/{requesting_time.year}/{param}_era5_oper_pl_{requesting_time.get_month()}.nc")

    return result

paths = get_data_path_with_range(data_var_picked_list, time_range, data_parent_path)
print(paths['pv'][0])

def check_data_existence(paths: dict[list[str]]) -> dict[list[str]]:
    result: dict[list[bool]] = {}
    for key in paths.keys():
        result[key] = []

    for key, ps in paths.items():
        for p in ps:
            result[key].append(os.path.exists(p))

    return result

existence_results = check_data_existence(paths)

def summarize_check_results(results: dict[list[bool]]) -> dict[str]:
    summary: dict[str] = {}
    for key, individual_result in results.items():
        any_false: bool = False in individual_result
        all_false: bool = not (True in individual_result)
        if all_false:
            summary[key] = "\033[31mAll the data for this variable\ncannot be fetched,\ncheck if the variable name\nis correct\033[0m"
        elif any_false:
            summary[key] = "\033[31mSome of the requested data\nin the given time range\ncannot be fetched.\nCheck the availability for the requested time range\033[0m"
        else:
            summary[key] = "\033[32mAll the data requested for\nthis variable in the given\ntime range can be fetched!\033[0m"

    return summary

# print(summarize_check_results(existence_results))

summaries = summarize_check_results(existence_results)
for key, summary_message in summaries.items():
    print("\n-------------------------")
    print(f"Summary for variable {key}")
    print("-------------------------")
    print(summary_message)
    print("-------------------------")
