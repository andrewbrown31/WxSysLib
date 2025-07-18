#!/usr/bin/env python

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def generate_datetimes_months(start_date, end_date, interval=1):

    # Create a list to store the datetimes
    date_list = []

    # Loop to generate datetimes at the specified interval
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += relativedelta(months=interval)

    return date_list

def generate_datetimes(start_date, end_date, interval=1):

    # Create a list to store the datetimes
    date_list = []

    # Loop to generate datetimes at the specified interval
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += relativedelta(hours=interval)

    return date_list


