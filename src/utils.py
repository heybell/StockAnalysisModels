import torch
import os
from datetime import datetime, timedelta
import math
import re

from src.models.mlp import MLP
from src.models.rnn import RNN

# 디바이스 가져오기
def get_device():
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    return device

# 모델 가져오기
def get_model(model, model_params):
    models = {
        'mlp' : MLP,
        'rnn' : RNN
    }
    return models.get(model.lower())(**model_params)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------
# 문자열 처리 유틸
# --------------------------------------------------------
def rm_sign(v):
    return re.sub(r'\+|\-', '', v)


# --------------------------------------------------------
# 시간 관련 유틸
# --------------------------------------------------------
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"
FORMAT_MONTH = "%Y/%m"
FORMAT_MONTHDAY = "%m/%d"

import time
from pytz import timezone

def get_str_now():
    str_now = datetime.now().strftime(FORMAT_DATETIME)
    return str_now


def get_today():
    dt = datetime.fromtimestamp(time.time(), timezone('Asia/Seoul'))
    date = dt.date()
    return date


def get_date_ago(n):
    return get_today() - timedelta(days=n)


def get_str_today():
    str_today = get_today().strftime(FORMAT_DATE)
    return str_today


def get_str_date_ago(n):
    str_date = get_date_ago(n).strftime(FORMAT_DATE)
    return str_date


def get_str_month():
    str_month = get_today().strftime(FORMAT_MONTH)
    return str_month


def get_str_date_nago(n=20, base_date=None):
    if base_date is None:
        base_date = get_today()
    if type(base_date) is str:
        base_date = datetime.strptime()
    d = base_date - timedelta(days=n)
    return d.strftime(FORMAT_DATE)


def get_dayofweek():
    """
    :return: 0-4 평일, 5-6 주말
    """
    date_today = datetime.date.today()
    int_week = date_today.weekday()
    return int_week


def get_hour_min():
    dt_now = datetime.now()
    int_hour = dt_now.hour
    int_minute = dt_now.minute
    return int_hour, int_minute


def convert_date2month(str_date):
    if len(str_date) != 8:
        return None
    return '{}/{}'.format(str_date[:4], str_date[4:6])


def convert_str2date(str_date):
    return datetime.strptime(str_date, FORMAT_DATE)


def convert_date2str(dt):
    return dt.strftime(FORMAT_DATE)


def add_months(dt, months=1):
    return dt.replace(year=dt.year + math.floor((dt.month + months) / 12), month=max((dt.month + months) % 12, 1))


def convert_datetime2str(x):
    for k in x:
        if isinstance(x[k], datetime):
            x[k] = x[k].__str__()
    return x


# --------------------------------------------------------
# 변환 관련 유틸
# --------------------------------------------------------
def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default

import datetime
from typing import Dict, List
import requests
import urllib.parse as urlparse
import xml.etree.ElementTree as ET
import json
import xmltodict

# 특일정보 공공 API
# https://www.data.go.kr/data/15012690/openapi.do
URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getHoliDeInfo"
SERVICEKEY = "CUxrX061gY3U3xkollqvEg1RH6R5KP6JNmxwwPflL4s%2B0adSAxjMLOvnZQxJFDr9YPJ4Oop2uFdUul%2FFUa7hBA%3D%3D"

def getHolidayInfo(date):
    params = {'solYear': date.year,
              'solMonth': date.month}
    params = urlparse.urlencode(params)
    request_query = URL + '?' + \
        params + '&' + 'serviceKey' + '=' + SERVICEKEY

    response = requests.get(url=request_query)
    jsonString = json.dumps(xmltodict.parse(response.text), indent=4)
    dict = json.loads(jsonString)
    print(dict)
    return dict['response']['body']


def isHoliday(data, today):
    weekday = today.weekday()

    if data['totalCount'] == '0':
        return weekday == 5 or weekday == 6

    holiday = ''
    items = data['items']['item']

    if type(items) == list:
        for item in items:
            holiday = datetime.datetime.strptime(
                item['locdate'], '%Y%m%d').date()

            if ((holiday == today) and items['isHoliday'] == 'Y') or weekday == 5 or weekday == 6:
                return True

    else:
        holiday = datetime.datetime.strptime(
            items['locdate'], '%Y%m%d').date()

    if ((holiday == today) and items['isHoliday'] == 'Y') or weekday == 5 or weekday == 6:
        return True

    return False