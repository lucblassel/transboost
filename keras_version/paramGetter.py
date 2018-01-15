"""
-----------------------------------------------
# @Author: Luc Blassel <lucblassel>
# @Date:   2018-01-15T12:04:51+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-15T15:08:16+01:00

parameter getter

-----------------------------------------------
"""
# import os
import json

def reader(filepath):
    data = json.load(open(filepath))
    return data

def main():
    data = reader('config.json')
    print(data)

if __name__ == "__main__":
    main()
