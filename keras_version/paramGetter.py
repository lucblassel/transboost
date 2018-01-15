"""
-----------------------------------------------
# @Author: Luc Blassel <lucblassel>
# @Date:   2018-01-15T12:04:51+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   lucblassel
# @Last modified time: 2018-01-15T12:47:39+01:00

parameter getter

-----------------------------------------------
"""
# import os
import json

def reader(filepath):
    data = json.load(open(filepath))
    return data

def main():
    reader('config.json')

if __name__ == "__main__":
    main()
