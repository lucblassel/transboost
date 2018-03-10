"""
# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-15T00:21:20+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T14:08:00+01:00

Romain Gautron
"""
import os
import sys
############################################################################
# MAIN                                                                        #
############################################################################

def readConfigs():
	"""
	Read configurations from json file. In this json file, it defines the values of some parameters that we will use in the other functions. 
	Run the program of boosting to get the results. 
	#Input:
	No argument, but it will read the directory to find the json file. 
	#Output:
	The results of running boosting.py
	"""

    if len(sys.argv) <= 1:
        print("please specify path to config batch...")
        sys.exit()
    else:
        configBatch = open(sys.argv[1],'r')
        configs = configBatch.read().split('\n')

        for conf in configs:
            if len(conf)>0 and conf[0]!="#":
                print(5*"\n"+"calling next configuration"+5*"\n")
                os.system('python3 boosting.py '+conf)




def main():
    readConfigs()

if __name__ == '__main__':
    main()
