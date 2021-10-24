import numpy
import pandas

import time


def Clean(path):
	training = pandas.read_csv(path)

	# Predictor we want to use (or make another variable based on it)
	to_use = ['TRAN_AMT', 'WF_dvc_age', 'ALERT_TRGR_CD', 'DVC_TYPE_TXT', 'CARR_NAME', 'PH_NUM_UPDT_TS' , 'PWD_UPDT_TS']


	for col in training.columns:
		if col == 'dataset_id': continue 
		if col == 'FRAUD_NONFRAUD': continue
		if col in  to_use: continue
		training.drop(columns = [col], inplace = True)



	def fraud_mapping(x):
	    if x=="Non-Fraud": return 1
	    elif x=="Fraud": return 0

	if 'FRAUD_NONFRAUD' in list(training.columns):
	    training['FRAUD_NONFRAUD'] = training['FRAUD_NONFRAUD'].map(lambda x : fraud_mapping(x) )



	# Mapping the two catigorical values to numeric
	def fun_ALERT_TRGR_CD(x):
	    if x=='MOBL': return 1
	    if x=='ONLN': return 2
	    
	# Mapping DVC_TYPE_TXT to 
	def fun_DVC_TYPE_TXT(x):
	    if x=='DESKTOP': return 1
	    if x=='MOBILE': return 2
	    if x=="PHONE": return 3
	    if x=="TABLET": return 4

	training['ALERT_TRGR_CD'] = training['ALERT_TRGR_CD'].map( fun_ALERT_TRGR_CD  ,  na_action='ignore' )
	training['ALERT_TRGR_CD'] = training['ALERT_TRGR_CD'].fillna(0)

	training['DVC_TYPE_TXT'] = training['DVC_TYPE_TXT'].map( fun_DVC_TYPE_TXT  ,  na_action='ignore' )
	training['DVC_TYPE_TXT'] = training['DVC_TYPE_TXT'].fillna(0)




	# Will add new variable: CARR_Missing 0: not missing  1: missing
	# for when CARR_NAME, RGN_NAME, STATE_PRVNC_TXT are all missing together

	training['CARR_Missing'] = training['CARR_NAME']
	training['CARR_Missing'] = training['CARR_Missing'].map( lambda x : 0 ,  na_action='ignore' )
	training['CARR_Missing'] = training['CARR_Missing'].fillna(1)
	training['CARR_Missing'] = training['CARR_Missing'].astype(int)
	training.drop(columns = ['CARR_NAME'], inplace=True)

	# Will add new variable: PH_NUM_UPDT_TS_Missing 0: not missing  1: missing

	training['PH_NUM_UPDT_TS_Missing'] = training['PH_NUM_UPDT_TS']
	training['PH_NUM_UPDT_TS_Missing'] = training['PH_NUM_UPDT_TS_Missing'].map( lambda x : 0 ,  na_action='ignore' )
	training['PH_NUM_UPDT_TS_Missing'] = training['PH_NUM_UPDT_TS_Missing'].fillna(1)
	training['PH_NUM_UPDT_TS_Missing'] = training['PH_NUM_UPDT_TS_Missing'].astype(int)


	# In[4]:


	# changing dates to python struct_time values
	dates =  [ 'PWD_UPDT_TS' ,'PH_NUM_UPDT_TS']

	# Differnt formatting options:
	#  month/day/year hour:minute:second  = %m/%d/%Y %H:%M:%S   (PWD_UPDT_TS, TRAN_TS, PH_NUM_UPDT_TS)
	#  year-month-day hour:minute:second  = %Y-%m-%d %H:%M:%S    (CUST_SINCE_DT)
	def str_to_time(x):
	    if "/" in x:
	        return time.strptime(x, "%m/%d/%Y %H:%M:%S")
	    else:
	        return time.strptime(x, "%Y-%m-%d %H:%M:%S")

	# it seems 'PH_NUM_UPDT_TS' gives days some days of the month as 0, will map these to day = 1
	# 'PWD_UPDT_TS' has a few 6/31's (the month of june only has 30 days) with map these to day = 30
	def fix_day(x):
	    if "/" in x:
	        x = x.split("/")
	        day = int(x[1])
	        month = int(x[0])
	        if day==0: day += 1
	        if day==31 and month==6: day = 30
	        x = x[0] + "/" + str(day) + "/"+ x[2]
	    return x

	for column in dates :
	    training[column] = training[column].map(lambda x : str_to_time(fix_day(x)) ,  na_action='ignore' )


	# In[5]:


	usefull_dates = ['PWD_UPDT_TS tm_year' ,'PWD_UPDT_TS tm_hour', 'PWD_UPDT_TS tm_min',  'PH_NUM_UPDT_TS tm_year', 'PH_NUM_UPDT_TS tm_mon', 'PH_NUM_UPDT_TS tm_hour', 'PH_NUM_UPDT_TS tm_sec', 'PH_NUM_UPDT_TS tm_yday' ]

	# 'PWD_UPDT_TS tm_year'
	col = 'PWD_UPDT_TS tm_year'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_year, na_action='ignore')

	# 'PWD_UPDT_TS tm_hour'
	col = 'PWD_UPDT_TS tm_hour'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_hour, na_action='ignore')

	# 'PWD_UPDT_TS tm_min'
	col = 'PWD_UPDT_TS tm_min'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_min, na_action='ignore')

	# PH_NUM_UPDT_TS tm_year
	col = 'PH_NUM_UPDT_TS tm_year'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_year, na_action='ignore')

	# PH_NUM_UPDT_TS tm_mon
	col = 'PH_NUM_UPDT_TS tm_mon'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_mon, na_action='ignore')

	# 'PH_NUM_UPDT_TS tm_hour'
	col = 'PH_NUM_UPDT_TS tm_hour'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_hour, na_action='ignore')

	# PH_NUM_UPDT_TS tm_sec
	col = 'PH_NUM_UPDT_TS tm_sec'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_sec, na_action='ignore')

	# 'PH_NUM_UPDT_TS tm_yday'
	col = 'PH_NUM_UPDT_TS tm_yday'
	training[col] = training[ col.split(' ')[0] ].map(lambda x : x.tm_yday, na_action='ignore')

	training.drop(columns = dates, inplace = True)


	# replace any missing values with averages
	for col in training.columns:
		if not training[col].isnull().values.any(): continue
		training[col] = training[col].fillna(training[col].mean() )





	# print out a new cleaned csv
	training.to_csv(path[:len(path)-4]+'_cleaned.csv' , index=False )


