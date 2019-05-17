from __future__ import division
import numpy as np
import random
import math
import time
import datetime as dt
import geocoder
from math import radians
from math import sin, cos, asin, sqrt

def log(x):
	return math.log(x)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def rep(x,y):
	new=[]
	for m in xrange(y):
		new.append(x)
	return new	        
def copy(x):
	new=[]
	for member in x:
		new.append(member)
	return new	
def strlist(x):
	new=[]
	for member in x:
		new.append(str(member))
	return new
def sigmoid(x):
	y=1/(1+np.exp(-x))
	return y
def time_between(a,b):
	exp=[int(x) for x in (a.split(',')[0]).split('/')]
# 	print exp
	a=dt.datetime(exp[0],exp[1],exp[2],exp[3],exp[4],exp[5])
	exp=[int(x) for x in (b.split(',')[0]).split('/')]
# 	print exp
	b=dt.datetime(exp[0],exp[1],exp[2],exp[3],exp[4],exp[5])
	return (b-a).total_seconds()	
def unique(x):
	new=[]
	for member in x:
		if not member in new:
			new.append(member)
	return new
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def isString(x):
	try:
		a = x + 'hello'
		return True
	except:
		return False
		
def isList(x):
	try:
		a = x + []
		return True
	except:
		return False

def inDict(element, dictionary):
	try:
		x = dictionary[element]
		return True
	except:
		return False

def find_calendar_month(timestamp):
	readable = time.ctime(timestamp)
	return readable.split(' ')[1]

def find_day_of_the_week(timestamp):
	readable = time.ctime(timestamp)
	return readable.split(' ')[0]


