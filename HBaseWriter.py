import happybase

def write_table1():
	HBASE_URL = '127.0.0.1'
	TABLE = 'StockMarket'
	COLUMN_FAMILY = 'cf'
	OPENING_PRICE = 'opening_price'
	FILENAME = 'financial2.txt'

	connection = happybase.Connection(HBASE_URL)
	table = connection.table(TABLE)
	file = open(FILENAME, 'r').readlines()

	for m in xrange(len(file)):
		line = file[m].replace('\n','')
		line = line.split('\t')
		ticker = line[7]
		timestamp = line[0]
		opening_price = line[1]
		if int(timestamp) < 1000000000:
			timestamp = '0' + timestamp
		rowkey = ticker + '_' + timestamp
		table.put(rowkey, {COLUMN_FAMILY + ':' + OPENING_PRICE: opening_price})

def write_table2():
	HBASE_URL = '127.0.0.1'
	TABLE = 'StockMarket2'
	COLUMN_FAMILY = 'cf'
	OPENING_PRICE = 'opening_price'
	FILENAME = 'financial2.txt'

	connection = happybase.Connection(HBASE_URL)
	table = connection.table(TABLE)
	file = open(FILENAME, 'r').readlines()

	for m in xrange(len(file)):
		line = file[m].replace('\n','')
		line = line.split('\t')
		ticker = line[7]
		timestamp = line[0]
		opening_price = line[1]
		if int(timestamp) < 1000000000:
			timestamp = '0' + timestamp
		rowkey = timestamp + '_' + ticker
		table.put(rowkey, {COLUMN_FAMILY + ':' + OPENING_PRICE: opening_price})
