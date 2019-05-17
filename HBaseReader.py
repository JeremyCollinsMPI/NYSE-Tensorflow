import happybase

def get_HBase_table(tablename, url = '127.0.0.1'):
	connection = happybase.Connection(url)
	table = connection.table(tablename)
	return table

def scan(table, start, stop):
	try:
		return table.scan(row_start=start, row_stop=stop)
	except:
		table = get_HBase_table(table)
		return table.scan(row_start=start, row_stop=stop)

