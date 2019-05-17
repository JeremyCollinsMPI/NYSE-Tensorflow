class OneHotEncoder:
	def __init__(self, sequence, categories = None):
		self.sequence = sequence
		if categories == None:
			categories = []
			for member in sequence:
				if not member in categories:
					categories.append(member)
			self.categories = categories		
		else:
			self.categories = categories
	def turn_category_to_index(self, category):	
		return self.categories.index(category)
	def encode(self):
		result = [self.turn_category_to_index(x) for x in self.sequence]
		self.encoding = result
		self.depth = len(self.categories)
		return self
