"""
Given: Input sentence and two possible parse trees

"""
from collections import defaultdict

# Input sentence
x = ("the boy saw the man with a telescope").split()

# Parse Tree 1
parse_tree_1 = [
("S", ("NP", "VP")), # level 1
("NP", ("D", "N")), ("VP", ("VP", "PP")), # level 2
("D", "the"), ("N", "boy"), ("V", "saw"), ("NP", ("NP", "PP")), # level 3
("NP", ("D", "N")), ("PP", ("P", "NP")), # level 4
("D", "the"), ("N", "man"), ("P", "with"), ("NP", ("D", "N")), # level 5
("D", "a"), ("N", "telescope") # level 6
]

# Parse Tree 2
parse_tree_2 = [
("S", ("NP", "VP")), # level 1
("NP", ("D", "N")), ("VP", ("V", "NP")), # level 2
("D", "the"), ("N", "boy"), ("VP", ("V", "NP")), ("PP", ("P", "NP")), # level 3
("V", "saw"), ("NP", ("D", "N")), ("P", "with"), ("NP", ("D", "N")), # level 4
("D", "the"), ("N", "man"), ("D", "a"), ("N", "telescope") # level 5
]

# Counting Rules
unary_rules_count = defaultdict(int)
binary_rules_count = defaultdict(int)
N = set()

count_X = defaultdict(int)

for rule in parse_tree_1:
	N.add(rule[0])
	count_X[rule[0]] += 1
	if type(rule[1]) == type(""):
		unary_rules_count[rule] += 1
	else:
		binary_rules_count[rule] += 1

for rule in parse_tree_2:
	N.add(rule[0])
	count_X[rule[0]] += 1
	if type(rule[1]) == type(""):
		unary_rules_count[rule] += 1
	else:
		binary_rules_count[rule] += 1

# Rule Probabilities
unary_rules_prob = defaultdict(int)
binary_rules_prob = defaultdict(int)

print("Unary Rule Probabilities")
for rule in unary_rules_count:
	unary_rules_prob[rule] = unary_rules_count[rule]/count_X[rule[0]]
	print(rule, unary_rules_prob[rule])

print("Binary Rule Probabilities")
for rule in binary_rules_count:
	binary_rules_prob[rule] = binary_rules_count[rule]/count_X[rule[0]]
	print(rule, binary_rules_prob[rule])

def inside_algo(x):
	n = len(x)
	alpha = defaultdict(int)

	# Base case
	for i in range(n):
		for state in N:
			if unary_rules_prob[(state, x[i])]:
				alpha[(i, i, state)] = unary_rules_prob[(state, x[i])]
				# print("alpha ", i, i, state, " : ", alpha[(i, i, state)])

	for l in range(1, n-1):
		for i in range(n-l):
			j = i + l
			# X -> YZ
			for X in N:
				for b_rule in binary_rules_prob:
					if b_rule[0] == X:
						Y = b_rule[1][0]
						Z = b_rule[1][1]
						for k in range(i, j+1):
							if alpha[(i, k, Y)] and alpha[(k+1, j, Z)]:
								alpha[(i, j, X)] += alpha[(i, k, Y)]*alpha[(k+1, j, Z)]*binary_rules_prob[b_rule]
				
				# if alpha[(i, j, X)]:
					# print("alpha ", i, j, X, " : ", alpha[(i, j, X)])

	return alpha

def outside_algo(x):
	n = len(x)
	beta = defaultdict(int)

	# Base case
	beta[(0, n-1, 'S')] = 1
	
	for l in range(n-2, -1, -1):
		for i in range(n-l):
			j = i + l
			
			for X in N:
				for b_rule in binary_rules_prob:
					# A -> XC
					if b_rule[1][0] == X: 
						A = b_rule[0]
						C = b_rule[1][1]
						for k in range(j+1, n):
							beta[(i, j, X)] += beta[(i, k, A)]*alpha[(j+1, k, C)]*binary_rules_prob[b_rule]

					# A -> BX
					elif b_rule[1][1] == X: 
						A = b_rule[0]
						B = b_rule[1][0]
						for k in range(i):
							beta[(i, j, X)] += beta[(k, j, A)]*alpha[(k, i-1, B)]*binary_rules_prob[b_rule]

				# if beta[(i, j, X)]:
					# print("beta ", i, j, X, " : ", beta[(i, j, X)])

	return beta


alpha = inside_algo(x)
beta = outside_algo(x)

print("---------------------------------------------")
print("Problem 2 - Part 2")
print("Probability under the PCFG that NP spans (4, 8) (i.e., “the man with a telescope”) conditioning on x")
print(alpha[(3, 7, 'NP')]*beta[(3, 7, 'NP')])

print("---------------------------------------------")
print("Problem 2 - Part 3")
print("Probability under the PCFG that VP spans (3, 5) (i.e., “saw the man”) conditioning on x")
print(alpha[(2, 4, 'VP')]*beta[(2, 4, 'VP')])
