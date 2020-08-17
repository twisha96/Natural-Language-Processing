from collections import defaultdict

x = [["the", "man", "saw", "the", "cut"], ["the", "saw", "cut", "the", "man"], ["the", "saw"]]
y = [["D", "N", "V", "D", "N"], ["D", "N", "V", "D", "N"], ["N", "N"]]

num_examples = len(x)
tag_counts = defaultdict(int)
emission_counts = defaultdict(int)
vocab = set()
tags = set()

for itr in range(num_examples):

	# Take a single example from the corpus
	xi = x[itr].copy()
	xi.insert(0, "*")
	xi.append("STOP") 

	yi = y[itr].copy()
	yi.insert(0, "*")
	yi.append("STOP") 

	n = len(xi)

	tag_counts["*"] += 1
	for i in range(1, n-1):
		tag_counts[yi[i]] += 1
		tag_counts[tuple((yi[i-1], yi[i]))] += 1
		emission_counts[tuple((xi[i], yi[i]))] += 1
		vocab.add(xi[i])
		tags.add(yi[i])
	tag_counts[(yi[n-2],yi[n-1])] += 1
		

emission_prob = defaultdict(int)
transition_prob = defaultdict(int)

for itr in range(num_examples):

	# Take a single example from the corpus
	xi = x[itr].copy()
	xi.insert(0, "*")
	xi.append("STOP") 

	yi = y[itr].copy()
	yi.insert(0, "*")
	yi.append("STOP")

	n = len(xi)
	
	transition_prob[tuple(("*", yi[1]))] = float(tag_counts[tuple(("*", yi[1]))])/tag_counts["*"]
	for i in range(1, n-1):
		transition_prob[tuple((yi[i-1], yi[i]))] = float(tag_counts[tuple((yi[i-1], yi[i]))])/tag_counts[yi[i-1]]
		emission_prob[tuple((xi[i], yi[i]))] = float(emission_counts[tuple((xi[i], yi[i]))])/tag_counts[yi[i]]
	transition_prob[tuple((yi[n-2], "STOP"))] = float(tag_counts[tuple((yi[n-2], "STOP"))])/tag_counts[yi[n-2]]

# Print all non-zero Emission probabilities
print("Emission probabilities: o(x|y)")
for k in emission_prob:
	print(k, "   ", emission_prob[k])
print()

# Print all non-zero Transition probabilities 
print("Transition probabilities: t(y'|y)")
for k in transition_prob:
	print(k, "   ", transition_prob[k])

# print("Vocab: ", vocab)
# print("Tags: ", tags)

# Forward Algorithm
def forward(x, y):
	n = len(x)
	alpha = defaultdict(int)
	
	# Base case
	for tag in y:
		alpha[tuple((0, tag))] = transition_prob[tuple(("*", tag))]*emission_prob[tuple((x[0], tag))]
		# print("Alpha: 0", tag, alpha[tuple((0, tag))])

	# For i>=1
	for i in range(1, n):
		for y2 in y:
			for y1 in y:
				alpha[tuple((i, y2))] += alpha[tuple((i-1, y1))]*transition_prob[tuple((y1, y2))]*emission_prob[tuple((x[i], y2))]			
			# print("Alpha:", i, y2, alpha[tuple((i, y2))])
	return alpha

# Backward Algorithm
def backward(x, y):
	n = len(x)
	beta = defaultdict(int)

	# Base Case
	for tag in y:
		beta[tuple((n-1, tag))] = transition_prob[tuple((tag, "STOP"))]
		# print("Beta:", n-1, tag, beta[tuple((n-1, tag))])

	for i in range(n-2, -1, -1):
		for y1 in y:
			for y2 in y:
				beta[tuple((i, y1))] += beta[tuple((i+1, y2))]*transition_prob[tuple((y1, y2))]*emission_prob[tuple((x[i+1], y2))]
			# print("Beta:", i, y1, beta[tuple((i, y1))])
	return beta

"""
Probability under the HMM that the third word is tagged with V conditioning on x(2)
"""
print("---------------------------------------------")
print("Problem 1 - Part 2")
query_word_sequence = x[1]
query_tag_sequence = y[1]
query_index = 2
query_tag = "V"

print("Forward Algorithm")
alpha = forward(query_word_sequence, tags)
print("Backward Algorithm")
beta = backward(query_word_sequence, tags)

prob = alpha[tuple((query_index, query_tag))]*beta[tuple((query_index, query_tag))]

print("Probability under the HMM that the third word is tagged with V conditioning on x(2)")
print("Probabilty: ", prob)

"""
Probability under the HMM that the fifth word is tagged with N conditioning on x(1)
"""
print("---------------------------------------------")
print("Problem 1 - Part 3")
query_word_sequence = x[0]
query_tag_sequence = y[0]
query_index = 4
query_tag = "N"

print("Forward Algorithm")
alpha = forward(query_word_sequence, tags)
print("Backward Algorithm")
beta = backward(query_word_sequence, tags)

prob = alpha[tuple((query_index, query_tag))]*beta[tuple((query_index, query_tag))]

print("Probability under the HMM that the fifth word is tagged with N conditioning on x(1)")
print("Probabilty: ", prob)
