import math
import sys
import pdb
from collections import Counter
from functools import reduce


def compute_bleu(reflists, hyps, n_max=4, use_shortest_ref=False):
    assert len(reflists) == len(hyps)
    
    N = len(hyps)

    prec_mean = 1.0  # TODO: Implement
    for n in range(1, 5):
    	pn_num = 0
    	pn_deno = 0
    	for i in range(N):
    		an, bn = get_ngram_counts(reflists[i], hyps[i], n)
    		pn_num += an
    		pn_deno += bn
    		# print (an, bn)
    	prec_mean *= (float(pn_num)/pn_deno)
    	# print("p", float(pn_num)/pn_deno)
    prec_mean = prec_mean ** 0.25
    # print("Prec mean: ", prec_mean)

    brevity_penalty = 0  # TODO:Implement
    H = 0
    R = 0
    for i in range(N):
    	h = len(hyps[i])
    	H += h
    	
    	r_max = sys.maxsize
    	r = 0
    	refs = reflists[i]
    	for ref in refs:
    		if abs(len(ref) - h) < r_max:
    			r = len(ref)
    	R += r
    brevity_penalty = min(1, math.exp(1 - (float(R)/H)))
    # print("Brevity Penalty: ", brevity_penalty)

    bleu = brevity_penalty * prec_mean

    return bleu


def get_ngram_counts(refs, hyp, n):
    hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
    num_hyp_ngrams = max(1, len(hyp_ngrams))  # Avoid empty
    
    hyp_ngrams_count = Counter(hyp_ngrams)
    # print (hyp_ngrams_count)

    refs_ngrams_count = []
    for ref in refs:
    	ref_ngrams = [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
    	refs_ngrams_count.append(Counter(ref_ngrams))

    # TODO: Implement
    num_hyp_ngrams_in_refs_clipped = 0  
    for n_gram in hyp_ngrams_count:
    	c = hyp_ngrams_count[n_gram]
    	# print(n_gram)
    	# print ("Count in hypothesis: ", c)
    	max_ref_count = 0
    	for i, ref in enumerate(refs):
    		max_ref_count = max(max_ref_count, refs_ngrams_count[i][n_gram])
    	# print ("Max count in ref: ", max_ref_count)
    	num_hyp_ngrams_in_refs_clipped += min(c, max_ref_count) 
		
    return num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams
