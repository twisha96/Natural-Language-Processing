import matplotlib.pyplot as plt
import numpy as np

x = [1, 5, 10, 100, 200] 
lr1 = [964.849455, 584.203715, 544.6881, 52.470337, 7.95881]
lr2 = [558.5273, 201.9646, 121.3206, 3.9874, 2.4408]
lr3 = [216.260753, 74.481032, 31.483695, 2.309834, 3.569439]
lr4 = [152.704598, 41.323861, 15.678454, 13.13868, 132.379785]
# lr5 = [170.087492, 89.930339, 74.010526, 2262.670114, 1386712.669]


plt.plot(lr1, "-bo", label="lr = 0.00001")
plt.plot(lr2, "-ro", label="lr=0.00003")
plt.plot(lr3, "-go", label="lr=0.0001")
plt.plot(lr4, "-mo", label="lr=0.0003")
# plt.plot(lr5, "-co", label="lr=0.001")

plt.xticks(np.arange(5), ('1', '5', '10', '100', '200'))
plt.legend(loc="upper right")
plt.xlabel('Embedding size and hidden layer size')
plt.ylabel('Perplexity')
plt.title('nlayer=0, Perplexity w.r.t. varying Embedding size')
plt.show()
plt.close()

lr1 = [964.849455, 584.203715, 544.6881, 52.470337, 7.95881]
lr2 = [781.672677, 246.509965, 179.219962, 4.592082, 1.75678]
lr3 = [411.108823, 145.855155, 87.999586, 2.595045, 2.726216]
lr4 = [174.60614, 120.391, 75.205279, 107.508802, 147.332368]
lr5 = [192.837153, 184.140702, 197.06997, 220.053004, 247.400328]

plt.plot(lr1, "-bo", label="lr = 0.00001")
plt.plot(lr2, "-ro", label="lr=0.00003")
plt.plot(lr3, "-go", label="lr=0.0001")
plt.plot(lr4, "-mo", label="lr=0.0003")
plt.plot(lr5, "-co", label="lr=0.001")

plt.xticks(np.arange(5), ('1', '5', '10', '100', '200'))
plt.legend(loc="upper right")
plt.xlabel('Embedding size and hidden layer size')
plt.ylabel('Perplexity')
plt.title('nlayer=1, Perplexity w.r.t. varying Embedding size')
plt.show()
plt.close()

plt.plot(x[0], "-bo", label="nlayers = 0") 
plt.plot(x[1], "-ro", label="nlayers = 1")
plt.legend(loc="upper right")
plt.xlabel('Number of epochs')
plt.ylabel('Average Training Loss')
plt.title('Training Loss w.r.t. number of epochs')
plt.show()