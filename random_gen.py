import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.DataFrame(columns=(0, 1, 2))

class_A = [-0.5, 0]
class_B = [0.5, 0]

_2dArray = []

#generate, and label datapoints 
for i in range(20):

    arrayVal = []
    randomx = np.random.uniform(-1, 1)
    randomy = np.random.uniform(-1, 1)
    
    distanceA = math.sqrt(((class_A[0]-randomx)**2)+((class_A[1]-randomy)**2))
    distanceB = math.sqrt(((class_B[0]-randomx)**2)+((class_B[1]-randomy)**2))
    
    arrayVal.append(randomx)
    arrayVal.append(randomy)

    if(distanceA<distanceB):
        arrayVal.append(-1)
    
    else:
        arrayVal.append(1)
    
    print(arrayVal)


    print("D_A=", distanceA)
    print("D_B=", distanceB)
    
    _2dArray.append(arrayVal)



for i in range(20):
    df.loc[i] = _2dArray[i]


print(df)



plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter([class_A[0], class_B[0]], [class_A[1], class_B[1]])
plt.show()
np.savetxt('sinegen.txt', df.values)



















