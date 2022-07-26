import os
import sys
import subprocess

x = {}

for k in range(100):
    
    wrong = 0
    correct = 0

    for i in range(1,8):
        for j in range(1,5):
            fn = "p" + str(i) + "-s"+str(j) + "-v1.csv" 
            # result = os.popen("python3 model.py " + "p" + str(i) + "-s"+str(j) +"-v1.csv")
            result = str(subprocess.run(["python3" , "model.py" , fn , str(k)] , capture_output = True , text=True))
            # CompletedProcess(args=['python3', 'model.py', 'p1-s1-v1.csv'], returncode=0, stdout=b'1\n', stderr=b'')
            result = result[90:91]

            # print(result)
            if str(result) == str(0):
                wrong += 1
            else:
                correct += 1

    print()
    # print("Correct = " + str(correct))
    # print("Wrong = " + str(wrong))

    accuracy = str(100 * correct/(correct + wrong))
    print("k = " + str(k) + " : Accuracy = " + accuracy)
    x[k] = accuracy

xSort = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
print(x)
