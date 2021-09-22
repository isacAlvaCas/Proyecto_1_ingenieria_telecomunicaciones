val=[7.2,787,87,87,9,-1]
for i in val:
    a = int(bin(int(i) if int(i)>0 else int(i)+(1<<16))[2:])
    print(a)