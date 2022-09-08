import numpy as np
import pandas as pd
import math

## Log Sigmoidal Transfer Function
def func1(Ih):
    return 1/(1+np.exp(-Ih))
## Tan Sigmoidal Transfer Function
def func2(Io):
    return 1/(1+np.exp(-Io))


iter =5000 # Number of Iterations
mse=0 # mean square error


n=0.7 #learning rate
P=55
H=9
op=1
L=5

inp_data="Data_input.xlsx"
df=pd.read_excel(inp_data)
Ii=np.array(df)


print(Ii)
Ii=Ii.transpose()


 
for l in range (L):
    a = Ii[l].min()
    b = Ii[l].max()
    for p in range (P):
        Ii[l][p]=(0.1+0.8*(Ii[l][p]-a)/(b-a))
# Ii= Ii.transpose()
print("\nnormalized input\n",Ii)
trp_pat=Ii





ip=len(trp_pat[0]) # Number of Inputs : 5
tr_p=len(trp_pat)  # Number of Training patterns : 90s
# trg_op=np.loadtxt('job_output_data_2.txt')
# To=trg_op

out_data="Data_output.xlsx"
df=pd.read_excel(out_data)
To=np.array(df)
print("To\n",To)
c = To.min()
d = To.max()
print("\n c,d \n",c,d)
for p in range (P):
    for o in range (op):
        To[p][o]=(0.1+0.8*(To[p][o]-c)/(d-c))
# To= To.transpose()
print("\nnormalized output\n",To)




Ih=np.zeros((tr_p,H))
Oh=np.zeros((tr_p,H))
Oo=np.zeros((tr_p,op))
del_V=np.zeros((tr_p,ip,H))
del_W=np.zeros((tr_p,H,op))
del_V_fin=np.zeros((ip,H))
del_W_fin=np.zeros((H,op))
del_e=np.zeros((op,ip,H))
del_V_sum=np.zeros((ip,H))

print("No. of Inputs: ",ip)   # Number of Inputs : 5
print("No. of Training Patterns: ",tr_p) #No. of Training Patterns:  90
print("No. of Hidden Neurons: ",H)
print("Learning Rate: ",n)
print("No. of Output Neurons: ",op)
print("No. of Iterations: ",iter)

np.random.seed(4)
V=0.3*np.random.rand(ip,H)
np.random.seed(5)
W=0.3*np.random.rand(H,op)

print("\n\nStuct V----------------- \n")
print(V)
print("\n\nStuct W----------------- \n")
print(W)

I=trp_pat
for p in range(0,tr_p):
    Ih[p,:]=I[p,:].dot(V)
print("\n\nInput to Hidden Neurons----------------- \n")
print(Ih)

for p in range(0,tr_p):
    for j in range(0,H):
        Oh[p,j]=func1(Ih[p,j])

print("\n\n Output of Hidden Neurons----------------- \n")
print(Oh)

Io=Oh.dot(W)
print("\n\n Input to Output Neurons----------------- \n")
print("Io\n",Io)

for p in range(0,tr_p):
    for k in range(0,op):
        Oo[p,k]=func2(Io[p,k])

print("\n\n Output of Output Neurons Initial ----------------- \n")
print(Oo)

with open("iteration_mse.txt","w") as f1:
    f1.write("iteration_no.\t\t\tmse")




for a in range(1,iter+1):
    for p in range(0,tr_p):
        for h in range(0,H):
            for k in range(0,op):
                del_W[p,h,k]=n*Oh[p,h]*(1-pow(Oo[p],2))*(To[p]-Oo[p])

        for k in range(0,op):
            for i in range(0,ip):
                for j in range(0,H):
                    del_e[k,i,j]=n*(To[p]-Oo[p])*(1-pow(Oo[p],2))*W[j,k]*Oh[p,j]*(1-Oh[p,j])*I[p,i]
            del_V[p,:,:]=del_V[p,:,:]+(del_e[k,:,:])

    for p in range(0,tr_p):
        del_W_fin=(1/tr_p)*del_W[p,:,:]
        del_V_fin=(1/tr_p)*del_V[p,:,:]

    V=V+del_V_fin
    W=W+del_W_fin

    for p in range(0,tr_p):
        Ih[p,:]=I[p,:].dot(V)

    for p in range(0,tr_p):
        for j in range(0,H):
            Oh[p,j]=func1(Ih[p,j])

    Io=Oh.dot(W)
    mse = 0
    for p in range(0,tr_p):
        for k in range(0,op):
            Oo[p,k]=func2(Io[p,k])
            mse=mse+0.5*(To[p]-Oo[p,k])**2
    mse=(1/tr_p)*mse
    mse=float(mse+(1/iter)*mse)
    print("mse\n",mse)
    with open("iteration_mse.txt","a") as f1:
        f1.write("\n")
        f1.write(str(a))
        f1.write("\t\t\t")
        f1.write(str(mse))
        


print("\n \n V Final  \n")
print(V)
print("\n---------\n")
print("\n W Final \n")
print(W)

print("\n\n Target Output 1 ----------------- \n")
print(To)
print("\n\n Output of Output Neurons Initial ----------------- \n")
print(Oo)

print("\n\n Mean Square Error :  ",mse)

fpr=open("Output_data.txt","w")
for p in range(0,tr_p):
    fpr.write(str(Oo[p]))
    fpr.write("\n")
fpr.close()
