a = [1,2,3]
b= [2,2,3]
c= [3,0,2]

num =[]
for index in range(3):
    a1=a[index]
    b1=b[index]
    c1=c[index]
    d = [a1,b1,c1]
    # print(d)
    i = max(d,key=d.count)
    num.append(i)
print(num)