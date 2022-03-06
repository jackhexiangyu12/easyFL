day_of_month={31,28,31,30,31,30,31,31,30,31,30,31}
remainder={}
for i in range(1,13):
    remainder[i]=day_of_month[i]%7
print(remainder)

day_of_month_leap={31,29,31,30,31,30,31,31,30,31,30,31}
year=range(1900,2001)
leap=[0]*10

#已知1900.1.1是星期一，判断1901年第一天是星期几

for i in range(1,13):
    if i in year:
        if i%4==0:
