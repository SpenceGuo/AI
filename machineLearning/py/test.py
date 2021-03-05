import random

test = [1,1,1,1,2,3,3,3,3,3]

count1 = 0
count2 = 0
count3 = 0
for i in range(10000):
    if random.choice(test) == 1:
        count1 += 1
    elif random.choice(test) == 2:
        count2 += 1
    else:
        count3 += 1
print('1-{} \n2-{} \n3-{}'.format(round(count1/10000, 3), round(count2/10000, 2),
                                  round(count3/10000, 2)))
