import pandas as pd

cnt = 0
names = dict() 
with open("dsl.py") as f:
    for line in f:
        if 'def' in line:
            names[line[4:-2]] = 0
f.close()
#print(names)

with open("solvers.py") as f:
    for line in f:
        line = line.strip()
        if 'def' not in line and 'return' not in line:
            for i in names.keys():
                if i in line:
                    names[i]+=1
f.close()
    #print(check.count('branch'))
    #for i in names.keys():
        
    #    names[i]+=check.count(i)
    #f.close()

#print(names)
import matplotlib.pyplot as plt



names = {k: v for k, v in sorted(names.items(), key=lambda item: item[1], reverse=True)}
items = list(names.items())

last_5_items = items[:30]

# Convert back to a dictionary
last_5_dict = dict(last_5_items)

print(f"30 most occuring primitives {last_5_dict}")
print(f'\n\nAll of them {names}')
#values = list(names.values())
#x_labels = range(len(names))

#plt.plot(x_labels, values)
#plt.xticks(x_labels, names.keys())  


#plt.savefig('test.png')
