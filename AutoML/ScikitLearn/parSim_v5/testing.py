from time import perf_counter as pf

long = lambda: [i**2 for i in range(100000)]

t1 = pf()
long()
t2 = pf()
long()
t3 = pf()
long()
t4 = pf()

print(f"t4 - t1 diff: {t4 - t1}")
print(f"t3 - t2 diff: {t3 - t2}")