n, c, r = map(int, input().split())
zabuli = set(map(int, input().split()))
zapas = set(map(int, input().split()))

count = 0

both = zabuli & zapas
zabuli -= both
zapas -= both

for i in range(1, n + 1):
    if i in zabuli:
        if i - 1 in zapas:
            zapas.remove(i - 1)
            zabuli.remove(i)
        elif i + 1 in zapas:
            zapas.remove(i + 1)
            zabuli.remove(i)

print(n - len(zabuli))