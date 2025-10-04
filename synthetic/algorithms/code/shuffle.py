import re,random
random.seed(1337)
a=re.split(r"\s+", open(r"samples\\train_big.txt","r",encoding="utf-8",errors="ignore").read().strip())
random.shuffle(a)
open(r"samples\\train_big_shuf.txt","w",encoding="utf-8").write(" ".join(a))