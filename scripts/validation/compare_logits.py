import sys, numpy as np
ref, ours, V = sys.argv[1], sys.argv[2], int(sys.argv[3])
toks = np.fromfile(sys.argv[4], dtype=np.uint32)
a = np.fromfile(ref, dtype=np.float32).reshape(-1, V)
b = np.fromfile(ours, dtype=np.float32).reshape(-1, V)
n = min(len(a), len(b)); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
def logsm(x): x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
la, lb = logsm(a), logsm(b)
pa = np.exp(la)
kl = (pa * (la - lb)).sum(1)
top1 = (a.argmax(1) == b.argmax(1))
t5a = np.argsort(-a, 1)[:, :5]; t5b = np.argsort(-b, 1)[:, :5]
top5 = np.array([len(set(x) & set(y)) / 5 for x, y in zip(t5a, t5b)])
ca = a - a.mean(1, keepdims=True); cb = b - b.mean(1, keepdims=True)
dmax = np.abs(ca - cb).max(1); rel = np.abs(ca - cb).max(1) / np.abs(ca).max(1)
nll_a = -la[np.arange(n - 1), toks[1:n]]; nll_b = -lb[np.arange(n - 1), toks[1:n]]
print(f"positions {n}")
print(f"KL(ref||ours): mean {kl.mean():.2e}  median {np.median(kl):.2e}  p95 {np.percentile(kl,95):.2e}  max {kl.max():.2e}")
print(f"top-1 agreement {top1.mean()*100:.1f}%  ({(~top1).sum()} disagreements)   top-5 overlap {top5.mean()*100:.1f}%")
print(f"max|Δ centred logit|: mean {dmax.mean():.3f}  max {dmax.max():.3f}  (relative {rel.mean()*100:.2f}%)")
print(f"perplexity on text: ref {np.exp(nll_a.mean()):.3f}   ours {np.exp(nll_b.mean()):.3f}   (Δ {100*(np.exp(nll_b.mean())/np.exp(nll_a.mean())-1):+.2f}%)")
for i in np.where(~top1)[0][:8]:
    s = np.sort(pa[i])[::-1]
    print(f"  pos {i}: ref top {a[i].argmax()} p={s[0]:.3f} (2nd {s[1]:.3f}), ours top {b[i].argmax()}")
