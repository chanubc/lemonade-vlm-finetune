"""before/after 평가 결과를 비교 표로 출력."""
import json

def load(f):
    return json.load(open(f, encoding="utf-8"))["summary"]

b = load("out/eval_before.json")
a = load("out/eval_after.json")
t = load("out/eval_textonly.json")

def p(x):
    return f"{x*100:.1f}"

def d(x):
    return f"{x*100:+.1f}"

print(f"평가 문항 수: {a['n']}  (before/after 동일 조건: 8프레임, 200704픽셀)\n")
print(f"{'구분':<14}{'before':>9}{'after':>9}{'변화':>9}")
print("-" * 42)
ov_b, ov_a = b["overall"], a["overall"]
print(f"{'전체':<13}{p(ov_b):>9}{p(ov_a):>9}{d(ov_a-ov_b):>9}")

print("[유형별]")
for k in a["by_subcategory"]:
    bb, aa = b["by_subcategory"][k], a["by_subcategory"][k]
    print(f"  {k:<12}{p(bb):>9}{p(aa):>9}{d(aa-bb):>9}")

print("[난이도별]")
for k in ["easy", "medium", "hard"]:
    bb, aa = b["by_difficulty"][k], a["by_difficulty"][k]
    print(f"  {k:<12}{p(bb):>9}{p(aa):>9}{d(aa-bb):>9}")

print(f"\n참고 — 텍스트만(before): {p(t['overall'])}%  |  찍기: 25%")
