import numpy as np

BIG = 1e6
CLASS_NAMES = ['Low', 'Moderate', 'Severe']

def trimf(x, a, b, c):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / (c - b + 1e-12), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    right = np.where(d > c, (d - x) / (d - c + 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)

# Output sets for Mamdani (risk score 0-100)
OUT_SETS = {0: ('trap', (0, 0, 15, 40)), 1: ('tri', (25, 50, 75)), 2: ('trap', (60, 85, 100, 100))}
SCORE_CUTS = (35.0, 65.0)   # score <= 35 Low, 35-65 Moderate, > 65 Severe

def _mf(kind, p, x):
    if kind == 'tri':
        return trimf(x, *p)
    if kind == 'trap':
        return trapmf(x, *p)
    if kind == 'month':
        return np.asarray(p, float)[np.asarray(x).astype(int) - 1]
    raise ValueError(kind)

class FuzzySystem:
    def __init__(self, variables, rules=None, default_class=1):
        self.variables = variables          # {var: {label: (kind, params)}}
        self.rules = list(rules or [])      # [(antecedent dict, class_int)]
        self.default_class = default_class  # used when no rule fires

    # ---- fuzzification and rule firing --------------------------------
    def fuzzify(self, data):
        mu = {}
        for v, labels in self.variables.items():
            x = np.asarray(data[v], float)
            for lab, (kind, p) in labels.items():
                mu[(v, lab)] = _mf(kind, p, x)
        return mu

    def firing(self, mu, rules=None):
        rules = self.rules if rules is None else rules
        n = len(next(iter(mu.values())))
        F = np.ones((n, len(rules)))
        for j, (ant, _) in enumerate(rules):
            for v, lab in ant.items():
                F[:, j] = np.minimum(F[:, j], mu[(v, lab)])
        return F

    # ---- class-vote inference (fast, used by GA) -------------------------
    def class_scores(self, data):
        F = self.firing(self.fuzzify(data))
        S = np.zeros((F.shape[0], 3))
        for j, (_, c) in enumerate(self.rules):
            S[:, c] = np.maximum(S[:, c], F[:, j])
        return S

    def predict_vote(self, data):
        S = self.class_scores(data)
        pred = S.argmax(1)
        pred[S.max(1) == 0] = self.default_class
        return pred

    # ---- Mamdani inference with centroid defuzzification -----------------
    def mamdani_score(self, data, resolution=501):
        F = self.firing(self.fuzzify(data))
        z = np.linspace(0, 100, resolution)
        out = {c: _mf(k, p, z) for c, (k, p) in OUT_SETS.items()}
        agg = np.zeros((F.shape[0], resolution))
        for j, (_, c) in enumerate(self.rules):
            agg = np.maximum(agg, np.minimum(F[:, [j]], out[c][None, :]))
        den = agg.sum(1)
        return np.where(den > 0, (agg * z).sum(1) / np.where(den > 0, den, 1), 50.0)

    @staticmethod
    def score_to_class(score, cuts=SCORE_CUTS):
        s = np.asarray(score)
        return np.where(s <= cuts[0], 0, np.where(s <= cuts[1], 1, 2))

    def predict_mamdani(self, data):
        return self.score_to_class(self.mamdani_score(data))

    # ---- explanation -----------------------------------------------------
    def explain(self, data, i=0, top=6):
        one = {k: np.asarray(v)[[i]] for k, v in data.items()}
        F = self.firing(self.fuzzify(one))[0]
        order = np.argsort(-F)[:top]
        return [(rule_text(*self.rules[j]), float(F[j])) for j in order if F[j] > 0]

def rule_text(ant, c):
    cond = ' AND '.join(f'{v} is {lab}' for v, lab in ant.items()) if ant else 'ALWAYS'
    return f'IF {cond} THEN risk is {CLASS_NAMES[c]}'

def wang_mendel(system, data, y, mode='max'):
    # mode 'max': classic WM, keep the highest-degree rule per antecedent
    # mode 'sum': keep the class with the largest total degree per antecedent (less noisy)
    mu = system.fuzzify(data)
    vars_ = list(system.variables)
    labs = {v: list(system.variables[v]) for v in vars_}
    M = {v: np.stack([mu[(v, l)] for l in labs[v]], 1) for v in vars_}
    arg = {v: M[v].argmax(1) for v in vars_}
    degree = np.prod([M[v].max(1) for v in vars_], axis=0)
    table = {}
    for i in range(len(y)):
        key = tuple(int(arg[v][i]) for v in vars_)
        c = int(y[i])
        slot = table.setdefault(key, {})
        slot[c] = max(slot.get(c, 0.0), degree[i]) if mode == 'max' else slot.get(c, 0.0) + degree[i]
    rules = []
    for key, cd in table.items():
        c = max(cd, key=cd.get)
        rules.append(({v: labs[v][k] for v, k in zip(vars_, key)}, c))
    return rules
