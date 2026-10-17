"""A choice model that learns from the visit data.

Idea in plain words:
- Every option on the shelf (4 cleaners + "buy nothing") gets a score.
- The score is a sum of small effects: "this is the entrant", "the price is $X",
  "this is the household's usual brand", and so on.
- Scores turn into chances with softmax (same as the starter).
- Unlike the starter, the size of each effect is LEARNED from the build visits.

Which effects are used is picked with `groups`, so we can add one idea at a
time and check whether it really helps.

Only the Python standard library is used.
"""

import math

PRODUCTS = ["entrant", "standard", "budget", "scented"]
MESSAGES = ["cleaning", "environment", "low_odor"]  # blank message is the base case

GOOD_NOTES = {"It removed the stain I noticed.", "The smell was easy to live with.", "The bottle was easy to use."}
BAD_NOTES = {"It left a stain behind.", "The smell lasted too long.", "The bottle was hard to use."}


def option_features(option, price, message, household, visit, history, groups):
    """Return {feature name: value} for one option on one visit."""
    f = {}
    is_entrant = option == "entrant"

    if "brand" in groups:  # a fixed pull for each product, "none" is the zero point
        if option in PRODUCTS:
            f[f"is_{option}"] = 1.0

    if "price" in groups and option != "none":
        f["price"] = price

    if "household" in groups and option != "none":
        if household.get("usual_brand") == option:
            f["usual_brand"] = 1.0
        if household.get("budget") == "tight":
            f["price_if_tight_budget"] = price
        if household.get("budget") == "flexible":
            f["price_if_flexible_budget"] = price

    if "levers" in groups and is_entrant:
        if message in MESSAGES:
            f[f"entrant_msg_{message}"] = 1.0
        if visit.get("channel") == "online":
            f["entrant_online"] = 1.0

    if "history" in groups and history:
        last = history[-1]["choice"]
        if option != "none" and option == last:
            f["same_as_last_visit"] = 1.0
        if option == "none" and last != "none":
            f["none_after_buying_last_visit"] = 1.0
        if is_entrant:
            entrant_visits = [h for h in history if h["choice"] == "entrant"]
            if entrant_visits:
                f["entrant_bought_before"] = 1.0
            notes = {h.get("after_use_note", "") for h in entrant_visits}
            if notes & BAD_NOTES:
                f["entrant_bad_note_before"] = 1.0
            if notes & GOOD_NOTES:
                f["entrant_good_note_before"] = 1.0
    return f


def options_for(offers):
    """(option, price, message) for each shelf item plus 'none'."""
    out = [(o["product_id"], float(o["price"]), o.get("message", "")) for o in offers]
    return out + [("none", 0.0, "")]


def softmax(scores):
    top = max(scores)
    w = [math.exp(s - top) for s in scores]
    total = sum(w)
    return [x / total for x in w]


def solve(matrix, vector):
    """Solve A x = b with Gaussian elimination (A is small and square)."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def invert(matrix):
    n = len(matrix)
    columns = [solve(matrix, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


class ChoiceModel:
    def __init__(self, groups=("brand", "price"), ridge=0.01):
        self.groups = set(groups)
        self.ridge = ridge  # a small pull toward 0 so rare effects do not blow up
        self.weights = {}
        self.std_errors = {}
        self.covariance = {}

    # ----- turning visits into numbers -----
    def _rows(self, offers, household, visit, history):
        return [(name, option_features(name, price, msg, household, visit, history, self.groups))
                for name, price, msg in options_for(offers)]

    def _examples(self, visits, households, offers_for):
        """One example per visit: features for each option, and which one was chosen.

        History only holds EARLIER visits of the same household, so the model never sees the future.
        """
        history, examples = {}, []
        for v in visits:
            hid = v["household_id"]
            past = history.setdefault(hid, [])
            rows = self._rows(offers_for(v), households[hid], v, past)
            chosen = [name for name, _ in rows].index(v["choice"])
            examples.append(([f for _, f in rows], chosen))
            past.append(v)
        return examples

    # ----- learning -----
    def fit(self, visits, households, offers_for, max_steps=50):
        examples = self._examples(visits, households, offers_for)
        names = sorted({k for feats, _ in examples for f in feats for k in f})
        k = len(names)
        X = [[[f.get(n, 0.0) for n in names] for f in feats] for feats, _ in examples]
        y = [c for _, c in examples]
        w = [0.0] * k

        def loss_grad_hess(w):
            loss, grad = 0.0, [0.0] * k
            hess = [[0.0] * k for _ in range(k)]
            for options, chosen in zip(X, y):
                p = softmax([sum(a * b for a, b in zip(w, x)) for x in options])
                loss -= math.log(p[chosen])
                mean = [sum(p[j] * options[j][i] for j in range(len(options))) for i in range(k)]
                for i in range(k):
                    grad[i] -= options[chosen][i] - mean[i]
                for j, x in enumerate(options):
                    d = [x[i] - mean[i] for i in range(k)]
                    for i in range(k):
                        if d[i]:
                            row = hess[i]
                            for m in range(k):
                                row[m] += p[j] * d[i] * d[m]
            for i in range(k):  # ridge penalty
                loss += 0.5 * self.ridge * w[i] ** 2
                grad[i] += self.ridge * w[i]
                hess[i][i] += self.ridge
            return loss, grad, hess

        loss, grad, hess = loss_grad_hess(w)
        for _ in range(max_steps):  # Newton steps, halve the step if the loss goes up
            step = solve(hess, grad)
            size = 1.0
            while True:
                new_w = [a - size * b for a, b in zip(w, step)]
                new_loss, new_grad, new_hess = loss_grad_hess(new_w)
                if new_loss <= loss or size < 1e-4:
                    break
                size /= 2
            done = loss - new_loss < 1e-8
            w, loss, grad, hess = new_w, new_loss, new_grad, new_hess
            if done:
                break

        cov = invert(hess)
        self.weights = dict(zip(names, w))
        self.std_errors = {n: math.sqrt(max(cov[i][i], 0.0)) for i, n in enumerate(names)}
        self.covariance = {(a, b): cov[i][j] for i, a in enumerate(names) for j, b in enumerate(names)}
        self.train_log_loss = (loss - 0.5 * self.ridge * sum(x * x for x in w)) / len(y)
        return self

    # ----- saving and loading -----
    def to_dict(self):
        return {"groups": sorted(self.groups), "ridge": self.ridge,
                "weights": self.weights, "std_errors": self.std_errors}

    @classmethod
    def from_dict(cls, d):
        model = cls(d["groups"], d["ridge"])
        model.weights = dict(d["weights"])
        model.std_errors = dict(d.get("std_errors", {}))
        return model

    def score_parts(self, household, offers, visit, history):
        """For each option: {feature: weight * value}. Used to explain a prediction."""
        rows = self._rows(offers, household, visit, history)
        return {name: {n: self.weights.get(n, 0.0) * v for n, v in f.items()} for name, f in rows}

    # ----- predicting -----
    def predict(self, household, offers, visit, history):
        """Same shape as the starter: chances for each option, plus a short reason."""
        rows = self._rows(offers, household, visit, history)
        scores = [sum(self.weights.get(n, 0.0) * v for n, v in f.items()) for _, f in rows]
        probs = dict(zip([name for name, _ in rows], softmax(scores)))
        top = max(probs, key=probs.get)
        top_feats = dict(rows)[top]
        parts = sorted(((self.weights.get(n, 0.0) * v, n) for n, v in top_feats.items()), reverse=True)
        reason = f"Top pick '{top}'. Biggest pushes: " + ", ".join(f"{n} ({c:+.2f})" for c, n in parts[:3])
        return {"probabilities": probs, "reason": reason}
