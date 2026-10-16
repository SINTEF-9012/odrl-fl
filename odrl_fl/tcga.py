"""TCGA-BRCA: real patients, real institutions, real terms of use.

Everything measured so far ran on synthetic data partitioned by a Dirichlet
draw, which leaves two things unverified: whether the utility proxy tracks real
model quality, and whether a federation partitioned the way real institutions
actually divide is anything like the one the sweeps assumed. This module
replaces the first half of that.

**What is real here.** 1096 breast-cancer patients from The Cancer Genome Atlas,
39 clinical covariates, and a survival outcome `(event, time)`. Crucially the
partition is real too: every patient id carries a Tissue Source Site code, and
`centers.csv` maps those to **40 named institutions** -- University of
Pittsburgh, Walter Reed, Roswell Park, Mayo, Greater Poland Cancer Center. The
federation is not a partition invented to look federated; it is the one the data
already had, with the volume imbalance real institutions actually have (150
patients down to one).

The task is the one the data supports: a linear Cox proportional-hazards model
scored by Harrell's concordance index, following Andreux et al. (2020) and the
FLamby benchmark. 13.8% of patients have an observed event, so this is heavily
censored -- accuracy would be meaningless and C-index is the standard answer.

**What is still modelled.** The data-use terms. This is the *open-access* GDC
clinical tier: FLamby states plainly that it uses unrestricted data only and
excludes the genetic component precisely because that part is controlled. So
there are no dbGaP Data Use Limitations attached to these rows, and no DUO codes
to cite for them -- a premise this project's README got wrong until this
increment checked it.

What can be cited is the GDC open-access policy itself, and it is thin: no
purpose restriction, and one prohibition, on attempting to re-identify
participants. `gdc_open_access_policy` expresses exactly that and nothing more.
Per-institution variation still has to be modelled, and `sited_corpus` does so
explicitly rather than pretending otherwise.

That thinness is itself the finding. Permission erosion needs heterogeneous
terms, and the tier of data anyone can download has almost none -- which is why
the phenomenon lives in the controlled-access tier, and why an honest study of
it cannot be run on open data alone.

**Terms.** Data from the GDC (https://portal.gdc.cancer.gov/). Users must not
attempt to identify individual participants. See
https://gdc.cancer.gov/about-gdc/gdc-policies. Nothing here attempts
re-identification: the covariates are as FLamby preprocessed them and no
linkage to any other source is performed.
"""

from __future__ import annotations

import csv
import urllib.request
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

#: Where FLamby publishes the preprocessed cohort. Two small CSVs, ~117 KB.
SOURCE = (
    "https://raw.githubusercontent.com/owkin/FLamby/main/"
    "flamby/datasets/fed_tcga_brca/"
)
FILES = ("brca.csv", "centers.csv")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tcga_brca"

#: 39 covariates between the patient id and the (E, T) outcome columns.
N_FEATURES = 39


def fetch(directory: Path | str = DATA_DIR, *, force: bool = False) -> Path:
    """Download the cohort if it is not already present.

    The data is gitignored rather than vendored: it belongs to the GDC, and a
    copy in this repository would be a redistribution nobody asked for.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name
        if target.exists() and not force:
            continue
        with urllib.request.urlopen(SOURCE + name, timeout=60) as response:
            target.write_bytes(response.read())
    return out


@dataclass(frozen=True)
class Site:
    """One real institution's cohort."""

    name: str
    tss_codes: tuple[str, ...]
    region: str
    country: str
    x: np.ndarray
    y: np.ndarray  # (event, time)

    @property
    def n_patients(self) -> int:
        return len(self.y)

    @property
    def n_events(self) -> int:
        return int(self.y[:, 0].sum())


def _centers(directory: Path) -> dict[str, dict[str, str]]:
    with (directory / "centers.csv").open(encoding="utf-8") as fh:
        return {row["TSS Code"].strip(): row for row in csv.DictReader(fh)}


def load_sites(
    directory: Path | str = DATA_DIR, *, min_patients: int = 1
) -> list[Site]:
    """Every source site, largest first.

    `min_patients` drops sites too small to train on at all. Filtering by size
    is a modelling choice with teeth -- the smallest sites hold a single patient
    -- so it is a parameter rather than a constant buried in the body.
    """
    directory = fetch(directory)
    centers = _centers(directory)

    with (directory / "brca.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    columns = list(rows[0])
    feature_names = columns[1 : 1 + N_FEATURES]

    grouped: dict[str, list[dict]] = {}
    codes: dict[str, set[str]] = {}
    for row in rows:
        tss = row["pid"].split("-")[1]
        center = centers.get(tss)
        if center is None:  # pragma: no cover - every code maps today
            continue
        name = center["Source Site"].strip()
        grouped.setdefault(name, []).append(row)
        codes.setdefault(name, set()).add(tss)

    sites: list[Site] = []
    for name, patients in grouped.items():
        if len(patients) < min_patients:
            continue
        first = centers[next(iter(codes[name]))]
        x = np.array(
            [[float(p[c]) for c in feature_names] for p in patients],
            dtype=np.float32,
        )
        y = np.array(
            [[float(p["E"]), float(p["T"])] for p in patients], dtype=np.float32
        )
        sites.append(
            Site(
                name=name,
                tss_codes=tuple(sorted(codes[name])),
                region=first["Region"].strip(),
                country=first["Country"].strip(),
                x=x,
                y=y,
            )
        )

    return sorted(sites, key=lambda s: (-s.n_patients, s.name))


def standardise(sites: Sequence[Site]) -> list[Site]:
    """Centre and scale features using statistics pooled across sites.

    A shared scaling is a simplification: computing it centrally means somebody
    saw every site's feature distribution, which a real federation would have to
    obtain by secure aggregation. Noted rather than solved, and it affects only
    conditioning, not which contributor is permitted to do what.
    """
    pooled = np.concatenate([s.x for s in sites])
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    std[std == 0] = 1.0
    return [
        Site(
            name=s.name,
            tss_codes=s.tss_codes,
            region=s.region,
            country=s.country,
            x=((s.x - mean) / std).astype(np.float32),
            y=s.y,
        )
        for s in sites
    ]


class CoxModel(nn.Module):
    """Linear proportional hazards. One risk score per patient."""

    def __init__(self, n_features: int = N_FEATURES) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def cox_loss(risk: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood, Breslow tie handling.

    Each patient with an observed event contributes the log-odds of that event
    happening to *them* rather than to anyone still at risk at that moment.
    Censored patients contribute only to the risk sets of others, which is the
    whole point of survival analysis: dropping them would discard 86% of this
    cohort.
    """
    event, time = y[:, 0], y[:, 1]
    order = torch.argsort(time, descending=True)
    risk, event, time = risk[order], event[order], time[order]

    # Descending time means the cumulative sum over the prefix is exactly the
    # risk set: everyone whose time is at least this patient's.
    log_risk_sum = torch.logcumsumexp(risk, dim=0)
    observed = event > 0
    if not observed.any():
        return risk.sum() * 0.0
    return -(risk[observed] - log_risk_sum[observed]).mean()


def c_index(risk: np.ndarray, y: np.ndarray) -> float:
    """Harrell's concordance index. 0.5 is a coin flip, 1.0 is perfect.

    The standard metric for censored survival data, and the one the FLamby
    benchmark reports, so a number here is comparable with published work.
    A pair is comparable when the one that failed first is known to have done
    so; a higher risk score should go with the shorter survival.
    """
    event, time = y[:, 0], y[:, 1]
    concordant = permissible = 0.0

    for i in range(len(time)):
        if not event[i]:
            continue
        for j in range(len(time)):
            if time[j] <= time[i]:
                continue
            permissible += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5

    return concordant / permissible if permissible else 0.5


def train_cox(
    model: CoxModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 0.05,
) -> float:
    """Fit on one site's cohort. Full batch -- these are hundreds of rows."""
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    features = torch.from_numpy(x)
    outcome = torch.from_numpy(y)

    loss = torch.tensor(0.0)
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = cox_loss(model(features), outcome)
        loss.backward()
        optimiser.step()
    return float(loss.item())


def evaluate_cox(model: CoxModel, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        risk = model(torch.from_numpy(x)).numpy()
    return c_index(risk, y)


def pooled(sites: Iterable[Site]) -> tuple[np.ndarray, np.ndarray]:
    sites = list(sites)
    return (
        np.concatenate([s.x for s in sites]),
        np.concatenate([s.y for s in sites]),
    )


def _site_seed(name: str) -> int:
    """A per-site seed that does not depend on which process asks for it.

    The obvious `hash(name)` is wrong here, and was used for three increments
    before anyone noticed. Python randomises string hashing per interpreter
    under `PYTHONHASHSEED`, so a split derived from it is reproducible *within*
    a run and never *between* runs -- the same `seed=0` yields a different
    train/test fold in every process. That made one test intermittently fail,
    which is how it surfaced, but the cost that mattered was silent: every
    committed TCGA figure came from a fold nobody could reconstruct.

    CRC32 is not a cryptographic hash and does not need to be. It needs to be
    the same number everywhere, forever, which the builtin is not.
    """
    return zlib.crc32(name.encode("utf-8"))


def split(
    site: Site, *, holdout: float = 0.25, seed: int = 0
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Train/test split within one site, stratified on the event indicator.

    Stratified because events are rare: an unstratified split of a small site
    can leave a test fold with no events at all, and a C-index over no
    comparable pairs is undefined rather than merely noisy.
    """
    rng = np.random.default_rng([seed, _site_seed(site.name)])
    train_idx: list[int] = []
    test_idx: list[int] = []

    for label in (0.0, 1.0):
        members = np.where(site.y[:, 0] == label)[0]
        rng.shuffle(members)
        cut = round(len(members) * (1 - holdout))
        # Guarantee the stratum actually appears on both sides when it can.
        # Rounding alone does not: a site with exactly two events gives
        # round(1.5) == 2, putting both in train and leaving the test fold with
        # no events -- so the split would silently stop being stratified for
        # precisely the small sites it exists to protect.
        if len(members) >= 2:
            cut = min(max(cut, 1), len(members) - 1)
        train_idx.extend(members[:cut].tolist())
        test_idx.extend(members[cut:].tolist())

    return (
        (site.x[train_idx], site.y[train_idx]),
        (site.x[test_idx], site.y[test_idx]),
    )


# --- The real terms attached to this data ----------------------------------


def gdc_open_access_policy(
    *, uid: str = "urn:policy:gdc-open-access", assigner: str | None = None
):
    """The GDC open-access terms, as ODRL. Real and citable, and very thin.

    Everything the tier actually says: use it for anything, and do not try to
    re-identify a participant. No purpose limitation, no expiry, no duty --
    because this is the tier anyone may download.

    Expressing the one prohibition needs `identify`, which neither ODRL core nor
    DUO defines. It is a profile extension here, and the gap is worth naming:
    the single enforceable term on the most widely used open genomics resource
    has no standard vocabulary term. DUO has the same problem from the other
    direction with COL, which this project already records as out-of-fragment.
    """
    from odrl_fl.odrl import (
        Action,
        Permission,
        Policy,
        PolicyType,
        Prohibition,
    )

    return Policy(
        uid=uid,
        type=PolicyType.OFFER,
        assigner=assigner,
        target="urn:asset:tcga-brca-clinical",
        permissions=frozenset(
            Permission(action=a)
            for a in (
                Action.USE,
                Action.AGGREGATE,
                Action.DERIVE,
                Action.DISTRIBUTE,
            )
        ),
        prohibitions=frozenset({Prohibition(action=Action.IDENTIFY)}),
    )
