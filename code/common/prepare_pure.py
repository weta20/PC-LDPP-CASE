"""Lambda-aware point-only preparation; one decompression per cached array."""

import time
from pathlib import Path

import numpy as np

from . import pure_model as model
from .io import BASE, atomic, digest, load, read_points, sha, tag


def prepare(case, c, lam, source=None, cache_root=None):
    start = time.perf_counter()
    source = Path(source) if source else BASE / "data/points" / f"{case}.txt"
    opts = c["pure"]
    key = digest(
        dict(
            input=sha(source),
            turn_weight=lam,
            options={k: opts[k] for k in ["max_blocks", "target_holes", "max_states"]},
            source=sha(Path(__file__).with_name("pure_model.py")),
            prepare_source=sha(Path(__file__)),
            dll=sha(BASE / "code/common/bin/point_dp.dll"),
        )
    )[:20]
    cache_root = Path(cache_root) if cache_root is not None else BASE / "result/cache"
    folder = cache_root / f"{case}_lambda{tag(lam)}_{key}"
    meta = folder / "metadata.json"
    hit = meta.exists()
    points = read_points(source)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or len(np.unique(points, axis=0)) != len(points)
    ):
        raise ValueError("Invalid/duplicate points")
    if not hit:
        model.MAX_BLOCKS = opts["max_blocks"]
        model.TARGET_HOLES = opts["target_holes"]
        model.MAX_STATES = opts["max_states"]
        blocks = model.partition(points)
        states, sets, stats = model.candidates(points, blocks, lam)
        cost, constant, scale, initial, bound = model.global_cost(points, states, sets, lam)
        folder.mkdir(parents=True, exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(s["order"]) for s in states])]
        co = np.r_[0, np.cumsum([len(s["choices"]) for s in states])]
        np.savez_compressed(
            folder / "data.npz",
            cost=cost,
            state_offsets=offsets,
            state_ids=np.concatenate([s["order"] for s in states]),
            choice_offsets=co,
            choices=np.concatenate([s["choices"] for s in states]),
        )
        with (folder / "instance.gtsp").open("w", encoding="ascii", newline="\n") as f:
            f.write(
                f"NAME : {case}_lambda{tag(lam)}\nTYPE : AGTSP\nDIMENSION : {len(states)}\nGTSP_SETS : {len(sets)}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n"
            )
            for row in cost:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("GTSP_SET_SECTION\n")
            for i, group in enumerate(sets):
                f.write(f"{i+1} " + " ".join(str(j + 1) for j in group) + " -1\n")
            f.write("EOF\n")
        (folder / "initial.tour").write_text(
            f"NAME : {case}\nTYPE : TOUR\nDIMENSION : {len(sets)}\nTOUR_SECTION\n"
            + "".join(f"{i+1}\n" for i in initial)
            + "-1\nEOF\n",
            encoding="ascii",
        )
        atomic(
            meta,
            dict(
                case=case,
                key=key,
                input_sha256=sha(source),
                turn_weight=lam,
                distance_weight=1,
                holes=len(points),
                sets=sets,
                statistics=stats,
                constant=constant,
                scale=scale,
                initial=initial,
                bound=bound,
                prepare_s=time.perf_counter() - start,
                states=[
                    {k: v for k, v in s.items() if k not in ["order", "choices"]} for s in states
                ],
                files={n: sha(folder / n) for n in ["data.npz", "instance.gtsp", "initial.tour"]},
            ),
        )
    metadata = load(meta)
    if metadata["turn_weight"] != lam or metadata["input_sha256"] != sha(source):
        raise ValueError("Cache objective/input mismatch")
    for n, h in metadata["files"].items():
        if sha(folder / n) != h:
            raise ValueError("Corrupt cache: " + n)
    with np.load(folder / "data.npz", allow_pickle=False) as z:
        arrays = {n: z[n].copy() for n in z.files}
    states = []
    offsets = arrays["state_offsets"]
    co = arrays["choice_offsets"]
    for i, s in enumerate(metadata["states"]):
        if abs(s["local_cost"] - s["distance"] - lam * s["turns"]) > 1e-7:
            raise ValueError("Unweighted local cost")
        states.append(
            dict(
                **s,
                order=arrays["state_ids"][offsets[i] : offsets[i + 1]],
                choices=arrays["choices"][co[i] : co[i + 1]],
            )
        )
    metadata.update(
        cache_hit=hit,
        load_or_prepare_s=time.perf_counter() - start,
        folder=str(folder.relative_to(BASE)).replace("\\", "/"),
    )
    return points, states, arrays["cost"], metadata
