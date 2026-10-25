#!/usr/bin/env python3
"""A pseudo NN friend for a dress rehearsal of the NN chain before a friend production exists (NN spec 3.4, 3.6): the
events of stored tau skims scored with installed models as the CROWN friend sm_nn_classifier scores them, written
into a copy of the skims that ShapeSmith reads as if the friend had been skimmed with them.

    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh   # onnxruntime, pyarrow
    python scripts/pseudo_nn_friend.py -c configs/smrun2_binned_v7b_ff_nn_m0_v1.yaml -s era=2018 -s skim_dir=COPY \\
        --channel tt --source /ceph/sdaigler/shapesmith/smrun2_binned_v7b/skims/tau_ff_nn/2018 \\
        --crown ../KingMaker/CROWN/analysis_configurations/bbtautau --payload-root DIR \\
        --model payloads/nn_classifier/<model>=nn_m0_ --model payloads/nn_classifier/<model k4>=nn_m0k4_ [--workers 16]

The run YAML (with its overrides) is the run that reads the copy: its skim_dir is the target and its NN friend under
ntuples.friends the friend the copy claims. Every `--model PAYLOAD=PREFIX` is one model of the friend production (the
MODELS of sm_nn_classifier.py): the model directory the trainer installs (bbtautau-nn `install --target`,
<payload>/<era>/<scope>/manifest.json) relative to --payload-root (default: --crown), read and checked by CROWN's
nn_payloads.read_payload, and the prefix of its columns; the run needs the columns of its switches nn_prefix and
nn_friend_prefixes. Per event, as the friend:
- the gate is CROWN's nn_gate.skim_cuts(scope, era) (pandas.eval of its pandas-syntax cuts); outside it every output
  is -1 (no skimmed event should be outside: the gate contains the skim selections);
- the inputs in the order of the manifest, cast to float32, its constants at their positions; a non-finite input
  inside the gate is an error (the friend throws);
- the model of residue `event % folds` on the unsigned 64-bit event number, one event per call ([1, N]), softmax only
  for a model that gives logits;
- `<prefix>predicted_class` (int32, the first largest of the class scores), `<prefix>predicted_max_value` and
  `<prefix>nn_score_<class>` (float32; `<prefix>binary_score` with a binary head).

The copy holds the samples of the channel of the run, every Parquet file of the source with the columns appended
(the others unchanged) and the source manifest with its contract extended by the friend bases of the run and the new
columns, plus a "pseudo_friend" record (payloads, sha256 of manifests and ONNX files, gate counts, residues,
onnxruntime version). The copy is then checked with ShapeSmith's own skim reuse check (shapesmith.skim.reuse_problems
against the contract of the run): `shapesmith skim` would reuse it, could it list the friend files of the production.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import logging
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapesmith.config import load_analysis, load_config
from shapesmith.expressions import columns_of, mask
from shapesmith.skim import contract, exported_by, needed_columns, reuse_problems
from shapesmith.store import SKIM_COLUMNS, manifest_path, read_manifest, schema, write_manifest

logger = logging.getLogger("pseudo_nn_friend")


def crown_modules(crown: Path):
    """CROWN bbtautau's nn_payloads and nn_gate (standard library only), imported from the configuration directory."""
    sys.path.append(str(Path(crown).resolve()))
    return importlib.import_module("nn_payloads"), importlib.import_module("nn_gate")


def output_columns(prefix: str, classes: tuple[str, ...], binary_head: bool) -> list[str]:
    """The friend's columns of one model (CROWN producers/sm_nn.output_names)."""
    return [f"{prefix}predicted_class", f"{prefix}predicted_max_value", *(f"{prefix}nn_score_{name}" for name in classes)] + ([f"{prefix}binary_score"] if binary_head else [])


def score(model: dict, x: np.ndarray, events: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """The outputs of the model for every event ([n, outputs], NaN outside the gate), one onnxruntime call per event."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = [onnxruntime.InferenceSession(file, options, providers=["CPUExecutionProvider"]) for file in model["files"]]
    names = [(s.get_inputs()[0].name, s.get_outputs()[0].name) for s in sessions]
    classes = len(model["classes"])
    result = np.full((len(x), sessions[0].get_outputs()[0].shape[1]), np.nan, dtype=np.float32)
    residues = events % np.uint64(len(sessions))
    for i in np.flatnonzero(gate):
        r = int(residues[i])
        values = sessions[r].run([names[r][1]], {names[r][0]: x[i: i + 1]})[0][0]
        if model["softmax"]:
            exp = np.exp(values[:classes] - values[:classes].max())
            values = np.concatenate([exp / exp.sum(), values[classes:]]).astype(np.float32)
        result[i] = values
    return result


def friend_columns(model: dict, frame, gate: np.ndarray) -> dict[str, np.ndarray]:
    """The columns of one model for the events of `frame`, -1 outside the gate."""
    n = len(frame)
    x = np.column_stack([np.full(n, model["constants"][name], dtype=np.float32) if name in model["constants"] else frame[name].to_numpy().astype(np.float32)
                         for name in model["inputs"]]) if n else np.zeros((0, len(model["inputs"])), dtype=np.float32)
    bad = gate & ~np.isfinite(x).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} events in the gate with a non-finite input of {model['payload']}")
    scores = score(model, x, frame["event"].to_numpy().astype(np.uint64), gate)
    classes = np.where(gate[:, None], scores[:, : len(model["classes"])], 0.0)
    names = model["columns"]
    columns = {names[0]: np.where(gate, np.argmax(classes, axis=1), -1).astype(np.int32), names[1]: np.where(gate, np.max(classes, axis=1), -1.0).astype(np.float32)}
    columns.update({name: np.where(gate, scores[:, k], -1.0).astype(np.float32) for k, name in enumerate(names[2:])})
    return columns


def score_file(job: dict) -> dict:
    """Score one Parquet file of the source with every model and write it with the friend columns to the target."""
    table = pq.read_table(job["source"])
    frame = table.select([c for c in table.column_names if c in job["read"]]).to_pandas()
    gate = mask(frame, job["gate"])
    for model in job["models"]:
        for name, values in friend_columns(model, frame, gate).items():
            table = table.append_column(name, pa.array(values))
    Path(job["target"]).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, job["target"], compression="snappy")
    folds = len(job["models"][0]["files"])
    residues = (frame["event"].to_numpy().astype(np.uint64) % np.uint64(folds)).astype(np.int64)
    return {"events": len(frame), "gate": int(gate.sum()), "residues": np.bincount(residues, minlength=folds).tolist()}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_models(specs: list[str], root: Path, era: str, channel: str, nn_payloads) -> list[dict]:
    """The models of `--model PAYLOAD=PREFIX`, read and checked by CROWN's read_payload."""
    models = []
    for spec in specs:
        payload_dir, _, prefix = spec.rpartition("=")
        payload = nn_payloads.read_payload(str(root), payload_dir, era, channel)
        manifest = Path(root) / payload_dir / era / channel / "manifest.json"
        files = [str(Path(root) / file) for file in payload.files]
        models.append({
            "payload": str(manifest.parent), "model": payload.model, "prefix": prefix, "inputs": list(payload.inputs), "constants": dict(payload.constants),
            "classes": list(payload.classes), "softmax": payload.softmax, "files": files, "columns": output_columns(prefix, payload.classes, payload.binary_head),
            "manifest_sha256": sha256(manifest), "onnx_sha256": [sha256(Path(file)) for file in files],
        })
        logger.info(f"model {payload.model} ({manifest}): {len(payload.inputs)} inputs, classes {', '.join(payload.classes)}, {payload.folds} folds, prefix {prefix}")
    if len({len(model["files"]) for model in models}) > 1:
        logger.info("the models have different fold counts; the residues of the record are those of the first")
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="the run YAML that reads the copy (its skim_dir is the target)")
    parser.add_argument("-s", "--set", dest="overrides", action="append", default=[], help="override of the run YAML (repeatable)")
    parser.add_argument("--channel", required=True, choices=("et", "mt", "tt"))
    parser.add_argument("--source", required=True, type=Path, help="the stored skims of the era (<source>/<channel>/<nick>/)")
    parser.add_argument("--crown", required=True, type=Path, help="CROWN analysis_configurations/bbtautau (nn_payloads, nn_gate)")
    parser.add_argument("--model", dest="models", action="append", required=True, help="PAYLOAD=PREFIX: a model of the friend (repeatable)")
    parser.add_argument("--payload-root", type=Path, help="the directory the payloads are relative to (default: --crown)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="replace an existing copy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import onnxruntime

    config = load_config(args.config, args.overrides)
    analysis = load_analysis(config)
    channel = analysis.channel(args.channel)
    nn_payloads, nn_gate = crown_modules(args.crown)
    models = read_models(args.models, args.payload_root or args.crown, config.era, args.channel, nn_payloads)
    gate_cuts = list(nn_gate.skim_cuts(args.channel, config.era).values())
    columns = [name for model in models for name in model["columns"]]
    record = {"script": "scripts/pseudo_nn_friend.py", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": str(args.source / args.channel),
              "models": [{key: model[key] for key in ("payload", "model", "prefix", "columns", "manifest_sha256", "onnx_sha256")} for model in models],
              "gate": gate_cuts, "onnxruntime": onnxruntime.__version__}
    read = {"event"} | columns_of(gate_cuts) | {name for model in models for name in model["inputs"] if name not in model["constants"]}
    jobs, manifests = [], {}
    for sample in channel.samples:
        source_dir, target_dir = args.source / args.channel / sample.nick, Path(config.skim_dir) / args.channel / sample.nick
        if target_dir.exists():
            if not args.force:
                raise FileExistsError(f"{target_dir} exists (--force replaces it)")
            shutil.rmtree(target_dir)
        manifests[sample.nick] = read_manifest(source_dir / "manifest.json")
        for name in manifests[sample.nick]["completed"]:
            stem = Path(name).stem
            jobs.append({"nick": sample.nick, "source": str(source_dir / f"{stem}.parquet"), "target": str(target_dir / f"{stem}.parquet"), "read": read, "gate": gate_cuts, "models": models})
    logger.info(f"{args.channel}: {len(manifests)} samples, {len(jobs)} files of {args.source / args.channel} -> {Path(config.skim_dir) / args.channel}, columns {', '.join(columns)}")
    counts: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:  # as ShapeSmith: no fork
        for job, result in zip(jobs, pool.map(score_file, jobs)):
            counts.setdefault(job["nick"], []).append(result)
    problems = []
    for sample in channel.samples:
        stored, results = manifests[sample.nick], counts.get(sample.nick, [])
        expected = contract(config, channel, sample, needed_columns(channel, sample, exported_by(analysis, channel)))
        extended = copy.deepcopy(stored)
        friends = stored["contract"].get("friends", [])
        extended["contract"]["friends"] = friends + [base for base in expected.get("friends", []) if base not in friends]
        extended["contract"]["required_columns"] = sorted(set(stored["contract"]["required_columns"]) | set(columns))
        extended["columns"] = sorted(set(stored["columns"]) | set(columns))
        extended["pseudo_friend"] = {**record, "events": sum(r["events"] for r in results), "gate_passed": sum(r["gate"] for r in results),
                                     "residues": np.sum([r["residues"] for r in results], axis=0).tolist() if results else []}
        write_manifest(manifest_path(config.skim_dir, args.channel, sample.nick), extended)
        stored_columns = schema(config.skim_dir, args.channel, [sample.nick]) - set(SKIM_COLUMNS) if results else set()
        found = reuse_problems(extended["contract"], expected, stored_columns)
        found += [f"{name} missing" for name in extended["completed"] if not (Path(config.skim_dir) / args.channel / sample.nick / f"{Path(name).stem}.parquet").exists()]
        if found:
            problems.append(f"{sample.nick}: {', '.join(found)}")
        entry = extended["pseudo_friend"]
        logger.info(f"{sample.nick}: {entry['events']} events, {entry['events'] - entry['gate_passed']} outside the gate, residues {entry['residues']}")
    versions = args.source / "versions"
    if versions.is_dir():
        shutil.copytree(versions, Path(config.skim_dir) / "versions", dirs_exist_ok=True)
    if problems:
        logger.error("the copy would not be reused by `shapesmith skim`:\n" + "\n".join(problems))
        return 1
    logger.info(f"{len(manifests)} samples: the copy passes the skim reuse check of the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
