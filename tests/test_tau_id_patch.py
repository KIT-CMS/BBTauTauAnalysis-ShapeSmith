"""The patched HttSystematics_TauIDRun2.cc (patches/TauIDSFMeasurement-tauID_ES.patch, which carries the file with full
context) against the tau-ID measurement of every era: MorphingTauID2017 throws on a declared shape that the synced
shapes lack (CombineTools TFileIO.cc: "TH1 ... not found"), so every shape it declares for a process of the measurement
must be one the analysis produces. v7: no HEM shape (E29), CMS_prefiring in every era (E11), the lumi lnN the total of
the Cholesky scheme of the year (6.6); v7b: the recoil shapes of DY and W in 2018 only (the MET fix of 2026-10-18).

The file is read by a small interpreter of its few constructs: if / else if / else on the arguments of
AddTauIDRun2Systematics, range-for over string arrays, string list assignments and the cb.cp()...AddSyst(...)
statements."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pytest
from shapesmith.histogram import is_part, is_template
from shapesmith.measurements.tau_id_es.grid import CATEGORIES
from shapesmith.model import TemplateShift, VariationSum
from shapesmith.validate import validate

from bbtautau_shapesmith import tau_id_measurement
from bbtautau_shapesmith.constants import YEAR
from bbtautau_shapesmith.systematics import LUMI
from bbtautau_shapesmith.tau_id_systematics import ERA_TAG, unused_shifts
from tests.helpers import ERAS, REPO, config
from tests.run2 import list_name

PATCH = REPO / "patches" / "TauIDSFMeasurement-tauID_ES.patch"
SYSTEMATICS = "src/HttSystematics_TauIDRun2.cc"
# the arguments of the measurement (shapesmith.measurements.tau_id_es.combine.morphing_command: --jetfakes=0
# --embedding=1); regional_jec and tauES keep their defaults
FLAGS = {"jetfakes": False, "embedding": True, "regional_jec": False, "tauES": False}
# the processes of MorphingTauID2017.cpp with embedding and without jet fakes: bkgs without the genuine taus plus the
# embedded signal of the category; bkgs_mm of the control region
PROCESSES = {
    "mt": {"W", "QCD", "ZL", "ZJ", "TTL", "TTJ", "STL", "STJ", "VVJ", "VVL"} | {f"EMB_{category}" for category in CATEGORIES},
    "mm": {"W", "TTL", "VVL", "QCD", "MUEMB"},
}
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|.', re.S)
STRING = re.compile(r'"((?:\\.|[^"\\])*)"')


def post_image(patch_text: str, path: str) -> str:
    """The file after the patch; its diff must be one hunk over the whole file."""
    section = patch_text.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("\ndiff --git ", 1)[0]
    hunks = [line for line in section.splitlines() if line.startswith("@@ ")]
    assert len(hunks) == 1 and re.match(r"@@ -1,\d+ \+1,(\d+) @@", hunks[0]), f"{path}: the patch must carry the whole file (git diff -U100000)"
    lines = section.split(hunks[0], 1)[1].splitlines()[1:]
    image = [line[1:] for line in lines if line[:1] in (" ", "+")]
    assert len(image) == int(re.match(r"@@ -1,\d+ \+1,(\d+) @@", hunks[0]).group(1))
    return "\n".join(image) + "\n"


def events(code: str) -> list[tuple[str, str]]:
    """("open", header), ("close", "") and ("statement", text) of the code without comments and preprocessor lines; a
    brace opens a block after a closing parenthesis, `else` or `namespace <name>`, otherwise it belongs to an
    initializer of the statement."""
    code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#"))
    result, buffer, depth = [], "", 0
    for token in TOKEN.findall(code):
        if token.startswith(("//", "/*")):
            continue
        header = " ".join(buffer.split())
        if token == "{" and depth == 0 and (header.endswith(")") or header.endswith("else") or re.fullmatch(r"namespace \w+", header)):
            result.append(("open", header))
            buffer = ""
        elif token == "}" and depth == 0:
            assert not header, f"unfinished statement {header!r}"
            result.append(("close", ""))
        elif token == ";" and depth == 0:
            result.append(("statement", header))
            buffer = ""
        else:
            depth += token == "{"
            depth -= token == "}"
            buffer += token
    assert not " ".join(buffer.split()) and depth == 0
    return result


@dataclass(frozen=True)
class Syst:
    name: str
    kind: str
    channels: frozenset[str]
    processes: frozenset[str]
    value: float | None


def _condition(text: str, era_tag: str) -> bool:
    expression = text.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return bool(eval(expression, {"__builtins__": {}}, {**FLAGS, "era_tag": era_tag, "true": True, "false": False}))  # noqa: S307 - the patch's own conditions


def declarations(code: str, era_tag: str) -> list[Syst]:
    """The AddSyst calls that MorphingTauID2017 runs for the era tag."""
    stream = events(code)
    arrays: dict[str, list[str]] = {}
    variables: dict[str, list[str] | float] = {}
    found: list[Syst] = []

    def value_of(expression: str, bindings: dict[str, str]) -> str:
        parts = [part.strip() for part in expression.split("+")]
        return "".join(STRING.fullmatch(part).group(1) if STRING.fullmatch(part) else bindings[part] for part in parts)

    def statement(text: str, bindings: dict[str, str]) -> None:
        if "AddSyst" in text:
            match = re.search(r"\.AddSyst\(cb, (.+?), \"(\w+)\", SystMap<>::init\(([\w.]+)\)\)$", text)
            assert match, f"unparsed AddSyst: {text}"
            channels = frozenset(STRING.findall(re.search(r"\.channel\((\{.*?\})\)", text).group(1)))
            process = re.search(r"\.process\((.+?)\)\s*\.AddSyst", text).group(1).strip()
            processes = frozenset(STRING.findall(process)) if process.startswith("{") else frozenset(variables[process])
            raw = match.group(3)
            value = float(raw) if re.fullmatch(r"[\d.]+", raw) else variables[raw]
            found.append(Syst(value_of(match.group(1), bindings), match.group(2), channels, processes, value))
        elif match := re.fullmatch(r"std::string (\w+)\[\d+\] = (\{.*\})", text):
            arrays[match.group(1)] = STRING.findall(match.group(2))
        elif match := re.fullmatch(r"(?:std::vector<std::string> )?(\w+) = (\{.*\})", text):
            variables[match.group(1)] = STRING.findall(match.group(2))
        elif match := re.fullmatch(r"(?:float )?(\w+) = ([\d.]+)", text):
            variables[match.group(1)] = float(match.group(2))

    def run(index: int, active: bool, bindings: dict[str, str]) -> int:
        """Runs the block that starts after `index` (its open event) up to its close; returns the close's index."""
        taken = None  # the if / else chain of this level: whether a branch ran
        i = index + 1
        while stream[i][0] != "close":
            kind, text = stream[i]
            if kind == "statement":
                if active:
                    statement(text, bindings)
                taken = None
                i += 1
                continue
            if loop := re.fullmatch(r"for \(auto (\w+) : (\w+)\)", text):
                end = i
                for item in arrays[loop.group(2)] if active else [None]:
                    end = run(i, active and item is not None, {**bindings, loop.group(1): item})
                i, taken = end + 1, None
                continue
            if condition := re.fullmatch(r"if \((.*)\)", text):
                branch = _condition(condition.group(1), era_tag)
                taken = branch
            elif condition := re.fullmatch(r"else if \((.*)\)", text):
                assert taken is not None, text
                branch = not taken and _condition(condition.group(1), era_tag)
                taken = taken or branch
            elif text == "else":
                assert taken is not None, text
                branch, taken = not taken, True
            else:  # namespace, function
                branch, taken = True, None
            i = run(i, active and branch, bindings) + 1
        return i

    stream.insert(0, ("open", "file"))
    stream.append(("close", ""))
    run(0, True, {})
    return found


@pytest.fixture(scope="module")
def code():
    return post_image(PATCH.read_text(), SYSTEMATICS)


def shapes(code: str, era: str, channel: str) -> set[str]:
    """The shape names the patched file declares for a process of the measurement in the channel."""
    tag = ERA_TAG[era]
    return {s.name.replace("$ERA", tag).replace("$CHANNEL", channel) for s in declarations(code, tag)
            if s.kind == "shape" and channel in s.channels and s.processes & PROCESSES[channel]}


def produced(era: str) -> dict[str, set[str]]:
    """channel -> the shape names the tau-ID measurement of the era writes: its variations (no ES grid point, no part)
    and the outputs of its template shifts and summed variations."""
    analysis = tau_id_measurement.build(config(("mt", "mm"), "bbtautau_shapesmith.tau_id_measurement:build", era=era, sample_lists=[list_name("tau_id", era)]))
    validate(analysis)
    result = {}
    for name, channel in analysis.channels.items():
        names = {v.name for v in channel.variations if not is_part(v.name) and not is_template(v.name)}
        names |= {f"{e.name}{d}" for e in channel.estimators if isinstance(e, (TemplateShift, VariationSum)) for d in ("Up", "Down")}
        assert all(n.endswith(("Up", "Down")) for n in names)
        result[name] = {n.removesuffix("Up").removesuffix("Down") for n in names}
    return result


@pytest.mark.parametrize("era", ERAS)
def test_every_declared_shape_is_produced(code, era):
    made = produced(era)
    declared = shapes(code, era, "mt")
    assert declared and not declared - made["mt"], sorted(declared - made["mt"])
    assert not shapes(code, era, "mm") - made["mm"]  # the control region: normalisations only
    assert "CMS_prefiring" in declared and not [name for name in declared if "HEM" in name]  # E11, E29
    assert f"CMS_emb_ttbar_contamination_{ERA_TAG[era]}" in declared and f"CMS_scale_t_dm1_{ERA_TAG[era]}" in declared
    # the MET fix (2026-10-18): the recoil calibration of DY and W in 2018 only
    recoil = {f"CMS_scale_met_{ERA_TAG[era]}", f"CMS_res_met_{ERA_TAG[era]}"}
    assert (recoil <= declared) if era == "2018" else not recoil & declared
    # the other way round: only the shifts without a shape in MorphingTauID2017 (filled for the skim) and the
    # predecessor's undeclared CMS_PileUp stay out of the fit
    assert made["mt"] - declared == {name for name, _ in unused_shifts(era)} | {"CMS_PileUp"}


@pytest.mark.parametrize("era", ERAS)
def test_the_lumi_is_the_total_of_the_cholesky_scheme(code, era):
    tag = ERA_TAG[era]
    (lumi,) = [s for s in declarations(code, tag) if s.name.startswith("lumi_")]
    total = math.sqrt(sum((values[YEAR[era]] - 1.0) ** 2 for values in LUMI.values() if YEAR[era] in values))
    assert (lumi.name, lumi.kind) == ("lumi_13TeV_$ERA", "lnN") and lumi.value == pytest.approx(1.0 + total, abs=5e-5)


def test_the_interpreter_follows_the_branches():
    code = """
    void f(bool embedding, std::string era_tag) {
      std::vector<std::string> mc;
      if (embedding){ mc = {"A"}; } else { mc = {"B"}; }
      float x = 1.0;
      if (era_tag == "Run2017") { x = 1.5; } else if (era_tag == "Run2018") { x = 2.5; }
      std::string bins[2] = {"dm0", "dm1"};
      for (auto b : bins){ cb.cp().channel({"mt"}).process(mc).AddSyst(cb, "s_"+b+"_$ERA", "shape", SystMap<>::init(1.00)); }
      if (era_tag != "Run2018") {
      cb.cp().channel({"mt", "mm"}).process({{"C"}}).AddSyst(cb, "n", "lnN", SystMap<>::init(x));
      }
    }
    """
    syst = declarations(code, "Run2017")
    assert [(s.name, s.processes, s.value) for s in syst] == [("s_dm0_$ERA", {"A"}, 1.0), ("s_dm1_$ERA", {"A"}, 1.0), ("n", {"C"}, 1.5)]
    assert [s.name for s in declarations(code, "Run2018")] == ["s_dm0_$ERA", "s_dm1_$ERA"]
