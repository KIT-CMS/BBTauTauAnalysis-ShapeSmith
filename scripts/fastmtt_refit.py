"""FastMTT of CROWN core on numpy arrays (NN input validation NV3, nn_validation.py): the sources of a CROWN checkout
(src/SVFit, include/SVFit) built with ROOT's ACLiC into a build directory, with a stub of CROWN's logger and a driver that
sets the decay types per leg as quantities::FastMtt does (et, mt, tt, em; mm and ee as two muons or electrons). On the
validation skims of v7b it reproduces the FastMTT friend bitwise (2018 et/mt/tt, 89 535 events, CROWN core bf01115).

    from fastmtt_refit import refit
    columns = refit(frame, "mm", crown="/work/sdaigler/bbtautau/KingMaker/CROWN", build="/tmp/fastmtt", workers=16)

`frame` holds INPUTS (and tau_decaymode_1/2 for the tau legs); the result pt, eta, phi and m_fastmtt per row. Needs ROOT
(LCG_108). Scripts that call it with more than one worker need the `if __name__ == "__main__":` guard (spawn).
"""
from __future__ import annotations

import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

INPUTS = ("pt_1", "pt_2", "eta_1", "eta_2", "phi_1", "phi_2", "mass_1", "mass_2", "met", "metphi", "metcov00", "metcov01", "metcov11")
OUTPUTS = ("pt_fastmtt", "eta_fastmtt", "phi_fastmtt", "m_fastmtt")
LEG_TYPES = {"et": (1, 0), "mt": (2, 0), "tt": (0, 0), "em": (1, 2), "mm": (2, 2), "ee": (1, 1)}  # 0 tau_h, 1 electron, 2 muon
SOURCES = ("AuxFunctions", "MeasuredTauLepton", "FastMTT")
LOGGER = """// stub of CROWN's utility/Logger.hxx: the messages are dropped
#pragma once
#include <memory>
#include <string>
struct NullLogger {
    template <typename... A> void debug(A &&...) {}
    template <typename... A> void info(A &&...) {}
    template <typename... A> void warn(A &&...) {}
    template <typename... A> void error(A &&...) {}
};
struct Logger {
    static std::shared_ptr<NullLogger> get(const std::string &) {
        static auto logger = std::make_shared<NullLogger>();
        return logger;
    }
};
"""
DRIVER = """// FastMTT per event with the decay types of quantities::FastMtt: leg types 0 tau_h, 1 electron, 2 muon; the decay
// mode of a lepton is -1
#include "AuxFunctions.cxx"
#include "MeasuredTauLepton.cxx"
#include "FastMTT.cxx"
#include "Math/Vector4D.h"
#include "TMatrixD.h"

namespace nnv {
int decay_type(int type) {
    if (type == 1) return fastmtt::MeasuredTauLepton::kTauToElecDecay;
    if (type == 2) return fastmtt::MeasuredTauLepton::kTauToMuDecay;
    return fastmtt::MeasuredTauLepton::kTauToHadDecay;
}

void run(long n, const float *pt_1, const float *pt_2, const float *eta_1, const float *eta_2, const float *phi_1,
         const float *phi_2, const float *mass_1, const float *mass_2, const float *met_pt, const float *met_phi,
         const float *cov_xx, const float *cov_xy, const float *cov_yy, const int *type_1, const int *type_2,
         const int *dm_1, const int *dm_2, float *out_pt, float *out_eta, float *out_phi, float *out_m) {
    for (long i = 0; i < n; ++i) {
        std::vector<fastmtt::MeasuredTauLepton> legs;
        TMatrixD cov(2, 2);
        cov[0][0] = cov_xx[i];
        cov[1][0] = cov_xy[i];
        cov[0][1] = cov_xy[i];
        cov[1][1] = cov_yy[i];
        ROOT::Math::PtEtaPhiMVector met(met_pt[i], 0.0, met_phi[i], 0.0);
        legs.push_back(fastmtt::MeasuredTauLepton(decay_type(type_1[i]), pt_1[i], eta_1[i], phi_1[i], mass_1[i], type_1[i] == 0 ? dm_1[i] : -1));
        legs.push_back(fastmtt::MeasuredTauLepton(decay_type(type_2[i]), pt_2[i], eta_2[i], phi_2[i], mass_2[i], type_2[i] == 0 ? dm_2[i] : -1));
        FastMTT algo;
        ROOT::Math::PtEtaPhiMVector result = algo.run(legs, met.X(), met.Y(), cov);
        out_pt[i] = result.Pt();
        out_eta[i] = result.Eta();
        out_phi[i] = result.Phi();
        out_m[i] = result.M();
    }
}
}  // namespace nnv
"""


def _write(path: Path, text: str) -> None:
    """Write only a changed file: ACLiC rebuilds whenever a source is newer than the library."""
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def prepare(crown: Path, build: Path) -> Path:
    """Copy the FastMTT sources of the CROWN checkout into `build` (includes made local), with the stub logger and the
    driver; returns the driver."""
    build = Path(build)
    build.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        for kind, suffix in (("src", "cxx"), ("include", "hxx")):
            text = (Path(crown) / kind / "SVFit" / f"{name}.{suffix}").read_text()
            _write(build / f"{name}.{suffix}", text.replace('"../../include/SVFit/', '"').replace('"../../include/utility/Logger.hxx"', '"Logger.hxx"'))
    _write(build / "Logger.hxx", LOGGER)
    _write(build / "nnv_fastmtt.cxx", DRIVER)
    return build / "nnv_fastmtt.cxx"


def load(crown: Path, build: Path, compile_: bool = True):
    """The compiled driver in ROOT: prepared and built (if out of date) by the calling process, loaded by the workers."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT, "nnv") or not hasattr(ROOT.nnv, "run"):
        driver = prepare(crown, build) if compile_ else Path(build) / "nnv_fastmtt.cxx"
        ROOT.gSystem.SetBuildDir(str(Path(build) / "lib"), True)
        if ROOT.gSystem.CompileMacro(str(driver), "kO") != 1:
            raise RuntimeError(f"FastMTT build failed in {build}")
    return ROOT


def _run(job: tuple) -> np.ndarray:
    crown, build, arrays = job
    ROOT = load(crown, build, compile_=False)
    n = len(arrays["pt_1"])
    columns = [np.ascontiguousarray(arrays[c], dtype=np.float32) for c in INPUTS]
    types = [np.ascontiguousarray(arrays[c], dtype=np.int32) for c in ("type_1", "type_2", "dm_1", "dm_2")]
    outputs = [np.zeros(n, dtype=np.float32) for _ in OUTPUTS]
    if n:
        ROOT.nnv.run(n, *columns, *types, *outputs)
    return np.stack(outputs)


def refit(frame, channel: str, crown: Path, build: Path, workers: int = 16, chunk: int = 20000) -> dict[str, np.ndarray]:
    """pt, eta, phi and m of FastMTT for every row of `frame`, in chunks on `workers` processes."""
    n = len(frame)
    type_1, type_2 = LEG_TYPES[channel]
    arrays = {c: frame[c].to_numpy() for c in INPUTS}
    arrays["type_1"], arrays["type_2"] = np.full(n, type_1), np.full(n, type_2)
    arrays["dm_1"] = frame["tau_decaymode_1"].to_numpy() if type_1 == 0 else np.full(n, -1)
    arrays["dm_2"] = frame["tau_decaymode_2"].to_numpy() if type_2 == 0 else np.full(n, -1)
    load(crown, build)  # build once before the workers start
    jobs = [(crown, build, {k: v[i:i + chunk] for k, v in arrays.items()}) for i in range(0, max(n, 1), chunk)]
    if workers <= 1 or len(jobs) == 1:
        parts = [_run(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_run, jobs))
    result = np.concatenate(parts, axis=1)[:, :n]
    return dict(zip(OUTPUTS, result))


def clean(build: Path) -> None:
    shutil.rmtree(build, ignore_errors=True)
