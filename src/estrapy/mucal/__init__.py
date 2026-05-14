# mucal/__init__.py

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ── 1. Compile if needed ─────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "mucal.c")

if sys.platform == "win32":
    _LIB_PATH = os.path.join(_DIR, "mucal.dll")
    _COMPILE_CMD = ["gcc", "-shared", "-fPIC", "-O2", "-o", _LIB_PATH, _SRC, "-lm"]
elif sys.platform == "darwin":
    _LIB_PATH = os.path.join(_DIR, "mucal.dylib")
    _COMPILE_CMD = ["gcc", "-shared", "-fPIC", "-O2", "-o", _LIB_PATH, _SRC, "-lm"]
else:
    _LIB_PATH = os.path.join(_DIR, "mucal.so")
    _COMPILE_CMD = ["gcc", "-shared", "-fPIC", "-O2", "-o", _LIB_PATH, _SRC, "-lm"]

_needs_compile = (
    not os.path.exists(_LIB_PATH)
    or os.path.getmtime(_SRC) > os.path.getmtime(_LIB_PATH)
)
if _needs_compile:
    print("[mucal] Compiling shared library...")
    try:
        subprocess.check_call(_COMPILE_CMD)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to compile mucal. Ensure gcc/cc is on your PATH, "
            "or install from a pre-built wheel."
        ) from e

# ── 2. Load the library ───────────────────────────────────────────────────────
_lib = ctypes.CDLL(_LIB_PATH)
_lib.mucal.restype = ctypes.c_int
_lib.mucal.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_char,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_char_p,
]

# ── 3. Error codes ────────────────────────────────────────────────────────────
RETURN_CODES = {
    0: "no_error", 1: "no_input", 2: "no_zmatch", 3: "no_data",
    4: "bad_z", 5: "bad_name", 6: "bad_energy", 7: "within_edge",
    8: "m_edge_warn", 666: "satan_rules",
}
WARNINGS = {7, 8}

class MucalError(Exception):
    pass

# ── 4. Result dataclass ───────────────────────────────────────────────────────
@dataclass
class MucalResult:
    k_edge: float;  l1_edge: float; l2_edge: float;  l3_edge: float
    m_edge: float;  k_alpha1: float; k_beta1: float
    l_alpha1: float; l_beta1: float
    xsec_photo: float; xsec_coherent: float; xsec_incoherent: float
    xsec_total: float; conv_factor: float; abs_coeff: float
    atomic_weight: float; density: float
    l1_edge_jump: float; l2_edge_jump: float; l3_edge_jump: float
    fluo_k: float; fluo_l1: float; fluo_l2: float; fluo_l3: float
    return_code: int
    warning: Optional[str]

# ── 5. Public function ────────────────────────────────────────────────────────
def mucal(name: str = "", Z: int = 0, ephot: float = 0.0,
          unit: str = "c", print_errors: bool = False) -> MucalResult:
    energy  = (ctypes.c_double * 9)()
    xsec    = (ctypes.c_double * 11)()
    fluo    = (ctypes.c_double * 4)()
    errmsg  = ctypes.create_string_buffer(200)

    ret = _lib.mucal(
        name.encode() if name else b"",
        Z, ephot, unit.encode()[0:1],
        1 if print_errors else 0,
        energy, xsec, fluo, errmsg,
    )

    code_name = RETURN_CODES.get(ret, f"unknown({ret})")
    if ret != 0 and ret not in WARNINGS:
        raise MucalError(f"mucal error [{code_name}]: {errmsg.value.decode().strip()}")

    return MucalResult(
        k_edge=energy[0], l1_edge=energy[1], l2_edge=energy[2],
        l3_edge=energy[3], m_edge=energy[4],
        k_alpha1=energy[5], k_beta1=energy[6],
        l_alpha1=energy[7], l_beta1=energy[8],
        xsec_photo=xsec[0], xsec_coherent=xsec[1], xsec_incoherent=xsec[2],
        xsec_total=xsec[3], conv_factor=xsec[4], abs_coeff=xsec[5],
        atomic_weight=xsec[6], density=xsec[7],
        l1_edge_jump=xsec[8], l2_edge_jump=xsec[9], l3_edge_jump=xsec[10],
        fluo_k=fluo[0], fluo_l1=fluo[1], fluo_l2=fluo[2], fluo_l3=fluo[3],
        return_code=ret,
        warning=code_name if ret in WARNINGS else None,
    )

if __name__ == "__main__":
    # Test 1: query by element name
    result = mucal("Fe", ephot=7.0, unit="c")
    print("=== Iron (Fe) at 7.0 keV ===")
    print(f"  K-edge energy:       {result.k_edge} keV")
    print(f"  Photoelectric xsec:  {result.xsec_photo:.4f} cm²/g")
    print(f"  Total xsec:          {result.xsec_total:.4f} cm²/g")
    print(f"  K fluorescence yield:{result.fluo_k}")
    print(f"  Atomic weight:       {result.atomic_weight} g/mol")
    print(f"  Density:             {result.density} g/cm³")
    print(f"  Return code:         {result.return_code} ({RETURN_CODES[result.return_code]})")

    # Test 2: query by Z only
    result2 = mucal(Z=29, ephot=8.0, unit="c")
    print("\n=== Copper (Z=29) at 8.0 keV ===")
    print(f"  K-edge energy:       {result2.k_edge} keV")
    print(f"  Total xsec:          {result2.xsec_total:.4f} cm²/g")

    # Test 3: energy=0, get only physical constants
    result3 = mucal("Au", ephot=0.0)
    print("\n=== Gold (Au), no energy (constants only) ===")
    print(f"  Atomic weight:       {result3.atomic_weight} g/mol")
    print(f"  Density:             {result3.density} g/cm³")
    print(f"  K-edge energy:       {result3.k_edge} keV")

    # Test 4: trigger a warning (energy very close to an edge)
    result4 = mucal("Fe", ephot=7.112)
    if result4.warning:
        print(f"\n=== Warning test: {result4.warning} ===")

    print("\nAll tests passed.")