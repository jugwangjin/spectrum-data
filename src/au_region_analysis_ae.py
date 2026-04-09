"""
DEPRECATED. `au_region_analysis.main()`을 그대로 호출한다.

예전에는 AE 전용 CLI 래퍼였으나, 메인 파이프라인에서 PyTorch AE를 제거해
torch 없는 환경과 동일하게 동작한다. 스펙트럼 AE 코드는
`deprecated/spectrum_autoencoder.py`에만 보관된다.
"""

from __future__ import annotations

import sys
import warnings


def main() -> None:
    warnings.warn(
        "au_region_analysis_ae is deprecated; use au_region_analysis (same behavior now).",
        DeprecationWarning,
        stacklevel=1,
    )
    import au_region_analysis as aur

    aur.main()


if __name__ == "__main__":
    main()
