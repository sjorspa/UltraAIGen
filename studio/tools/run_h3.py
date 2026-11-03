"""Launch musubi's MiniMax-H3 generator with LoRA support for the token-refiner blocks.

musubi only attaches LoRAs to ``DiTBlock`` modules, but LoRAs such as the H3
turbo LoRA also carry deltas for the BF16 ``RefinerBlock`` text-token refiner.
This adds that class to the target list, then runs the unmodified script:

    python run_h3.py /path/to/minimax_h3_generate_video.py [generator args...]
"""

import runpy
import sys


def main() -> None:
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    import musubi_tuner.networks.lora_minimax_h3 as lora_h3

    if "RefinerBlock" not in lora_h3.MINIMAX_H3_TARGET_REPLACE_MODULES:
        lora_h3.MINIMAX_H3_TARGET_REPLACE_MODULES.append("RefinerBlock")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
