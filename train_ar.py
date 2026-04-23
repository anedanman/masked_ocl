import sys

from train_mar import main


if __name__ == "__main__":
    has_config = any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:])
    if not has_config:
        sys.argv.extend(["--config", "configs/ar_coco.yaml"])
    main()
