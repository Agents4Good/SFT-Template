"""Push a trained Model folder to the Hugging Face Hub.

Standalone use (latest run under ./src/output)::

    python -m src.utils.push --repo-id USER/MODEL

Explicit folder::

    python -m src.utils.push --folder src/output/run-20261007_184444/Model \
        --repo-id USER/MODEL

Auth via ``HF_TOKEN`` env or ``huggingface login``.
"""

import argparse
import glob
import os


def resolve_model_folder(output_dir, folder=None):
    """Return the Model folder to push.

    Explicit ``folder`` wins; otherwise the newest ``run-*/Model`` under
    ``output_dir``. Raises FileNotFoundError when nothing resolves.
    """
    if folder is not None:
        if not os.path.isdir(folder) or not os.listdir(folder):
            raise FileNotFoundError(
                f"Model folder '{folder}' missing or empty."
            )
        return folder
    candidates = sorted(
        glob.glob(os.path.join(output_dir, "run-*", "Model")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run-*/Model under '{output_dir}'."
        )
    return candidates[-1]


def push_model(folder, repo_id, revision="main", private=False,
               dry_run=False):
    """Upload ``folder`` to ``repo_id``. Returns the folder pushed."""
    if not repo_id:
        raise ValueError("repo_id is required.")
    if not os.path.isdir(folder) or not os.listdir(folder):
        raise FileNotFoundError(
            f"Model folder '{folder}' missing or empty."
        )
    if dry_run:
        print(f"DRY-RUN: would push '{folder}' to '{repo_id}@{revision}'.")
        return folder
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        private=private,
    )
    print(f"Pushed '{folder}' to '{repo_id}@{revision}'.")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=None,
                        help="Model folder to push (default: latest run)")
    parser.add_argument("--output-dir", default="./src/output",
                        help="Run directory for latest-run resolution")
    parser.add_argument("--repo-id", default=None,
                        help="Hub repo id (default: output.push.repo_id)")
    parser.add_argument("--config", default="src/config.yaml",
                        help="Config file (only for output.push.repo_id)")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_id = args.repo_id
    if repo_id is None:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        repo_id = ((cfg.get("output") or {}).get("push") or {}).get("repo_id")
    folder = resolve_model_folder(args.output_dir, args.folder)
    push_model(folder, repo_id, revision=args.revision,
               private=args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
