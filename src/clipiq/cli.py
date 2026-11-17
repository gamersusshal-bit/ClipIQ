import argparse
import sys

from clipiq.workflow import run_clipiq


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"run", "review"}:
        from clipiq.commands import main as commands_main
        return commands_main(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="ClipIQ - AI-powered video highlight detection"
    )

    parser.add_argument(
        "video",
        help="Path to the input video"
    )

    parser.add_argument(
        "--output",
        default="data/processed/clipiq_run",
        help="Directory for generated results"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top clips to export per strategy"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Whisper model name (default: configured model)"
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    result = run_clipiq(
        video_path=args.video,
        output_dir=args.output,
        model_name=args.model,
        top_k=args.top_k
    )

    print("\nClipIQ processing completed.")
    print("Output directory:", result["output_dir"])

    for strategy, clips in result["exported_clips"].items():
        print(f"{strategy}: {len(clips)} clips exported")


if __name__ == "__main__":
    main()
