"""Run one vision provider and export anonymous audience session events."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from audience import AudienceSessionTracker
from audience.io import JsonlEventWriter
from configs import CFG
from preprocess import preprocess
from providers.factory import create_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("uniface",), default="uniface")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--webcam", type=int, nargs="?", const=0)
    parser.add_argument("--standee-id", default="ST-LOCAL")
    parser.add_argument("--campaign-id", default="CMP-LOCAL")
    parser.add_argument("--creative-id", default="AD-LOCAL")
    parser.add_argument("--events-out", default="outputs/audience-events.jsonl")
    parser.add_argument("--render-out")
    parser.add_argument(
        "--with-attributes",
        action="store_true",
        help="enable experimental age/gender inference (disabled by default)",
    )
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def draw_observations(frame, observations) -> None:
    for observation in observations:
        x1, y1, x2, y2 = observation.bbox
        color = (0, 200, 0) if observation.attentive else (140, 140, 140)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        parts = [f"#{observation.track_id}"]
        if observation.age is not None:
            parts.append(
                f"A:{observation.age:.0f}/{observation.age_group} n:{observation.attribute_sample_count}"
            )
        if observation.gender:
            parts.append(observation.gender)
        if observation.yaw is not None:
            parts.append(f"Y:{observation.yaw:.0f} P:{observation.pitch:.0f}")
        cv2.putText(
            frame,
            " | ".join(parts),
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def open_writer(path: str | None, fps: float, size: tuple[int, int]):
    if not path:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def main() -> None:
    args = parse_args()
    source = args.video if args.video else args.webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open input: {source}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 0:
        input_fps = 25.0
    provider = create_provider(args.provider, enable_attributes=args.with_attributes)
    sessions = AudienceSessionTracker(args.standee_id, args.campaign_id, args.creative_id)
    video_writer = None
    frame_index = 0
    started = time.perf_counter()

    try:
        with JsonlEventWriter(args.events_out) as event_writer:
            while True:
                ok, source_frame = cap.read()
                if not ok:
                    break
                frame = preprocess(source_frame)
                timestamp = time.time() if args.webcam is not None else frame_index / input_fps
                observations = provider.observe(frame, timestamp, source_frame=source_frame)
                event_writer.write(sessions.update(observations, timestamp))
                draw_observations(frame, observations)

                if args.render_out and video_writer is None:
                    video_writer = open_writer(args.render_out, input_fps, (frame.shape[1], frame.shape[0]))
                if video_writer is not None:
                    video_writer.write(frame)
                if args.display:
                    cv2.imshow(f"audience-poc ({args.provider})", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
            event_writer.write(sessions.flush())
    finally:
        provider.reset()
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    measured_fps = frame_index / elapsed if elapsed else 0.0
    print(f"Processed {frame_index} frames with {args.provider} at {measured_fps:.2f} FPS")
    print(f"Session events: {args.events_out}")


if __name__ == "__main__":
    main()
