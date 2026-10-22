"""Render a real saved native trajectory or a closed-loop skill run to MP4."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .executor import MountRideConfig, sha256


def _font(size):
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load_metrics(path):
    if path is None:
        return None
    path = Path(path)
    rows = []
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
    else:
        payload = json.loads(path.read_text())
        rows = payload.get("samples", payload)
    return rows


def _metric(row, name):
    if name in row:
        return float(row[name])
    nested = {
        "left_palm_Fz_N": ("actual_support_admission", "arms", "left", "Fz_N"),
        "right_palm_Fz_N": ("actual_support_admission", "arms", "right", "Fz_N"),
        "seat_Fz_N": ("mixed_measurement", "seat_vertical_N"),
    }
    if name in nested:
        value = row
        for key in nested[name]:
            value = value[key]
        return float(value)
    if name in ("left_foot_Fz_N", "right_foot_Fz_N"):
        side = "left" if name.startswith("left") else "right"
        return float(
            row["native_stage_metrics"]["feet_GRF_world_N"][side][2]
        )
    if name == "unapproved_T1_Fn_N":
        return float(
            row.get(
                "unapproved_T1_Fn_N",
                row.get(
                    "illegal_normal_scalar_sum_N",
                    row.get("unverified_T1_Fn_N", 0.0),
                ),
            )
        )
    raise KeyError(name)


def _nearest_metric_index(metrics, timestamp):
    times = np.asarray([float(row["time_s"]) for row in metrics])
    return int(
        np.clip(np.searchsorted(times, timestamp), 0, len(metrics) - 1)
    )


def render(
    trajectory,
    out,
    config_path="configs/demo_handload_cross17.json",
    metrics_path=None,
    fps=30,
    width=1280,
    height=720,
    label="Q1 / T1  REAL NATIVE MUJOCO MOUNT DEMO",
):
    config = MountRideConfig.load(config_path)
    config.verify_assets()
    model = mujoco.MjModel.from_xml_path(str(config.resolve("model")))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    with np.load(trajectory, allow_pickle=False) as saved:
        times = saved["time"].copy()
        qpos = saved["qpos"].copy()
    if qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("invalid trajectory")
    metrics = _load_metrics(metrics_path or config.resolve("validated_metrics"))
    output = Path(out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 72.0
    camera.elevation = -10.0
    camera.distance = 1.78
    t1 = model.body("BASELINK").id
    pelvis = model.body("q1_pelvis").id
    arm_bodies = {}
    for side in ("left", "right"):
        arm_bodies[side] = {
            model.body(name).id
            for name in (
                f"q1_{side}_shoulder_pitch_link",
                f"q1_{side}_shoulder_roll_link",
                f"q1_{side}_elbow_link",
                f"q1_{side}_wrist_yaw_link",
            )
        }
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    writer = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    regular = _font(22)
    small = _font(18)
    count = int(np.ceil(times[-1] * fps)) + 1
    try:
        for frame_index in range(count):
            target = frame_index / fps
            index = int(
                np.clip(np.searchsorted(times, target), 0, len(times) - 1)
            )
            data.qpos[:] = qpos[index]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            camera.lookat[:] = (
                0.52 * data.xpos[t1] + 0.48 * data.xpos[pelvis]
            )
            camera.lookat[2] += 0.10
            renderer.update_scene(data, camera)
            for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                    continue
                if not 0 <= geom.objid < model.ngeom:
                    continue
                body_id = int(model.geom_bodyid[geom.objid])
                if body_id in arm_bodies["left"]:
                    geom.rgba[:3] = [0.93, 0.20, 0.15]
                elif body_id in arm_bodies["right"]:
                    geom.rgba[:3] = [0.16, 0.43, 0.96]
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            timestamp = float(times[index])
            draw.rectangle((0, 0, width, 96), fill=(5, 11, 20))
            draw.text(
                (22, 7),
                label,
                font=regular,
                fill=(245, 248, 252),
            )
            draw.text(
                (22, 36),
                "LOCAL SUPPORTED START - NOT AN ORDINARY STANDING MOUNT",
                font=small,
                fill=(255, 167, 84),
            )
            draw.text(
                (22, 63),
                f"t={timestamp:05.2f}s   left arm RED / right arm BLUE",
                font=small,
                fill=(255, 219, 88),
            )
            phase_text = (
                "CROSSING: RIGHT LEG OVER T1"
                if timestamp < 4.9
                else "POST-CROSS SUPPORT HOLD"
            )
            phase_color = (
                (255, 190, 60) if timestamp < 4.9 else (122, 232, 154)
            )
            draw.rounded_rectangle(
                (width - 350, 20, width - 20, 66),
                radius=10,
                fill=(16, 31, 42),
                outline=phase_color,
                width=2,
            )
            draw.text(
                (width - 337, 34),
                phase_text,
                font=small,
                fill=phase_color,
            )
            draw.rectangle(
                (0, height - 116, width, height), fill=(5, 11, 20)
            )
            if metrics:
                row = metrics[_nearest_metric_index(metrics, timestamp)]
                draw.text(
                    (22, height - 104),
                    (
                        f"arm vertical load  L={_metric(row, 'left_palm_Fz_N'):.1f} N   "
                        f"R={_metric(row, 'right_palm_Fz_N'):.1f} N    "
                        f"seat={_metric(row, 'seat_Fz_N'):.1f} N"
                    ),
                    font=small,
                    fill=(241, 245, 249),
                )
                draw.text(
                    (22, height - 76),
                    (
                        f"foot GRF  L={_metric(row, 'left_foot_Fz_N'):.1f} N   "
                        f"R={_metric(row, 'right_foot_Fz_N'):.1f} N    "
                        f"illegal/unverified Fn={_metric(row, 'unapproved_T1_Fn_N'):.1f} N"
                    ),
                    font=small,
                    fill=(241, 245, 249),
                )
            draw.text(
                (22, height - 48),
                "DEMO ONLY: temporary foot support; formal legal hold not claimed",
                font=small,
                fill=(255, 196, 100),
            )
            progress = min(1.0, timestamp / float(times[-1]))
            draw.rectangle(
                (22, height - 16, width - 22, height - 12),
                fill=(38, 48, 58),
            )
            draw.rectangle(
                (
                    22,
                    height - 16,
                    22 + int((width - 44) * progress),
                    height - 12,
                ),
                fill=(55, 183, 123),
            )
            writer.stdin.write(np.asarray(frame).tobytes())
    finally:
        writer.stdin.close()
        error = writer.stderr.read().decode()
        if writer.wait():
            raise RuntimeError(error)
        renderer.close()
    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = {
        "schema": "q1-t1-demo-render-v1",
        "video": str(output),
        "video_sha256": sha256(output),
        "trajectory": str(Path(trajectory).resolve()),
        "trajectory_sha256": sha256(trajectory),
        "fps": fps,
        "resolution": [width, height],
        "frame_count": count,
        "physics_steps": 0,
        "rendering": "saved_native_states_kinematics_only",
        "decode_stderr": decode.stderr,
        "formal_legal_hold_claimed": False,
        "label": label,
    }
    (output.parent / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo_handload_cross17.json")
    parser.add_argument(
        "--trajectory", default="data/handload_cross17_trajectory.npz"
    )
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--out", default="outputs/demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--label", default="Q1 / T1  REAL NATIVE MUJOCO MOUNT DEMO"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.trajectory,
                args.out,
                config_path=args.config,
                metrics_path=args.metrics,
                fps=args.fps,
                width=args.width,
                height=args.height,
                label=args.label,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
