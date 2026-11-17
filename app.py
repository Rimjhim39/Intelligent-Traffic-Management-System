from __future__ import annotations

import random
import tempfile
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "yolo26n.pt"
DQN_PATH = ROOT / "traffic_dqn.zip"
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}
VEHICLE_COLORS = {
    "Car": (42, 211, 190),
    "Motorcycle": (64, 166, 255),
    "Bus": (255, 190, 75),
    "Truck": (210, 125, 255),
}
DIRECTIONS = ("North", "East", "South", "West")

st.set_page_config(
    page_title="Intelligent Traffic | Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #07111f;
        --panel: #101e30;
        --panel-soft: #14253a;
        --line: #20364d;
        --cyan: #38d9f5;
        --muted: #91a7bc;
        --text: #e8f1f8;
    }
    .stApp { background: var(--navy); color: var(--text); }
    [data-testid="stSidebar"] {
        background: #0b1727;
        border-right: 1px solid var(--line);
    }
    h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.02em; }
    p, label, [data-testid="stCaptionContainer"] { color: #b7c8d8; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--panel), #0c1929);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--cyan); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(16, 30, 48, 0.62);
        border-color: var(--line) !important;
        border-radius: 12px;
    }
    div[data-testid="stAlert"] { border-radius: 10px; }
    div.stButton > button {
        border-radius: 8px;
        border: 1px solid #2b4964;
        min-height: 2.5rem;
    }
    .eyebrow {
        color: var(--cyan); font-size: .75rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase;
    }
    .muted { color: var(--muted); }
    .signal-card {
        background: #0a1625; border: 1px solid var(--line);
        border-radius: 10px; padding: 12px; text-align: center;
    }
    .signal-lamp {
        width: 14px; height: 14px; border-radius: 50%;
        display: inline-block; margin-right: 7px;
        vertical-align: -2px; background: #36475a;
    }
    .green { background: #2bd47d; box-shadow: 0 0 12px #2bd47d88; }
    .yellow { background: #f7c948; box-shadow: 0 0 12px #f7c94888; }
    .red { background: #f05263; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


@st.cache_resource
def load_dqn(model_path: str):
    from stable_baselines3 import DQN

    return DQN.load(model_path)


def dqn_unavailable_message(error: Exception) -> str:
    if isinstance(error, ModuleNotFoundError) and error.name == "stable_baselines3":
        return (
            "Stable-Baselines3 is not installed in this Python environment. "
            "The saved DQN file is present, but DQN inference is unavailable."
        )
    return f"Could not load the saved DQN model: {error}"


def new_world() -> dict:
    return {
        "queues": [0, 0, 0, 0],
        "phase": 0,
        "signal": "GREEN",
        "target_phase": 0,
        "tick": 0,
        "served": 0,
        "queue_vehicle_ticks": 0,
        "history": [],
    }


def controller_action(
    controller: str, world: dict, model=None, manual_phase: int = 0
) -> tuple[int, str]:
    north_south_queue = world["queues"][0] + world["queues"][2]
    east_west_queue = world["queues"][1] + world["queues"][3]

    if controller == "Manual":
        return manual_phase, f"Manual selection: {'NS' if manual_phase == 0 else 'EW'}"

    if controller == "DQN":
        if model is None:
            raise RuntimeError("The DQN model is not loaded.")
        observation = np.asarray(
            [
                min(north_south_queue, 50) / 50,
                min(east_west_queue, 50) / 50,
                float(world["phase"]),
            ],
            dtype=np.float32,
        )
        action, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
        if action not in (0, 1):
            raise ValueError(f"DQN returned unsupported signal action: {action}")
        return action, (
            f"DQN input NS={observation[0]:.2f}, EW={observation[1]:.2f}, "
            f"phase={int(observation[2])}; action={action} "
            f"({'NS' if action == 0 else 'EW'} green)"
        )

    if north_south_queue > east_west_queue:
        action = 0
    elif east_west_queue > north_south_queue:
        action = 1
    else:
        action = world["phase"]
    return action, (
        f"Traffic-aware: NS queue {north_south_queue}, "
        f"EW queue {east_west_queue}; "
        f"{'NS' if action == 0 else 'EW'} selected"
    )


def advance_world(
    world: dict, arrivals: list[int], controller: str, model=None,
    manual_phase: int = 0,
) -> str:
    reason = "Clearing yellow before changing phase"
    if world["signal"] == "YELLOW":
        for index, count in enumerate(arrivals):
            world["queues"][index] = min(50, world["queues"][index] + count)
        world["phase"] = world["target_phase"]
        world["signal"] = "GREEN"
    else:
        action, reason = controller_action(
            controller, world, model=model, manual_phase=manual_phase
        )
        for index, count in enumerate(arrivals):
            world["queues"][index] = min(50, world["queues"][index] + count)
        if action != world["phase"]:
            world["target_phase"] = action
            world["signal"] = "YELLOW"
        else:
            open_directions = (0, 2) if world["phase"] == 0 else (1, 3)
            for index in open_directions:
                departed = min(world["queues"][index], 3)
                world["queues"][index] -= departed
                world["served"] += departed

    world["tick"] += 1
    world["queue_vehicle_ticks"] += sum(world["queues"])
    world["history"].append(
        {
            "Tick": world["tick"],
            "North": world["queues"][0],
            "East": world["queues"][1],
            "South": world["queues"][2],
            "West": world["queues"][3],
            "Total queue": sum(world["queues"]),
            "Signal": world["signal"],
            "Green axis": "NS" if world["phase"] == 0 else "EW",
            "Vehicles served": world["served"],
            "Estimated wait (ticks)": (
                world["queue_vehicle_ticks"] / world["served"]
                if world["served"]
                else 0.0
            ),
        }
    )
    world["last_reason"] = reason
    return reason


def signal_card(direction: str, world: dict) -> str:
    if world["signal"] == "YELLOW":
        is_clearing_axis = (direction in ("North", "South")) == (
            world["phase"] == 0
        )
        lamp = "yellow" if is_clearing_axis else "red"
        color = "YELLOW" if is_clearing_axis else "RED"
    elif (direction in ("North", "South")) == (world["phase"] == 0):
        lamp = "green"
        color = "GREEN"
    else:
        lamp = "red"
        color = "RED"
    queue = world["queues"][DIRECTIONS.index(direction)]
    return (
        '<div class="signal-card">'
        f'<span class="signal-lamp {lamp}"></span><b>{color}</b><br>'
        f'<span class="muted">{direction} · queue {queue}</span>'
        "</div>"
    )


def draw_intersection(world: dict) -> None:
    north, east, south, west = st.columns(4)
    north.markdown(signal_card("North", world), unsafe_allow_html=True)
    east.markdown(signal_card("East", world), unsafe_allow_html=True)
    south.markdown(signal_card("South", world), unsafe_allow_html=True)
    west.markdown(signal_card("West", world), unsafe_allow_html=True)

    vehicles = []
    for direction_index, direction in enumerate(DIRECTIONS):
        queue = world["queues"][direction_index]
        for vehicle_index in range(min(queue, 12)):
            if direction == "North":
                x, y = 47, 7 + vehicle_index * 6
            elif direction == "South":
                x, y = 53, 93 - vehicle_index * 6
            elif direction == "West":
                x, y = 7 + vehicle_index * 6, 53
            else:
                x, y = 93 - vehicle_index * 6, 47
            vehicles.append(
                f'<circle cx="{x}" cy="{y}" r="2.1" fill="#38d9f5"/>'
            )
    svg = f"""
    <svg viewBox="0 0 100 100" role="img" aria-label="Four-way traffic intersection"
         style="width:100%;max-height:300px;background:#10261f;border-radius:12px">
      <rect x="38" y="0" width="24" height="100" fill="#303c4b"/>
      <rect x="0" y="38" width="100" height="24" fill="#303c4b"/>
      <path d="M50 0v34 M50 66v34 M0 50h34 M66 50h34"
            stroke="#d7e3ed" stroke-width=".7" stroke-dasharray="3 3"/>
      <rect x="38" y="38" width="24" height="24" fill="#273342"/>
      {''.join(vehicles)}
      <text x="50" y="15" text-anchor="middle" fill="#dbe8f2" font-size="5">N</text>
      <text x="50" y="91" text-anchor="middle" fill="#dbe8f2" font-size="5">S</text>
      <text x="9" y="51" text-anchor="middle" fill="#dbe8f2" font-size="5">W</text>
      <text x="91" y="51" text-anchor="middle" fill="#dbe8f2" font-size="5">E</text>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


def density_for(total: int) -> str:
    if total < 5:
        return "Low"
    if total < 12:
        return "Medium"
    return "High"


def poisson_arrival(rng: random.Random, rate: float) -> int:
    if rate <= 0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def release_video() -> None:
    capture = st.session_state.get("video_capture")
    if capture is not None:
        capture.release()
    st.session_state.video_capture = None

    temp_path = st.session_state.get("video_temp_path")
    if temp_path:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError as error:
            st.session_state.video_cleanup_error = str(error)
    st.session_state.video_temp_path = None


def start_video(video_bytes: bytes, filename: str) -> None:
    release_video()
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_path = temp_video.name

    capture = cv2.VideoCapture(temp_path)
    if not capture.isOpened():
        capture.release()
        Path(temp_path).unlink(missing_ok=True)
        raise RuntimeError("OpenCV could not open the uploaded video.")

    st.session_state.video_capture = capture
    st.session_state.video_temp_path = temp_path
    st.session_state.video_running = True
    st.session_state.video_paused = False
    st.session_state.video_history = []
    st.session_state.video_frame = 0
    st.session_state.video_total_detections = 0
    st.session_state.video_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    st.session_state.video_error = None
    st.session_state.video_filename = filename


def init_state() -> None:
    defaults = {
        "video_capture": None,
        "video_temp_path": None,
        "video_running": False,
        "video_paused": False,
        "video_history": [],
        "video_frame": 0,
        "video_total_detections": 0,
        "video_counts": {name: 0 for name in VEHICLE_CLASSES.values()},
        "video_error": None,
        "video_filename": None,
        "video_cleanup_error": None,
        "simulation_running": False,
        "simulation_worlds": {
            "DQN": new_world(),
            "TRAFFIC_AWARE": new_world(),
            "Manual": new_world(),
        },
        "simulation_rng": random.Random(42),
        "simulation_records": [],
        "simulation_has_dqn": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">SMART CITY · TRAFFIC OPERATIONS</div>',
                unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


init_state()
st.sidebar.markdown("## 🚦 Traffic Operations")
st.sidebar.caption("INTELLIGENT TRANSPORT · LIVE DEMONSTRATION")
page = st.sidebar.radio(
    "Navigation",
    ("Command Center", "Live Detection", "Simulation", "Analytics & Reports"),
    label_visibility="collapsed",
)
active_controller = st.sidebar.selectbox(
    "Active signal controller",
    ("TRAFFIC_AWARE", "DQN"),
    help="DQN uses the original NS queue, EW queue, and phase observation.",
)

st.sidebar.markdown("---")
st.sidebar.caption("DEMO SYSTEM · No real traffic signals are connected.")

if not WEIGHTS_PATH.is_file():
    st.sidebar.error(f"YOLO weights not found: {WEIGHTS_PATH.name}")
if not DQN_PATH.is_file():
    st.sidebar.error(f"DQN model not found: {DQN_PATH.name}")

if page == "Command Center":
    page_header(
        "Intelligent Traffic Command Center",
        "Computer vision and reinforcement learning for a four-way intersection.",
    )
    primary_world = st.session_state.simulation_worlds[active_controller]
    dqn_ready = False
    if active_controller == "DQN":
        try:
            load_dqn(str(DQN_PATH))
            dqn_ready = True
            controller_status = "DQN model loaded"
        except Exception as error:
            controller_status = dqn_unavailable_message(error)
    else:
        controller_status = "Traffic-aware controller ready"
    visible_count = sum(st.session_state.video_counts.values())
    density = density_for(visible_count)
    signal_text = (
        "Controller unavailable"
        if active_controller == "DQN" and not dqn_ready
        else "YELLOW clearance"
        if primary_world["signal"] == "YELLOW"
        else "North–South green"
        if primary_world["phase"] == 0
        else "East–West green"
    )
    system_status = (
        "DETECTION LIVE"
        if st.session_state.video_running and not st.session_state.video_paused
        else "DETECTION PAUSED"
        if st.session_state.video_paused
        else "SIMULATION LIVE"
        if st.session_state.simulation_running
        else "SYSTEM READY · AWAITING INPUT"
    )

    metrics = st.columns(4)
    metrics[0].metric(
        "Total detections",
        st.session_state.video_total_detections,
    )
    metrics[1].metric("Traffic density", density)
    metrics[2].metric("Current signal", signal_text)
    metrics[3].metric("System status", system_status)

    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            st.subheader("Intersection overview")
            draw_intersection(primary_world)
            st.caption(
                f"Controller: {active_controller} · "
                f"Simulation tick: {primary_world['tick']}"
            )
    with right:
        with st.container(border=True):
            st.subheader("Controller & system")
            st.markdown(f"**Active AI controller:** `{active_controller}`")
            st.markdown(f"**Controller status:** {controller_status}")
            st.markdown(f"**Detection:** {system_status}")
            if st.session_state.video_filename:
                st.caption(f"Video source: {st.session_state.video_filename}")
            st.markdown(
                "Signals shown here are simulated recommendations only; "
                "they do not operate physical infrastructure."
            )

    with st.container(border=True):
        st.subheader("Vehicle snapshot")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Vehicle type": name, "Visible now": count}
                    for name, count in st.session_state.video_counts.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Counts describe the latest processed video frame, not unique "
            "vehicles across the recording."
        )

elif page == "Live Detection":
    page_header(
        "Live Vehicle Detection",
        "Upload a traffic clip, then start, pause, or stop frame-by-frame YOLO analysis.",
    )
    upload = st.file_uploader(
        "Traffic video",
        type=("mp4", "avi", "mov", "mkv"),
        help="The uploaded clip is processed locally using the existing yolo26n.pt weights.",
    )
    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.15,
        step=0.05,
    )

    controls = st.columns(3)
    start_clicked = controls[0].button(
        "▶ Start", type="primary", width="stretch"
    )
    pause_clicked = controls[1].button(
        "Ⅱ Pause / ▶ Resume",
        width="stretch",
        key="video_pause",
    )
    stop_clicked = controls[2].button(
        "■ Stop",
        width="stretch",
        key="video_stop",
    )

    if start_clicked:
        if upload is None:
            st.error("Choose a traffic video before starting detection.")
        elif not WEIGHTS_PATH.is_file():
            st.error(f"YOLO weights were not found at {WEIGHTS_PATH}.")
        else:
            try:
                start_video(upload.getvalue(), upload.name)
            except (OSError, RuntimeError) as error:
                st.session_state.video_error = str(error)
                st.error(f"Could not start video: {error}")
    if pause_clicked and st.session_state.video_running:
        st.session_state.video_paused = not st.session_state.video_paused
    if stop_clicked:
        release_video()
        st.session_state.video_running = False
        st.session_state.video_paused = False

    if st.session_state.video_error:
        st.error(st.session_state.video_error)
    if st.session_state.video_cleanup_error:
        st.warning(
            "The video capture stopped, but its temporary file could not be "
            f"removed: {st.session_state.video_cleanup_error}"
        )

    detection_status = (
        "Running"
        if st.session_state.video_running and not st.session_state.video_paused
        else "Paused"
        if st.session_state.video_paused
        else "Stopped"
    )
    status_cols = st.columns(3)
    status_cols[0].metric("Detection status", detection_status)
    status_cols[1].metric("Frame", st.session_state.video_frame)
    status_cols[2].metric(
        "Total detections processed",
        st.session_state.video_total_detections,
    )

    video_area = st.empty()
    counts_area = st.empty()

    @st.fragment(run_every=0.4)
    def detection_frame() -> None:
        if (
            not st.session_state.video_running
            or st.session_state.video_paused
            or st.session_state.video_capture is None
        ):
            video_area.info("Start a video to view annotated detection frames.")
            counts_area.empty()
            return

        capture = st.session_state.video_capture
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            release_video()
            st.session_state.video_running = False
            st.session_state.video_error = "No readable frames were found in the video."
            video_area.error(st.session_state.video_error)
            return

        try:
            detector = load_detector(str(WEIGHTS_PATH))
            results = detector(frame, conf=confidence, verbose=False)
            counts = {name: 0 for name in VEHICLE_CLASSES.values()}
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    name = VEHICLE_CLASSES[class_id]
                    counts[name] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = VEHICLE_COLORS[name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} {float(box.conf[0]):.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

            total = sum(counts.values())
            st.session_state.video_frame += 1
            st.session_state.video_counts = counts
            st.session_state.video_total_detections += total
            st.session_state.video_history.append(
                {
                    "Frame": st.session_state.video_frame,
                    **counts,
                    "Total": total,
                    "Density": density_for(total),
                }
            )
            st.session_state.video_history = st.session_state.video_history[-1000:]

            video_area.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )
            with counts_area.container():
                count_cols = st.columns(4)
                for column, name in zip(count_cols, VEHICLE_CLASSES.values()):
                    column.metric(name, counts[name])
                st.caption(
                    f"Frame {st.session_state.video_frame} · "
                    f"{total} visible vehicles · {density_for(total)} density"
                )
        except Exception as error:
            st.session_state.video_error = f"Detection failed: {error}"
            st.session_state.video_running = False
            video_area.error(st.session_state.video_error)

    detection_frame()

elif page == "Simulation":
    page_header(
        "Interactive Intersection Simulation",
        "Compare controllers against the same generated vehicle arrivals. "
        "Yellow clearance is applied before every signal phase change.",
    )
    config_left, config_right = st.columns(2)
    with config_left:
        mode = st.radio(
            "Control mode",
            ("Automatic", "Manual"),
            horizontal=True,
        )
        if mode == "Manual":
            manual_axis = st.radio(
                "Manual green phase",
                ("North–South", "East–West"),
                horizontal=True,
            )
            manual_phase = 0 if manual_axis == "North–South" else 1
        else:
            manual_phase = 0
    with config_right:
        arrival_rate = st.slider(
            "Vehicle arrival rate (mean per direction / tick)",
            0.0,
            5.0,
            1.2,
            0.2,
        )
        simulation_speed = st.slider(
            "Simulation speed (ticks per refresh)",
            1,
            5,
            1,
        )

    sim_controls = st.columns(3)
    sim_start = sim_controls[0].button(
        "▶ Start / Resume simulation",
        type="primary",
        width="stretch",
        key="simulation_start",
    )
    sim_pause = sim_controls[1].button(
        "Ⅱ Pause simulation",
        width="stretch",
    )
    sim_stop = sim_controls[2].button(
        "■ Stop and reset",
        width="stretch",
    )
    if sim_start:
        st.session_state.simulation_running = True
    if sim_pause:
        st.session_state.simulation_running = False
    if sim_stop:
        st.session_state.simulation_running = False
        st.session_state.simulation_worlds = {
            "DQN": new_world(),
            "TRAFFIC_AWARE": new_world(),
            "Manual": new_world(),
        }
        st.session_state.simulation_records = []
        st.session_state.simulation_rng = random.Random(42)
        st.session_state.simulation_has_dqn = False

    dqn_model = None
    dqn_error = None
    if active_controller == "DQN" or mode == "Automatic":
        try:
            if DQN_PATH.is_file():
                dqn_model = load_dqn(str(DQN_PATH))
            else:
                dqn_error = f"Saved DQN model not found at {DQN_PATH}."
        except Exception as error:
            dqn_error = dqn_unavailable_message(error)
    st.session_state.simulation_has_dqn = dqn_model is not None
    if dqn_error:
        st.warning(dqn_error)
        st.caption(
            "Traffic-aware and manual modes remain available. DQN comparison "
            "data is omitted while its model cannot be loaded."
        )

    @st.fragment(run_every=0.5)
    def simulation_view() -> None:
        worlds = st.session_state.simulation_worlds
        if st.session_state.simulation_running:
            rng = st.session_state.simulation_rng
            history = st.session_state.simulation_records
            for _ in range(simulation_speed):
                arrivals = [
                    poisson_arrival(rng, arrival_rate) for _direction in DIRECTIONS
                ]
                if mode == "Manual":
                    advance_world(
                        worlds["Manual"],
                        arrivals,
                        "Manual",
                        manual_phase=manual_phase,
                    )
                else:
                    if dqn_model is not None:
                        advance_world(
                            worlds["DQN"], arrivals, "DQN", model=dqn_model
                        )
                    advance_world(
                        worlds["TRAFFIC_AWARE"], arrivals, "TRAFFIC_AWARE"
                    )
                    history.append(
                        {
                            "Tick": worlds["TRAFFIC_AWARE"]["tick"],
                            "Arrivals": sum(arrivals),
                            "DQN queue": (
                                sum(worlds["DQN"]["queues"])
                                if dqn_model is not None
                                else None
                            ),
                            "Traffic-aware queue": sum(
                                worlds["TRAFFIC_AWARE"]["queues"]
                            ),
                            "DQN served": (
                                worlds["DQN"]["served"]
                                if dqn_model is not None
                                else None
                            ),
                            "Traffic-aware served": worlds["TRAFFIC_AWARE"]["served"],
                        }
                    )
            st.session_state.simulation_records = history[-2000:]

        simulation_active = (
            "Manual"
            if mode == "Manual"
            else active_controller
        )
        if simulation_active == "DQN" and dqn_model is None:
            st.error(
                "DQN is selected, but the model could not be loaded. "
                "Choose TRAFFIC_AWARE or install the existing project dependency "
                "to run DQN; no substitute controller is used."
            )
            active_world = None
        else:
            active_world = worlds[simulation_active]

        if active_world is not None:
            intersection_col, detail_col = st.columns([1.3, 1])
            with intersection_col:
                with st.container(border=True):
                    st.subheader("Four-way intersection")
                    draw_intersection(active_world)
            with detail_col:
                with st.container(border=True):
                    st.subheader(f"{simulation_active.replace('_', ' ')} controller")
                    st.markdown(
                        f"**Signal:** {'Yellow clearance' if active_world['signal'] == 'YELLOW' else 'North–South green' if active_world['phase'] == 0 else 'East–West green'}"
                    )
                    st.markdown(
                        f"**Decision rationale:** "
                        f"{active_world.get('last_reason', 'Waiting for first simulation tick.')}"
                    )
                    st.markdown(
                        f"**Tick:** {active_world['tick']} · "
                        f"**Vehicles served:** {active_world['served']}"
                    )
                    if mode == "Automatic":
                        st.caption(
                            "Controller comparison uses the same four per-direction "
                            "arrival values at every simulation tick."
                        )

            queue_cols = st.columns(4)
            for column, direction, queue in zip(
                queue_cols, DIRECTIONS, active_world["queues"]
            ):
                column.metric(f"{direction} queue", queue)

        if mode == "Automatic" and dqn_model is not None:
            compare_left, compare_right = st.columns(2)
            for column, name in (
                (compare_left, "DQN"),
                (compare_right, "TRAFFIC_AWARE"),
            ):
                world = worlds[name]
                with column:
                    with st.container(border=True):
                        st.subheader(name.replace("_", " "))
                        st.metric("Current queued vehicles", sum(world["queues"]))
                        st.metric("Vehicles served", world["served"])
                        average_wait = (
                            world["queue_vehicle_ticks"] / world["served"]
                            if world["served"]
                            else 0.0
                        )
                        st.metric(
                            "Estimated average wait (ticks)", f"{average_wait:.1f}"
                        )
        elif mode == "Automatic":
            st.info(
                "The traffic-aware baseline is running. DQN comparison is "
                "unavailable until Stable-Baselines3 and the saved model load."
            )
        else:
            st.info(
                "Manual mode controls the displayed simulation; automatic "
                "controller comparison is paused."
            )

        records = st.session_state.simulation_records
        if records:
            graph_data = pd.DataFrame(records).set_index("Tick")
            graph_data = graph_data.dropna(axis=1, how="all")
            columns = ["Traffic-aware queue"]
            if dqn_model is not None:
                columns.insert(0, "DQN queue")
            st.line_chart(graph_data[columns])

    simulation_view()

elif page == "Analytics & Reports":
    page_header(
        "Analytics & Reports",
        "Review observed video detections and simulated controller performance.",
    )
    video_history = st.session_state.video_history
    simulation_records = st.session_state.simulation_records

    if video_history:
        video_df = pd.DataFrame(video_history)
        counts_by_frame = video_df.set_index("Frame")[
            ["Car", "Motorcycle", "Bus", "Truck"]
        ]
        st.subheader("Vehicle detections over time")
        st.line_chart(counts_by_frame)

        st.subheader("Traffic density distribution")
        density_counts = (
            video_df["Density"]
            .value_counts()
            .reindex(("Low", "Medium", "High"), fill_value=0)
            .rename_axis("Density")
            .to_frame("Frames")
        )
        st.bar_chart(density_counts)
    else:
        st.info("Process a video to populate observed vehicle and density charts.")

    simulation_has_dqn = st.session_state.simulation_has_dqn
    if simulation_records:
        comparison_df = pd.DataFrame(simulation_records)
        latest = comparison_df.iloc[-1]
        aware_world = st.session_state.simulation_worlds["TRAFFIC_AWARE"]
        stats = st.columns(3 if simulation_has_dqn else 2)
        if simulation_has_dqn:
            stats[0].metric(
                "Average queue · DQN",
                f"{comparison_df['DQN queue'].mean():.1f}",
            )
            stats[1].metric(
                "Average queue · traffic-aware",
                f"{comparison_df['Traffic-aware queue'].mean():.1f}",
            )
            stats[2].metric("Last shared arrival count", int(latest["Arrivals"]))
        else:
            stats[0].metric(
                "Average queue · traffic-aware",
                f"{comparison_df['Traffic-aware queue'].mean():.1f}",
            )
            stats[1].metric("Last shared arrival count", int(latest["Arrivals"]))
        st.caption(
            "Average waiting time is reported in simulation ticks and estimated "
            "as accumulated queue vehicle-ticks divided by vehicles served."
        )
        worlds = st.session_state.simulation_worlds
        comparison_rows = [
            {
                "Controller": "Traffic-aware",
                "Average queue": comparison_df["Traffic-aware queue"].mean(),
                "Vehicles served": worlds["TRAFFIC_AWARE"]["served"],
                "Estimated average wait (ticks)": (
                    worlds["TRAFFIC_AWARE"]["queue_vehicle_ticks"]
                    / worlds["TRAFFIC_AWARE"]["served"]
                    if worlds["TRAFFIC_AWARE"]["served"]
                    else 0.0
                ),
            }
        ]
        if simulation_has_dqn:
            comparison_rows.insert(
                0,
                {
                    "Controller": "DQN",
                    "Average queue": comparison_df["DQN queue"].mean(),
                    "Vehicles served": worlds["DQN"]["served"],
                    "Estimated average wait (ticks)": (
                        worlds["DQN"]["queue_vehicle_ticks"] / worlds["DQN"]["served"]
                        if worlds["DQN"]["served"]
                        else 0.0
                    ),
                },
            )
        st.dataframe(
            pd.DataFrame(comparison_rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Run the automatic simulation to record controller comparisons.")

    report_rows = []
    for row in video_history:
        report_rows.append({"Record type": "Video", **row})
    for row in simulation_records:
        report_rows.append({"Record type": "Simulation", **row})
    if report_rows:
        st.download_button(
            "Download CSV report",
            data=pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"),
            file_name="traffic_management_report.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.caption("CSV export becomes available after video or simulation data is recorded.")
