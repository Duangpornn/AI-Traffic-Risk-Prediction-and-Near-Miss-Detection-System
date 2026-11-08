from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from traffic_risk import process_video


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "models" / "thai_vehicle_yolov8_best.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_HORIZON_SECONDS = 3.0

TEXT = {
    "ไทย": {
        "title": "AI Traffic Risk Prediction",
        "subtitle": "ตรวจจับและติดตามยานพาหนะ พร้อมคัดกรองเหตุการณ์เสี่ยงจากวิดีโอ",
        "language": "ภาษา / Language",
        "settings": "ตั้งค่าการวิเคราะห์",
        "model": "โมเดล",
        "confidence": "ความมั่นใจขั้นต่ำ",
        "confidence_help": "เกณฑ์คะแนนจากโมเดล: การตรวจจับที่ได้คะแนนต่ำกว่าค่านี้จะถูกตัดออก ค่าสูงช่วยกรองผลที่ไม่แน่ใจ แต่อาจพลาดรถบางคัน",
        "horizon": "ช่วงคาดการณ์ (วินาที)",
        "horizon_help": "เวลาสูงสุดที่ใช้คาดการณ์จุดเข้าใกล้จากการเคลื่อนที่ในภาพ ระบบจะคัดกรอง near-miss ที่คาดว่าจะเกิดภายในช่วงเวลานี้",
        "reset_settings": "รีเซ็ตการตั้งค่า",
        "tracker": "ระบบติดตามใช้ ByteTrack จาก Ultralytics",
        "ttc_notice": "TTC ในรุ่นนี้คำนวณจากการเคลื่อนที่ในพิกเซล ยังไม่ใช่ TTC หน่วยโลกที่ปรับเทียบกล้อง",
        "missing_runtime": "แพ็กเกจที่จำเป็นยังไม่พร้อม",
        "install_hint": "ติดตั้งแพ็กเกจด้วยคำสั่ง",
        "missing_model": "ยังไม่พบไฟล์โมเดล `.pt` กรุณาวางโมเดลที่ฝึกแล้วที่",
        "model_env_hint": "หรือกำหนดตัวแปร `TRAFFIC_MODEL_PATH`",
        "upload": "อัปโหลดวิดีโอจราจร",
        "upload_help": "วิดีโอจะถูกประมวลผลในเครื่องที่รัน Streamlit",
        "steps": "ขั้นตอนใช้งาน",
        "step_model": "วางโมเดล `.pt` ในตำแหน่งที่แสดงในแถบตั้งค่า",
        "step_upload": "อัปโหลดวิดีโอและกด **เริ่มวิเคราะห์**",
        "step_results": "ตรวจดูสถิติ เหตุการณ์เสี่ยง และดาวน์โหลดวิดีโอผลลัพธ์",
        "analyze": "เริ่มวิเคราะห์",
        "processing": "กำลังวิเคราะห์วิดีโอ...",
        "processing_percent": "กำลังวิเคราะห์... {percent:.0%}",
        "complete": "วิเคราะห์เสร็จแล้ว",
        "analysis_failed": "วิเคราะห์ไม่สำเร็จ",
        "unexpected_error": "เกิดข้อผิดพลาดระหว่างประมวลผล",
        "results": "ผลการวิเคราะห์",
        "summary_result": "ผลลัพธ์",
        "summary_value": "ค่า",
        "summary_duration": "ความยาววิดีโอ",
        "summary_trajectory_records": "ข้อมูลวิถีการเคลื่อนที่",
        "summary_total_track_ids": "จำนวน Track ID ทั้งหมด",
        "summary_tracks_10_frames": "Tracks ≥ 10 เฟรม",
        "summary_risk_events": "เหตุการณ์เสี่ยง",
        "summary_validated_risk_events": "เหตุการณ์เสี่ยงที่ผ่านเกณฑ์",
        "summary_high_risk": "ความเสี่ยงสูง",
        "summary_medium_high_risk": "ความเสี่ยงปานกลาง-สูง",
        "analyzed_video": "วิดีโอที่วิเคราะห์",
        "tracked_objects": "วัตถุที่ติดตาม",
        "near_misses": "Near-miss",
        "frames": "เฟรมที่วิเคราะห์",
        "duration": "ความยาววิดีโอ",
        "no_events": "ไม่พบเหตุการณ์ near-miss ตามเกณฑ์การคัดกรองในวิดีโอนี้",
        "download": "ดาวน์โหลดวิดีโอผลลัพธ์",
        "safety_note": "ผลความเสี่ยงเป็นการคัดกรองจากตำแหน่งในภาพและการเคลื่อนที่ของ track ควรปรับเทียบกล้องและตรวจสอบโดยผู้เชี่ยวชาญก่อนใช้เพื่อการตัดสินใจด้านความปลอดภัย",
        "event_time": "เวลาเริ่ม (วินาที)",
        "vehicles": "รถที่เกี่ยวข้อง",
        "level": "ระดับ",
        "risk_score": "Risk Score",
        "minimum_ttc": "TTC ต่ำสุด (วินาที)",
        "minimum_distance": "ระยะใกล้สุด (พิกเซล)",
        "high": "สูง",
        "medium": "ปานกลาง",
        "watch": "เฝ้าระวัง",
    },
    "English": {
        "title": "AI Traffic Risk Prediction",
        "subtitle": "Detect and track vehicles, then screen traffic video for potential risks.",
        "language": "Language / ภาษา",
        "settings": "Analysis settings",
        "model": "Model",
        "confidence": "Minimum confidence",
        "confidence_help": "Model score threshold: detections below this value are filtered out. A higher value filters uncertain results but may miss some vehicles.",
        "horizon": "Prediction horizon (seconds)",
        "horizon_help": "Maximum time used to predict closest approach from image motion. Near-misses are screened only when predicted within this period.",
        "reset_settings": "Reset settings",
        "tracker": "Tracking powered by Ultralytics ByteTrack",
        "ttc_notice": "TTC is estimated from pixel motion. It is not camera-calibrated world-coordinate TTC.",
        "missing_runtime": "Required packages are not ready",
        "install_hint": "Install dependencies with",
        "missing_model": "Trained `.pt` model not found. Place the model at:",
        "model_env_hint": "or set the `TRAFFIC_MODEL_PATH` environment variable.",
        "upload": "Upload traffic video",
        "upload_help": "Video is processed on the machine running Streamlit.",
        "steps": "Getting started",
        "step_model": "Place the trained `.pt` model at the path shown in the sidebar.",
        "step_upload": "Upload a video and click **Analyze**.",
        "step_results": "Review statistics and risk events, then download the result video.",
        "analyze": "Analyze",
        "processing": "Analyzing video...",
        "processing_percent": "Analyzing... {percent:.0%}",
        "complete": "Analysis complete",
        "analysis_failed": "Analysis failed",
        "unexpected_error": "An error occurred while processing the video",
        "results": "Analysis results",
        "summary_result": "Result",
        "summary_value": "Value",
        "summary_duration": "Video duration",
        "summary_trajectory_records": "Trajectory records",
        "summary_total_track_ids": "Total Track IDs",
        "summary_tracks_10_frames": "Tracks ≥ 10 frames",
        "summary_risk_events": "Risk Events",
        "summary_validated_risk_events": "Validated Risk Events",
        "summary_high_risk": "High Risk",
        "summary_medium_high_risk": "Medium-High Risk",
        "analyzed_video": "Analyzed video",
        "tracked_objects": "Tracked objects",
        "near_misses": "Near-misses",
        "frames": "Frames analyzed",
        "duration": "Video duration",
        "no_events": "No near-miss events met the screening criteria in this video.",
        "download": "Download analyzed video",
        "safety_note": "Risk results are a screening heuristic based on image position and track motion. Calibrate the camera and consult a qualified expert before using these results for safety decisions.",
        "event_time": "Start time (seconds)",
        "vehicles": "Vehicles involved",
        "level": "Level",
        "risk_score": "Risk score",
        "minimum_ttc": "Minimum TTC (seconds)",
        "minimum_distance": "Closest approach (pixels)",
        "high": "High",
        "medium": "Medium",
        "watch": "Monitor",
    },
}


def reset_analysis_settings() -> None:
    st.session_state["confidence"] = DEFAULT_CONFIDENCE
    st.session_state["horizon"] = DEFAULT_HORIZON_SECONDS


@st.cache_resource(show_spinner="กำลังเตรียมระบบวิเคราะห์...")
def check_runtime() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401

        return True, ""
    except ImportError as error:
        return False, str(error)


def render_result(copy: dict[str, str]) -> None:
    result_view = st.session_state.get("traffic_risk_result")
    if not result_view:
        return
    st.subheader(copy["results"])
    st.caption(f"{copy['analyzed_video']}: {result_view['filename']}")
    events = result_view["events"]
    high_risk_count = sum(event["level"] == "high" for event in events)
    medium_high_risk_count = sum(event["level"] == "medium" for event in events)
    summary_rows = [
        {
            copy["summary_result"]: copy["summary_duration"],
            copy["summary_value"]: f'{result_view["duration"]:.2f} s',
        },
        {
            copy["summary_result"]: copy["summary_trajectory_records"],
            copy["summary_value"]: f'{result_view["trajectory_records"]:,} records',
        },
        {
            copy["summary_result"]: copy["summary_total_track_ids"],
            copy["summary_value"]: f'{result_view["tracks"]:,} IDs',
        },
        {
            copy["summary_result"]: copy["summary_tracks_10_frames"],
            copy["summary_value"]: f'{result_view["tracks_at_least_10_frames"]:,} tracks',
        },
        {
            copy["summary_result"]: copy["summary_risk_events"],
            copy["summary_value"]: f'{len(events):,} events',
        },
        {
            copy["summary_result"]: copy["summary_validated_risk_events"],
            copy["summary_value"]: f'{high_risk_count + medium_high_risk_count:,} events',
        },
        {
            copy["summary_result"]: copy["summary_high_risk"],
            copy["summary_value"]: f'{high_risk_count:,} events',
        },
        {
            copy["summary_result"]: copy["summary_medium_high_risk"],
            copy["summary_value"]: f'{medium_high_risk_count:,} events',
        },
    ]
    st.table(summary_rows)

    if result_view["events"]:
        event_rows = [
            {
                copy["event_time"]: event["time"],
                copy["vehicles"]: f'#{event["first_id"]} & #{event["second_id"]}',
                copy["level"]: copy[event["level"]],
                copy["risk_score"]: event["score"],
                copy["minimum_ttc"]: event["ttc"],
                copy["minimum_distance"]: event["separation"],
            }
            for event in result_view["events"]
        ]
        st.dataframe(event_rows, width="stretch", hide_index=True)
    else:
        st.success(copy["no_events"])

    st.video(result_view["video"], format="video/mp4")
    st.download_button(
        copy["download"],
        data=result_view["video"],
        file_name=f"analyzed-{Path(result_view['filename']).stem}.mp4",
        mime="video/mp4",
    )
    st.caption(copy["safety_note"])


def main() -> None:
    st.set_page_config(
        page_title="AI Traffic Risk",
        page_icon="🚦",
        layout="wide",
    )
    st.session_state.setdefault("confidence", DEFAULT_CONFIDENCE)
    st.session_state.setdefault("horizon", DEFAULT_HORIZON_SECONDS)
    language = st.radio(
        "ภาษา / Language",
        options=("ไทย", "English"),
        horizontal=True,
        key="language",
    )
    copy = TEXT[language]
    st.title(f"🚦 {copy['title']}")
    st.caption(copy["subtitle"])

    model_path = Path(os.environ.get("TRAFFIC_MODEL_PATH", str(DEFAULT_MODEL)))
    with st.sidebar:
        st.header(copy["settings"])
        st.caption(f"{copy['model']}: `{model_path}`")
        confidence = st.slider(
            copy["confidence"],
            0.1,
            0.9,
            step=0.05,
            key="confidence",
        )
        st.caption(copy["confidence_help"])
        horizon = st.slider(
            copy["horizon"],
            0.5,
            5.0,
            step=0.5,
            key="horizon",
        )
        st.caption(copy["horizon_help"])
        st.button(
            copy["reset_settings"],
            on_click=reset_analysis_settings,
            width="stretch",
        )
        st.divider()
        st.caption(copy["tracker"])
        st.caption(copy["ttc_notice"])

    ready, runtime_error = check_runtime()
    if not ready:
        st.error(f"{copy['missing_runtime']}: {runtime_error}")
        st.code(f"{copy['install_hint']}: pip install -r requirements.txt")
        return

    if not model_path.is_file():
        st.warning(
            f"{copy['missing_model']} "
            "`machine/models/thai_vehicle_yolov8_best.pt` "
            f"{copy['model_env_hint']}"
        )

    uploaded_video = st.file_uploader(
        copy["upload"],
        type=["mp4", "mov", "avi", "mkv"],
        help=copy["upload_help"],
    )
    if uploaded_video is None:
        st.markdown(
            f"""
            **{copy['steps']}**
            1. {copy['step_model']}
            2. {copy['step_upload']}
            3. {copy['step_results']}
            """
        )
        render_result(copy)
        return

    st.video(uploaded_video)
    if st.button(
        copy["analyze"],
        type="primary",
        disabled=not model_path.is_file(),
    ):
        progress = st.progress(0, text=copy["processing"])
        status = st.empty()
        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.TemporaryDirectory(prefix="traffic-risk-") as temp_dir:
                source_path = Path(temp_dir) / f"input{suffix}"
                output_path = Path(temp_dir) / "traffic-risk-result.mp4"
                source_path.write_bytes(uploaded_video.getvalue())

                result = process_video(
                    source_path,
                    output_path,
                    model_path,
                    confidence=confidence,
                    horizon_seconds=horizon,
                    progress_callback=lambda value: progress.progress(
                        value, text=copy["processing_percent"].format(percent=value)
                    ),
                )
                video_bytes = output_path.read_bytes()

            progress.progress(1.0, text=copy["complete"])
            st.session_state["traffic_risk_result"] = {
                "video": video_bytes,
                "filename": uploaded_video.name,
                "fps": result.fps,
                "frames": result.frames_processed,
                "duration": result.duration_seconds,
                "tracks": result.tracked_objects,
                "trajectory_records": result.trajectory_records,
                "tracks_at_least_10_frames": result.tracks_at_least_10_frames,
                "events": [
                    {
                        "time": round(event.first_seen_seconds, 2),
                        "first_id": event.first_id,
                        "second_id": event.second_id,
                        "level": {
                            "สูง": "high",
                            "ปานกลาง": "medium",
                            "เฝ้าระวัง": "watch",
                        }[event.level],
                        "score": event.peak_score,
                        "ttc": round(event.minimum_ttc_seconds, 2),
                        "separation": round(
                            event.minimum_separation_pixels, 1
                        ),
                    }
                    for event in result.events
                ],
            }
        except (OSError, ValueError, RuntimeError) as error:
            status.error(f"{copy['analysis_failed']}: {error}")
            return
        except Exception as error:
            status.error(f"{copy['unexpected_error']}: {error}")
            raise

    render_result(copy)


if __name__ == "__main__":
    main()
