"""
ui.py – All OpenCV drawing helpers for the Sign Language Translator.
Call draw_collector_ui() or draw_translator_ui() and pass the frame;
each function returns the annotated frame.
"""
import cv2


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _bar(frame, x, y, w, h, ratio, fill_color, border_color=(60, 60, 60)):
    """Draw a filled progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
    if ratio > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), fill_color, -1)


def _text(frame, msg, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ─── Data-Collector UI ───────────────────────────────────────────────────────

def draw_collector_ui(frame, word, num_detected, samples_done, total_samples, is_capturing):
    """
    Draws the header banner and progress bar for 01_collect_data.py.
    Returns the annotated frame (modifies in-place and returns for chaining).
    """
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 85), (25, 25, 25), -1)
    _text(frame, f"Recording: {word}", 20, 35, scale=0.9, color=(0, 255, 255), thickness=2)

    status_color = (0, 255, 0) if is_capturing and num_detected > 0 else (200, 200, 200)
    _text(frame, f"Hands: {num_detected}  |  Samples: {samples_done}/{total_samples}",
          20, 68, scale=0.65, color=status_color, thickness=2)
    _text(frame, "[HOLD SPACE TO RECORD | Q: Cancel]", w - 440, 68,
          scale=0.52, color=(180, 180, 180))

    # Progress bar at bottom
    ratio = samples_done / total_samples if total_samples > 0 else 0
    _bar(frame, 20, h - 35, w - 40, 20, ratio, (0, 255, 0))

    return frame


# ─── Translator UI ───────────────────────────────────────────────────────────

def draw_translator_ui(frame, sentence, current_stable_sign,
                       stable_frame_count, in_cooldown, remaining_pause,
                       pause_total, hold_total):
    """
    Draws the full translator overlay:
      - Top header bar with controls hint
      - Cooldown / hold-to-add / ready status bar
      - Bottom sentence ribbon
    Returns the annotated frame.
    """
    h, w = frame.shape[:2]

    # 1. Top header
    cv2.rectangle(frame, (0, 0), (w, 55), (25, 25, 25), -1)
    _text(frame, "Sign Language Translator", 20, 36, scale=0.75, thickness=2)
    _text(frame, "[SPC:Add | B:Del | C:Clear | S:Speak | Q:Quit]",
          w - 520, 36, scale=0.50, color=(180, 180, 180))

    # 2. Status row (cooldown bar OR hold-to-add bar OR ready indicator)
    if in_cooldown:
        ratio = remaining_pause / pause_total
        _bar(frame, 20, 70, 200, 20, ratio, (0, 140, 255))
        _text(frame, f"Pause: {remaining_pause:.1f}s  (Prepare next sign)",
              230, 86, color=(0, 165, 255), thickness=2)
    elif current_stable_sign and stable_frame_count > 0:
        ratio = min(1.0, stable_frame_count / hold_total)
        _bar(frame, 20, 70, 180, 20, ratio, (0, 255, 0))
        _text(frame, f"Adding: {current_stable_sign}", 210, 86,
              color=(0, 255, 0), thickness=2)
    else:
        cv2.circle(frame, (35, 80), 8, (0, 255, 0), -1)
        _text(frame, "Ready for next sign", 55, 86, color=(0, 255, 0), thickness=2)

    # 3. Bottom sentence ribbon
    full_sentence = " ".join(sentence) if sentence else "(Perform sign and hold, or press SPACE to add...)"
    cv2.rectangle(frame, (0, h - 70), (w, h), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, h - 70), (w, h - 68), (0, 255, 255), -1)
    _text(frame, "Sentence:", 20, h - 28, scale=0.75, color=(0, 165, 255), thickness=2)
    _text(frame, full_sentence, 150, h - 28, scale=0.8, thickness=2)

    return frame


def draw_bounding_box(frame, all_x, all_y, w, h,
                      in_cooldown, detected_sign, confidence, remaining_pause):
    """Draws the hand bounding box with an appropriate label tag."""
    if not all_x or not all_y:
        return frame

    x_min = max(0, min(all_x) - 25)
    x_max = min(w, max(all_x) + 25)
    y_min = max(0, min(all_y) - 25)
    y_max = min(h, max(all_y) + 25)

    if in_cooldown:
        color = (0, 140, 255)
        tag   = f"Paused ({remaining_pause:.1f}s)"
    elif detected_sign:
        color = (0, 255, 0)
        tag   = f"{detected_sign} ({confidence*100:.0f}%)"
    else:
        color = (120, 120, 120)
        tag   = "Analyzing..."

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.rectangle(frame, (x_min, max(0, y_min - 32)),
                  (x_min + len(tag) * 12 + 10, y_min), color, -1)
    cv2.putText(frame, tag, (x_min + 5, y_min - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

    return frame
